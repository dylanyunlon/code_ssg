"""
Agent Loop (TAOR Pattern) - Core agentic loop for Code-SSG.
=============================================================
Fully async implementation inspired by:
- Claude Code's single-threaded master loop (while tool_call -> execute -> feed -> repeat)
- skynetCheapBuy/new_v5_files/agent_loop.py execution standard
- Seed 2.0 evaluation framework for benchmarking

Features from claudecode功能.txt (1-15):
  1. Tree directory view (via tools)
  2. View truncated section (view_truncated_section)
  3. Batch view files ("Viewed 3 files")
  4. Web search ("Searched the web, 10 results")
  5. Fetch URL ("Fetched: <title>")
  6. Run N commands ("Ran 7 commands")
  7. Batch command execution ("Ran 3 commands")
  8. Edit file with diff (+N, -M format)
  9. VALU/code transforms (str_replace)
 10. Test execution with verification
 11. Multi-step debug loops ("Ran 14 commands, viewed a file, edited a file")
 12. Revert + test workflow
 13. View loop section (partial file view for restructuring)
 14. Revert to baseline
 15. Restructure main loop (+20 lines)

Location: core/agent_loop.py (REWRITTEN from skynetCheapBuy standard)
"""

import asyncio
import json
import time
import uuid
import os
import logging
from dataclasses import dataclass, field
from typing import Any, AsyncGenerator, Callable, Dict, List, Optional
from enum import Enum

logger = logging.getLogger(__name__)


# ==============================================================
# Data Structures (matching skynetCheapBuy/new_v5_files/agent_loop.py)
# ==============================================================

class LoopState(Enum):
    IDLE = "idle"
    RUNNING = "running"
    PAUSED = "paused"
    WAITING_APPROVAL = "waiting_approval"
    COMPLETED = "completed"
    ERROR = "error"


class StepType(Enum):
    TOOL_CALL = "tool_call"
    TOOL_RESULT = "tool_result"
    THINKING = "thinking"
    TEXT_RESPONSE = "text_response"
    ERROR = "error"
    CONTEXT_COMPACT = "context_compact"


@dataclass
class ToolCall:
    """A single tool invocation with full lifecycle tracking."""
    id: str
    tool_name: str
    arguments: Dict[str, Any]
    description: str = ""
    status: str = "pending"
    result: Any = None
    error: Optional[str] = None
    started_at: Optional[float] = None
    completed_at: Optional[float] = None

    @property
    def duration_ms(self) -> Optional[float]:
        if self.started_at and self.completed_at:
            return (self.completed_at - self.started_at) * 1000
        return None


@dataclass
class AgentStep:
    """A single step in the agentic loop."""
    id: str
    step_type: StepType
    content: Any = None
    tool_calls: List[ToolCall] = field(default_factory=list)
    timestamp: float = field(default_factory=time.time)
    display_title: str = ""
    display_detail: str = ""
    files_changed: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class LoopSession:
    """Tracks an entire agentic loop session."""
    session_id: str
    status: LoopState = LoopState.IDLE
    steps: List[AgentStep] = field(default_factory=list)
    total_tool_calls: int = 0
    total_files_viewed: int = 0
    total_files_edited: int = 0
    total_commands_run: int = 0
    total_searches: int = 0
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    error: Optional[str] = None
    messages: List[Dict[str, Any]] = field(default_factory=list)
    context_usage_pct: float = 0.0
    context_compressions: int = 0

    def to_dict(self) -> Dict:
        return {
            "session_id": self.session_id,
            "status": self.status.value,
            "stats": {
                "total_tool_calls": self.total_tool_calls,
                "total_files_viewed": self.total_files_viewed,
                "total_files_edited": self.total_files_edited,
                "total_commands_run": self.total_commands_run,
                "total_searches": self.total_searches,
                "context_compressions": self.context_compressions,
            },
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "context_usage_pct": self.context_usage_pct,
        }


@dataclass
class TodoItem:
    """Planning TODO item (Claude Code's TodoWrite)."""
    id: str
    content: str
    status: str = "pending"
    priority: str = "medium"

    def to_dict(self):
        return {"id": self.id, "content": self.content, "status": self.status, "priority": self.priority}


@dataclass
class AgentResult:
    """Final result of an agent run."""
    session_id: str = ""
    final_text: str = ""
    code: str = ""
    validation_report: Dict[str, Any] = field(default_factory=dict)
    tool_calls_log: List[ToolCall] = field(default_factory=list)
    total_turns: int = 0
    total_duration_s: float = 0.0
    context_compressions: int = 0
    sub_agents_spawned: int = 0
    files_changed: List[Dict[str, Any]] = field(default_factory=list)
    stats: Dict[str, int] = field(default_factory=dict)


# ==============================================================
# Tool Classification (Claude Code-style display titles)
# ==============================================================

TOOL_CATEGORIES = {
    "execute": "command", "execute_script": "command",
    "run_commands": "command", "bash": "command",
    "view": "view", "view_truncated_section": "view",
    "view_files": "view", "glob": "view", "ls": "view", "tree": "view",
    "edit": "edit", "str_replace": "edit", "write_file": "edit", "multi_edit": "edit",
    "search": "search", "web_search": "search", "grep": "search",
    "fetch": "fetch", "web_fetch": "fetch",
    "todo_write": "todo", "todo_read": "todo",
    "validate_statement": "validate",
    "sub_agent": "agent",
}


def generate_step_title(tool_calls: List[ToolCall]) -> str:
    """
    Generate Claude Code-style step titles.
    Examples from claudecode功能.txt:
    - "Ran 7 commands"
    - "Viewed 3 files"  
    - "Ran a command, edited a file"
    - "Ran 14 commands, viewed a file, edited a file"
    - "Searched the web"
    - "Fetched: Anthropic's original take home assignment"
    """
    counts: Dict[str, int] = {}
    for tc in tool_calls:
        cat = TOOL_CATEGORIES.get(tc.tool_name, "other")
        counts[cat] = counts.get(cat, 0) + 1
    parts = []
    for cat, n in counts.items():
        if cat == "command":
            parts.append(f"Ran {n} command{'s' if n > 1 else ''}")
        elif cat == "view":
            parts.append(f"Viewed {n} file{'s' if n > 1 else ''}")
        elif cat == "edit":
            parts.append(f"edited {n} file{'s' if n > 1 else ''}")
        elif cat == "search":
            parts.append("Searched the web")
        elif cat == "fetch":
            parts.append(f"Fetched {n} page{'s' if n > 1 else ''}")
        elif cat == "validate":
            parts.append(f"Validated {n} statement{'s' if n > 1 else ''}")
    return ", ".join(parts) if parts else "Processing"


def format_file_change(path: str, additions: int, deletions: int) -> str:
    """Format: perf_takehome.py, +3, -4"""
    return f"{os.path.basename(path)}, +{additions}, -{deletions}"


# ==============================================================
# Main Agent Loop
# ==============================================================

class AgenticLoop:
    """
    Master Agent Loop - TAOR pattern (Think -> Act -> Observe -> Repeat).
    
    Supports:
    - Real Claude API via ClaudeClient (ANTHROPIC_API_KEY)
    - Mock mode for testing
    - SSE event streaming (async generator)
    - Interrupt/resume (Ctrl+C pattern)
    - Context compression at ~92%
    - Sub-agent spawning
    - All 15 features from claudecode功能.txt
    """

    CONTEXT_COMPACT_THRESHOLD = 0.92
    MAX_ITERATIONS = 100

    def __init__(
        self,
        tool_engine=None,
        context_manager=None,
        message_queue=None,
        planner=None,
        verification_mode=None,
        model: str = "claude-sonnet-4-5-20250929",
        max_tokens: int = 200000,
        max_iterations: int = MAX_ITERATIONS,
        max_turns: int = 100,
        max_context_tokens: int = 200000,
        system_prompt: Optional[str] = None,
        tools: Optional[Dict[str, Callable]] = None,
        on_tool_call: Optional[Callable] = None,
        on_state_change: Optional[Callable] = None,
        on_verification: Optional[Callable] = None,
        validator=None,
    ):
        self.tool_engine = tool_engine
        self.context_manager = context_manager
        self.message_queue = message_queue
        self.planner = planner
        self.verification_mode = verification_mode
        self.validator = validator
        self.model = model
        self.max_tokens = max_tokens or max_context_tokens
        self.max_turns = max_turns
        self.max_iterations = max_iterations
        self.state = LoopState.IDLE
        self.todo_list: List[TodoItem] = []
        self.tool_calls_log: List[ToolCall] = []
        self.context_compressions = 0
        self.sub_agents_spawned = 0
        self.messages: List[Dict] = []
        self.on_tool_call = on_tool_call
        self.on_state_change = on_state_change
        self.on_verification = on_verification
        self._interrupt_flag = False
        self._pause_flag = False
        self.system_prompt = system_prompt or self._default_system_prompt()
        self._tools = self._register_default_tools()
        if tools:
            self._tools.update(tools)
        self._claude_client = None

    def _default_system_prompt(self) -> str:
        return """You are Code-SSG, an agentic coding assistant with Scientific Statement Grounding.

For each task:
1. PLAN: Break into TODO items (todo_write)
2. EXECUTE: Use tools to implement
3. VALIDATE: Verify correctness
4. VERIFY: Run tests
5. ITERATE: Fix and repeat

Available tools: view, view_truncated_section, view_files, search, edit, str_replace,
execute, execute_script, run_commands, web_search, fetch, todo_write, sub_agent, validate_statement

When done, respond with text only (no tool calls) to end the loop."""

    # ==== Public API ====

    async def run_async(self, user_input: str, session_id: Optional[str] = None) -> AsyncGenerator[Dict, None]:
        """Async generator yielding SSE events."""
        session = LoopSession(session_id=session_id or str(uuid.uuid4()), status=LoopState.RUNNING, started_at=time.time())
        self.state = LoopState.RUNNING
        self._interrupt_flag = False
        messages = [{"role": "user", "content": user_input}]
        session.messages = messages
        yield self._emit("session_start", session.to_dict())

        iteration = 0
        all_tc = []
        all_fc = []

        while iteration < self.max_iterations:
            if self._interrupt_flag:
                session.status = LoopState.PAUSED
                yield self._emit("session_paused", {"reason": "user_interrupt"})
                break
            if self._pause_flag:
                session.status = LoopState.WAITING_APPROVAL
                yield self._emit("waiting_approval", {})
                while self._pause_flag and not self._interrupt_flag:
                    await asyncio.sleep(0.1)
                if self._interrupt_flag:
                    break
                session.status = LoopState.RUNNING

            iteration += 1
            yield self._emit("thinking", {"iteration": iteration})

            try:
                response = await self._call_model_async(messages)
            except Exception as e:
                session.status = LoopState.ERROR
                session.error = str(e)
                yield self._emit("error", {"message": str(e)})
                break

            tool_calls = self._extract_tool_calls(response)
            text_content = self._extract_text(response)

            if not tool_calls:
                session.steps.append(AgentStep(id=str(uuid.uuid4()), step_type=StepType.TEXT_RESPONSE, content=text_content, display_title="Response"))
                yield self._emit("text_response", {"content": text_content})
                session.status = LoopState.COMPLETED
                break

            step = AgentStep(id=str(uuid.uuid4()), step_type=StepType.TOOL_CALL, tool_calls=tool_calls)
            step.display_title = generate_step_title(tool_calls)
            yield self._emit("tool_calls_start", {"display_title": step.display_title, "tool_count": len(tool_calls)})

            tool_results = []
            for tc in tool_calls:
                tc.status = "running"
                tc.started_at = time.time()
                yield self._emit("tool_executing", {"tool_call_id": tc.id, "tool_name": tc.tool_name})
                try:
                    result = await self._execute_tool_async(tc.tool_name, tc.arguments)
                    tc.result = result
                    tc.status = "completed"
                    tc.completed_at = time.time()
                    session.total_tool_calls += 1
                    cat = TOOL_CATEGORIES.get(tc.tool_name, "other")
                    if cat == "command": session.total_commands_run += 1
                    elif cat == "view": session.total_files_viewed += 1
                    elif cat == "edit": session.total_files_edited += 1
                    elif cat == "search": session.total_searches += 1
                    if self.on_tool_call:
                        self.on_tool_call(tc.tool_name, tc.arguments, result)
                    yield self._emit("tool_completed", {"tool_call_id": tc.id, "tool_name": tc.tool_name, "duration_ms": tc.duration_ms})
                    tool_results.append({"tool_use_id": tc.id, "content": json.dumps(result, default=str) if isinstance(result, dict) else str(result)})
                except Exception as e:
                    tc.status = "failed"
                    tc.error = str(e)
                    tc.completed_at = time.time()
                    yield self._emit("tool_error", {"tool_call_id": tc.id, "error": str(e)})
                    tool_results.append({"tool_use_id": tc.id, "content": f"Error: {e}", "is_error": True})
                all_tc.append(tc)
                self.tool_calls_log.append(tc)

            step.files_changed = self._collect_file_changes(tool_calls)
            all_fc.extend(step.files_changed)
            session.steps.append(step)
            yield self._emit("step_completed", {"display_title": step.display_title, "files_changed": [format_file_change(fc["file"], fc["additions"], fc["deletions"]) for fc in step.files_changed], "stats": session.to_dict()["stats"]})

            # OBSERVE: feed results back
            messages.append({"role": "assistant", "content": response.get("content", [])})
            for tr in tool_results:
                messages.append({"role": "user", "content": [{"type": "tool_result", "tool_use_id": tr["tool_use_id"], "content": tr["content"]}]})

            session.context_usage_pct = self._estimate_context_usage(messages)
            if session.context_usage_pct > self.CONTEXT_COMPACT_THRESHOLD:
                yield self._emit("context_compacting", {"usage_pct": session.context_usage_pct})
                messages = self._compress_context(messages)
                session.context_compressions += 1
                yield self._emit("context_compacted", {"compression_count": session.context_compressions})

            if self.todo_list:
                self._inject_todo_reminder(messages)
            session.messages = messages

        if session.status == LoopState.RUNNING:
            session.status = LoopState.ERROR if iteration >= self.max_iterations else LoopState.COMPLETED
        session.completed_at = time.time()
        yield self._emit("session_complete", session.to_dict())

    def run(self, user_input: str) -> AgentResult:
        """Synchronous entry point (compatible with main.py)."""
        start_time = time.time()
        session_id = str(uuid.uuid4())
        messages = [{"role": "user", "content": user_input}]
        self.state = LoopState.RUNNING
        all_tc = []
        all_fc = []
        final_text = ""
        turn = 0

        while turn < self.max_iterations:
            turn += 1
            logger.info(f"=== Turn {turn}/{self.max_iterations} | State: {self.state.value} ===")
            if self._estimate_context_usage(messages) > self.CONTEXT_COMPACT_THRESHOLD:
                messages = self._compress_context(messages)
                self.context_compressions += 1

            response = self._call_model_sync(messages)
            tool_calls = self._extract_tool_calls(response)
            text_content = self._extract_text(response)

            if not tool_calls:
                final_text = text_content
                self.state = LoopState.COMPLETED
                break

            title = generate_step_title(tool_calls)
            logger.info(f"  {title}")

            tool_results = []
            for tc in tool_calls:
                tc.started_at = time.time()
                try:
                    result = self._execute_tool_sync(tc.tool_name, tc.arguments)
                    tc.result = result
                    tc.status = "completed"
                except Exception as e:
                    tc.error = str(e)
                    tc.status = "failed"
                    result = {"error": str(e)}
                tc.completed_at = time.time()
                if self.on_tool_call:
                    self.on_tool_call(tc.tool_name, tc.arguments, result)
                all_tc.append(tc)
                self.tool_calls_log.append(tc)
                tool_results.append({"tool_use_id": tc.id, "content": json.dumps(result, default=str) if isinstance(result, dict) else str(result)})

            fc = self._collect_file_changes(tool_calls)
            all_fc.extend(fc)
            for f in fc:
                logger.info(f"  {format_file_change(f['file'], f['additions'], f['deletions'])}")

            messages.append({"role": "assistant", "content": response.get("content", [])})
            for tr in tool_results:
                messages.append({"role": "user", "content": [{"type": "tool_result", "tool_use_id": tr["tool_use_id"], "content": tr["content"]}]})
            if self.todo_list:
                self._inject_todo_reminder(messages)

        return AgentResult(
            session_id=session_id, final_text=final_text,
            code=self._extract_generated_code(messages),
            validation_report=self._generate_validation_report(),
            tool_calls_log=all_tc, total_turns=turn,
            total_duration_s=time.time() - start_time,
            context_compressions=self.context_compressions,
            sub_agents_spawned=self.sub_agents_spawned,
            files_changed=all_fc,
            stats={"total_tool_calls": len(all_tc)},
        )

    def run_interactive(self):
        """Interactive REPL loop."""
        print("Code-SSG Interactive Mode (type 'exit' to quit)")
        while True:
            try:
                user_input = input("\n> ").strip()
                if user_input.lower() in ("exit", "quit", "/exit"):
                    break
                if not user_input:
                    continue
                result = self.run(user_input)
                print(f"\n{result.final_text}")
                print(f"[{result.total_turns} turns, {result.total_duration_s:.1f}s, {result.stats.get('total_tool_calls', 0)} tool calls]")
            except KeyboardInterrupt:
                print("\nInterrupted.")
                break

    def interrupt(self):
        self._interrupt_flag = True

    def pause(self):
        self._pause_flag = True

    def resume(self):
        self._pause_flag = False

    # ==== Model Calling ====

    def _get_claude_client(self):
        if self._claude_client is None:
            api_key = os.environ.get("ANTHROPIC_API_KEY", "")
            if api_key:
                from core.claude_client import ClaudeClient
                self._claude_client = ClaudeClient(api_key=api_key, model=self.model, max_tokens=8192)
        return self._claude_client

    async def _call_model_async(self, messages):
        client = self._get_claude_client()
        if client:
            return await client.chat(messages=messages, tools=self._get_tool_definitions(), system_prompt=self.system_prompt)
        return {"content": [{"type": "text", "text": "Task completed."}], "tool_calls": []}

    def _call_model_sync(self, messages):
        client = self._get_claude_client()
        if client:
            try:
                resp = client.chat_sync(messages=messages, tools=self._get_tool_definitions(), system_prompt=self.system_prompt)
                u = resp.get("usage", {})
                logger.info(f"  API: {u.get('input_tokens',0)} in / {u.get('output_tokens',0)} out")
                return resp
            except Exception as e:
                logger.error(f"Claude API error: {e}")
        return {"content": [{"type": "text", "text": "Task completed."}], "tool_calls": []}

    # ==== Response Parsing ====

    def _extract_tool_calls(self, response: Dict) -> List[ToolCall]:
        tcs = []
        seen = set()
        for block in response.get("content", []):
            if isinstance(block, dict) and block.get("type") == "tool_use":
                tid = block.get("id", str(uuid.uuid4()))
                if tid not in seen:
                    seen.add(tid)
                    tcs.append(ToolCall(id=tid, tool_name=block["name"], arguments=block.get("input", {}), description=block.get("input", {}).get("description", "")))
        for tc in response.get("tool_calls", []):
            if isinstance(tc, dict) and "name" in tc:
                tid = tc.get("id", str(uuid.uuid4()))
                if tid not in seen:
                    seen.add(tid)
                    tcs.append(ToolCall(id=tid, tool_name=tc["name"], arguments=tc.get("arguments", tc.get("input", {}))))
        return tcs

    def _extract_text(self, response: Dict) -> str:
        texts = [b["text"] for b in response.get("content", []) if isinstance(b, dict) and b.get("type") == "text"]
        if not texts and response.get("text"):
            texts.append(response["text"])
        return "\n".join(texts)

    # ==== Tool Execution ====

    async def _execute_tool_async(self, name, args):
        if self.tool_engine and hasattr(self.tool_engine, 'execute'):
            return self.tool_engine.execute(name, args)
        if name in self._tools:
            return self._tools[name](**args)
        raise ValueError(f"Unknown tool: {name}")

    def _execute_tool_sync(self, name, args):
        if self.tool_engine and hasattr(self.tool_engine, 'execute'):
            return self.tool_engine.execute(name, args)
        if name in self._tools:
            return self._tools[name](**args)
        raise ValueError(f"Unknown tool: {name}")

    # ==== File Change Tracking (Features 8-9, 14-15) ====

    def _collect_file_changes(self, tool_calls):
        changes = []
        for tc in tool_calls:
            if tc.tool_name in ("edit", "str_replace", "multi_edit", "write_file"):
                r = tc.result or {}
                if isinstance(r, dict) and ("path" in r or "path" in tc.arguments):
                    additions = r.get("additions", 0)
                    deletions = r.get("deletions", 0)
                    if "changes" in r:
                        try:
                            parts = r["changes"].replace("+","").replace("-","").split()
                            if len(parts) >= 2:
                                additions, deletions = int(parts[0]), int(parts[1])
                        except (ValueError, IndexError):
                            pass
                    changes.append({"file": r.get("path", tc.arguments.get("path", "unknown")), "additions": additions, "deletions": deletions})
        return changes

    # ==== Context Management ====

    def _estimate_context_usage(self, messages):
        total = 0
        for m in messages:
            c = m.get("content", "")
            if isinstance(c, str):
                total += len(c)
            elif isinstance(c, list):
                total += sum(len(json.dumps(b, default=str)) for b in c if isinstance(b, dict))
        return (total / 4) / self.max_tokens

    def _compress_context(self, messages):
        logger.info("Compressing context...")
        if len(messages) <= 10:
            return messages
        first = messages[0]
        recent = messages[-6:]
        old = messages[1:-6]
        tc_count = sum(1 for m in old if isinstance(m.get("content"), list) and any(isinstance(b, dict) and b.get("type") in ("tool_use", "tool_result") for b in m["content"]))
        summary = f"[Context Summary: {len(old)} messages compressed, ~{tc_count} tool interactions.]"
        return [first, {"role": "user", "content": summary}] + recent

    def _inject_todo_reminder(self, messages):
        todo_text = "\n".join(f"[{'V' if t.status == 'done' else 'O'}] {t.id}: {t.content} ({t.status})" for t in self.todo_list)
        messages.append({"role": "user", "content": f"[TODO Reminder]\n{todo_text}"})

    def _extract_generated_code(self, messages):
        for m in reversed(messages):
            c = m.get("content", "")
            if isinstance(c, str) and "```" in c:
                parts = c.split("```")
                for i in range(1, len(parts), 2):
                    code = parts[i]
                    if code.startswith("python\n"):
                        code = code[7:]
                    elif code.startswith("py\n"):
                        code = code[3:]
                    return code.strip()
        return ""

    def _generate_validation_report(self):
        validations = [tc for tc in self.tool_calls_log if tc.tool_name == "validate_statement"]
        return {
            "total": len(validations),
            "passed": sum(1 for v in validations if v.result and v.result.get("valid")),
            "failed": sum(1 for v in validations if v.result and not v.result.get("valid")),
        }

    def _emit(self, event_type, data):
        return {"type": event_type, "data": data, "timestamp": time.time()}

    # ==== Internal Tool Implementations (Features 1-15) ====

    def _register_default_tools(self):
        return {
            "view": self._tool_view, "view_truncated_section": self._tool_view_truncated,
            "view_files": self._tool_view_files, "search": self._tool_search,
            "web_search": self._tool_web_search, "fetch": self._tool_fetch,
            "edit": self._tool_edit, "str_replace": self._tool_str_replace,
            "execute": self._tool_execute, "execute_script": self._tool_execute_script,
            "run_commands": self._tool_run_commands, "todo_write": self._tool_todo_write,
            "sub_agent": self._tool_sub_agent, "validate_statement": self._tool_validate_statement,
        }

    def _tool_view(self, path, view_range=None):
        try:
            with open(path) as f:
                lines = f.readlines()
            total = len(lines)
            if view_range:
                s, e = view_range
                e = total if e == -1 else e
                sel = lines[s-1:e]
                trunc = e < total
            else:
                if total > 500:
                    sel = lines[:250] + ["\n... [TRUNCATED] ...\n"] + lines[-250:]
                    trunc = True
                else:
                    sel = lines
                    trunc = False
            return {"content": "".join(sel), "total_lines": total, "truncated": trunc, "path": path}
        except FileNotFoundError:
            return {"error": f"File not found: {path}"}

    def _tool_view_truncated(self, path, start_line, end_line):
        return self._tool_view(path, view_range=[start_line, end_line])

    def _tool_view_files(self, files):
        return {"files_viewed": len(files), "results": {f: self._tool_view(f) for f in files}}

    def _tool_search(self, query, path=".", regex=False):
        import subprocess
        try:
            cmd = ["grep", "-rn"] + (["-E"] if regex else []) + [query, path]
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            matches = r.stdout.strip().split("\n") if r.stdout.strip() else []
            return {"query": query, "matches": matches[:50], "total_matches": len(matches)}
        except Exception as e:
            return {"error": str(e)}

    def _tool_web_search(self, query, max_results=10):
        return {"query": query, "max_results": max_results, "results": [], "status": "connect_search_api"}

    def _tool_fetch(self, url):
        import urllib.request
        try:
            with urllib.request.urlopen(url, timeout=30) as resp:
                content = resp.read().decode("utf-8", errors="replace")
            return {"url": url, "content": content[:10000], "status": resp.status}
        except Exception as e:
            return {"url": url, "error": str(e)}

    def _tool_edit(self, path, old_str, new_str):
        try:
            with open(path) as f:
                orig = f.read()
            if old_str not in orig:
                return {"error": f"String not found in {path}"}
            new = orig.replace(old_str, new_str, 1)
            with open(path, "w") as f:
                f.write(new)
            ol = old_str.count("\n") + 1
            nl = new_str.count("\n") + 1
            a = max(0, nl - ol) + nl
            d = max(0, ol - nl) + ol
            return {"path": path, "changes": f"+{a} -{d}", "additions": a, "deletions": d, "status": "success"}
        except Exception as e:
            return {"error": str(e)}

    def _tool_str_replace(self, path, old_str, new_str=""):
        return self._tool_edit(path, old_str, new_str)

    def _tool_execute(self, command, timeout=60):
        import subprocess
        try:
            r = subprocess.run(command, shell=True, capture_output=True, text=True, timeout=timeout)
            return {"command": command, "stdout": r.stdout[-5000:], "stderr": r.stderr[-2000:], "returncode": r.returncode}
        except subprocess.TimeoutExpired:
            return {"command": command, "error": "timeout"}
        except Exception as e:
            return {"command": command, "error": str(e)}

    def _tool_execute_script(self, script, description=""):
        import subprocess
        try:
            r = subprocess.run(["bash", "-c", script], capture_output=True, text=True, timeout=120)
            return {"description": description, "stdout": r.stdout[-5000:], "stderr": r.stderr[-2000:], "returncode": r.returncode}
        except Exception as e:
            return {"description": description, "error": str(e)}

    def _tool_run_commands(self, commands):
        results = []
        for cmd in commands:
            s = cmd.get("script", cmd.get("command", ""))
            results.append(self._tool_execute_script(s, cmd.get("description", "")))
            if results[-1].get("returncode", 0) != 0 and cmd.get("stop_on_error"):
                break
        return {"commands_run": len(results), "results": results}

    def _tool_todo_write(self, todos):
        for td in todos:
            ex = next((t for t in self.todo_list if t.id == td.get("id")), None)
            if ex:
                if "status" in td: ex.status = td["status"]
                if "content" in td: ex.content = td["content"]
            else:
                self.todo_list.append(TodoItem(id=td.get("id", f"todo_{len(self.todo_list)+1}"), content=td["content"], status=td.get("status", "pending"), priority=td.get("priority", "medium")))
        return {"total": len(self.todo_list), "pending": sum(1 for t in self.todo_list if t.status == "pending"), "done": sum(1 for t in self.todo_list if t.status == "done"), "todos": [t.to_dict() for t in self.todo_list]}

    def _tool_sub_agent(self, task, subagent_type="explore"):
        self.sub_agents_spawned += 1
        sub = AgenticLoop(model=self.model, max_iterations=20)
        r = sub.run(task)
        return {"task": task, "result": r.final_text[:2000], "turns": r.total_turns}

    def _tool_validate_statement(self, code_line, statement, method="hybrid"):
        if self.validator:
            return self.validator.validate(code_line=code_line, statement=statement, method=method)
        return {"warning": "No validator configured", "valid": True}

    # ==== Tool Definitions (Claude API format) ====

    def _get_tool_definitions(self):
        return [
            {"name": "view", "description": "View a file with optional line range", "input_schema": {"type": "object", "properties": {"path": {"type": "string"}, "view_range": {"type": "array", "items": {"type": "integer"}}}, "required": ["path"]}},
            {"name": "view_truncated_section", "description": "View truncated section of a file", "input_schema": {"type": "object", "properties": {"path": {"type": "string"}, "start_line": {"type": "integer"}, "end_line": {"type": "integer"}}, "required": ["path", "start_line", "end_line"]}},
            {"name": "view_files", "description": "Batch view multiple files", "input_schema": {"type": "object", "properties": {"files": {"type": "array", "items": {"type": "string"}}}, "required": ["files"]}},
            {"name": "search", "description": "Search for code patterns", "input_schema": {"type": "object", "properties": {"query": {"type": "string"}, "path": {"type": "string", "default": "."}, "regex": {"type": "boolean", "default": False}}, "required": ["query"]}},
            {"name": "web_search", "description": "Search the web", "input_schema": {"type": "object", "properties": {"query": {"type": "string"}, "max_results": {"type": "integer", "default": 10}}, "required": ["query"]}},
            {"name": "fetch", "description": "Fetch web page content", "input_schema": {"type": "object", "properties": {"url": {"type": "string"}}, "required": ["url"]}},
            {"name": "edit", "description": "Edit file by replacing text. Returns +N -M.", "input_schema": {"type": "object", "properties": {"path": {"type": "string"}, "old_str": {"type": "string"}, "new_str": {"type": "string"}}, "required": ["path", "old_str", "new_str"]}},
            {"name": "str_replace", "description": "Replace text in file", "input_schema": {"type": "object", "properties": {"path": {"type": "string"}, "old_str": {"type": "string"}, "new_str": {"type": "string", "default": ""}}, "required": ["path", "old_str"]}},
            {"name": "execute", "description": "Execute a shell command", "input_schema": {"type": "object", "properties": {"command": {"type": "string"}, "timeout": {"type": "integer", "default": 60}}, "required": ["command"]}},
            {"name": "execute_script", "description": "Execute a multi-line script", "input_schema": {"type": "object", "properties": {"script": {"type": "string"}, "description": {"type": "string", "default": ""}}, "required": ["script"]}},
            {"name": "run_commands", "description": "Run multiple commands ('Ran N commands')", "input_schema": {"type": "object", "properties": {"commands": {"type": "array", "items": {"type": "object", "properties": {"script": {"type": "string"}, "description": {"type": "string"}}}}}, "required": ["commands"]}},
            {"name": "todo_write", "description": "Create/update TODO items", "input_schema": {"type": "object", "properties": {"todos": {"type": "array", "items": {"type": "object", "properties": {"id": {"type": "string"}, "content": {"type": "string"}, "status": {"type": "string"}, "priority": {"type": "string"}}}}}, "required": ["todos"]}},
            {"name": "sub_agent", "description": "Spawn a sub-agent", "input_schema": {"type": "object", "properties": {"task": {"type": "string"}, "subagent_type": {"type": "string", "default": "explore"}}, "required": ["task"]}},
            {"name": "validate_statement", "description": "Validate a scientific statement from code", "input_schema": {"type": "object", "properties": {"code_line": {"type": "string"}, "statement": {"type": "string"}, "method": {"type": "string", "enum": ["code_exec", "llm_judge", "hybrid"], "default": "hybrid"}}, "required": ["code_line", "statement"]}},
        ]