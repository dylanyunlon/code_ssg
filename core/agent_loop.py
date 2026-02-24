"""
Agent Loop (nO Pattern) - Core agentic loop for Code-SSG.

Inspired by Claude Code's single-threaded master loop architecture:
  while(tool_call) → execute tool → feed results → repeat

The loop integrates Scientific Statement Grounding (SSG) validation
at each code generation step, converting lines to verifiable statements.
"""

import json
import time
import logging
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Callable
from enum import Enum

logger = logging.getLogger(__name__)


class LoopState(Enum):
    """States of the agentic loop."""
    IDLE = "idle"
    PLANNING = "planning"
    EXECUTING = "executing"
    VALIDATING = "validating"
    COMPRESSING = "compressing"
    DONE = "done"
    ERROR = "error"


@dataclass
class ToolCall:
    """Represents a single tool invocation."""
    tool_name: str
    arguments: Dict[str, Any]
    result: Optional[Any] = None
    error: Optional[str] = None
    duration_ms: float = 0.0


@dataclass
class Message:
    """A message in the conversation history."""
    role: str  # "user", "assistant", "tool_result", "system"
    content: str
    tool_calls: List[ToolCall] = field(default_factory=list)
    timestamp: float = field(default_factory=time.time)


@dataclass
class AgentResult:
    """Result of an agent run."""
    code: str
    validation_report: Dict[str, Any]
    tool_calls_log: List[ToolCall]
    total_turns: int
    total_duration_s: float
    context_compressions: int
    sub_agents_spawned: int


@dataclass
class TodoItem:
    """A planning TODO item (inspired by Claude Code's TodoWrite)."""
    id: str
    content: str
    status: str = "pending"  # pending, in_progress, done, blocked
    priority: str = "medium"  # high, medium, low

    def to_dict(self):
        return {
            "id": self.id,
            "content": self.content,
            "status": self.status,
            "priority": self.priority,
        }


class AgenticLoop:
    """
    Main agentic loop implementing the nO pattern from Claude Code.

    Architecture:
    1. User input → Plan (TodoWrite)
    2. Plan → Execute tools in loop
    3. Each code line → SSG validation
    4. Loop until model decides done (no more tool calls)
    5. Context compression at ~92% usage

    Features:
    - View truncated section support
    - Multi-file batch viewing
    - Web search integration
    - Script execution pipeline
    - File edit with diff tracking (+N -M format)
    - Sub-agent spawning for exploration
    """

    CONTEXT_COMPRESSION_THRESHOLD = 0.92  # Trigger at 92% context usage

    def __init__(
        self,
        validator=None,
        model: str = "claude-sonnet-4-5-20250929",
        max_turns: int = 100,
        max_context_tokens: int = 200000,
        tools: Optional[Dict[str, Callable]] = None,
        system_prompt: Optional[str] = None,
    ):
        self.validator = validator
        self.model = model
        self.max_turns = max_turns
        self.max_context_tokens = max_context_tokens
        self.state = LoopState.IDLE
        self.messages: List[Message] = []
        self.todo_list: List[TodoItem] = []
        self.tool_calls_log: List[ToolCall] = []
        self.context_compressions = 0
        self.sub_agents_spawned = 0

        # Register tools
        self.tools = self._register_default_tools()
        if tools:
            self.tools.update(tools)

        # System prompt
        self.system_prompt = system_prompt or self._default_system_prompt()

    def _default_system_prompt(self) -> str:
        return """You are an agentic coding assistant with Scientific Statement Grounding.

For each task, follow this loop:
1. PLAN: Break the task into TODO items
2. EXECUTE: Use tools to implement each TODO
3. VALIDATE: For each code line, generate a scientific statement and verify it
4. VERIFY: Run tests and check results
5. ITERATE: Fix issues and repeat until all TODOs are done

Available tools: view, view_truncated, view_files, search, edit, execute,
web_search, fetch, todo_write, sub_agent

When done, respond with text only (no tool calls) to end the loop."""

    def _register_default_tools(self) -> Dict[str, Callable]:
        """Register the built-in tool set (Claude Code style)."""
        return {
            # File viewing tools
            "view": self._tool_view,
            "view_truncated_section": self._tool_view_truncated,
            "view_files": self._tool_view_files,

            # Search tools
            "search": self._tool_search,
            "web_search": self._tool_web_search,
            "fetch": self._tool_fetch,

            # Edit tools
            "edit": self._tool_edit,
            "str_replace": self._tool_str_replace,

            # Execution tools
            "execute": self._tool_execute,
            "execute_script": self._tool_execute_script,
            "run_commands": self._tool_run_commands,

            # Planning tools
            "todo_write": self._tool_todo_write,

            # Agent tools
            "sub_agent": self._tool_sub_agent,

            # SSG tools
            "validate_statement": self._tool_validate_statement,
        }

    def run(self, user_input: str) -> AgentResult:
        """
        Main entry point - runs the agentic loop until completion.

        This implements the core while(tool_call) pattern:
        1. Send messages to model
        2. If response has tool calls → execute them → feed results → repeat
        3. If response is text only → loop ends
        """
        start_time = time.time()
        self.state = LoopState.PLANNING
        turn = 0

        # Add user message
        self.messages.append(Message(role="user", content=user_input))

        while turn < self.max_turns:
            turn += 1
            logger.info(f"=== Turn {turn}/{self.max_turns} | State: {self.state.value} ===")

            # Check context usage and compress if needed
            if self._estimate_context_usage() > self.CONTEXT_COMPRESSION_THRESHOLD:
                self._compress_context()

            # Get model response (simulated - in production, call Claude API)
            response = self._call_model()

            # Add assistant message
            self.messages.append(Message(
                role="assistant",
                content=response.get("text", ""),
                tool_calls=[ToolCall(
                    tool_name=tc["name"],
                    arguments=tc.get("arguments", {})
                ) for tc in response.get("tool_calls", [])]
            ))

            # Check if loop should end (no tool calls = done)
            if not response.get("tool_calls"):
                self.state = LoopState.DONE
                logger.info("Loop ended - model produced text-only response")
                break

            # Execute each tool call
            self.state = LoopState.EXECUTING
            for tc in response["tool_calls"]:
                tool_result = self._execute_tool(tc["name"], tc.get("arguments", {}))

                # Add tool result to messages
                self.messages.append(Message(
                    role="tool_result",
                    content=json.dumps(tool_result, default=str),
                ))

            # Inject TODO reminder (Claude Code pattern)
            if self.todo_list:
                self._inject_todo_reminder()

        # Build final result
        total_duration = time.time() - start_time
        generated_code = self._extract_generated_code()
        validation_report = self._generate_validation_report()

        return AgentResult(
            code=generated_code,
            validation_report=validation_report,
            tool_calls_log=self.tool_calls_log,
            total_turns=turn,
            total_duration_s=total_duration,
            context_compressions=self.context_compressions,
            sub_agents_spawned=self.sub_agents_spawned,
        )

    def _call_model(self) -> Dict[str, Any]:
        """
        Call the LLM with current message history.

        In production, this calls the Claude API. For now, returns a
        placeholder that simulates the model's decision-making.
        """
        # Build messages payload
        payload = {
            "model": self.model,
            "max_tokens": 4096,
            "system": self.system_prompt,
            "messages": [
                {"role": m.role, "content": m.content}
                for m in self.messages
                if m.role in ("user", "assistant", "tool_result")
            ],
            "tools": self._get_tool_definitions(),
        }

        # TODO: Replace with actual API call
        # response = anthropic_client.messages.create(**payload)
        logger.debug(f"Model call payload size: {len(json.dumps(payload))} bytes")

        # Placeholder response
        return {"text": "Task completed.", "tool_calls": []}

    def _execute_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a tool and track the result."""
        start = time.time()
        tool_call = ToolCall(tool_name=tool_name, arguments=arguments)

        try:
            if tool_name not in self.tools:
                raise ValueError(f"Unknown tool: {tool_name}")

            result = self.tools[tool_name](**arguments)
            tool_call.result = result
            tool_call.duration_ms = (time.time() - start) * 1000

            logger.info(
                f"Tool '{tool_name}' completed in {tool_call.duration_ms:.1f}ms"
            )

        except Exception as e:
            tool_call.error = str(e)
            tool_call.duration_ms = (time.time() - start) * 1000
            logger.error(f"Tool '{tool_name}' failed: {e}")
            result = {"error": str(e)}

        self.tool_calls_log.append(tool_call)
        return result

    # ===== Tool Implementations =====

    def _tool_view(self, path: str, view_range: Optional[List[int]] = None) -> Dict:
        """View a file with optional line range. Supports truncation detection."""
        try:
            with open(path, "r") as f:
                lines = f.readlines()

            total_lines = len(lines)
            if view_range:
                start, end = view_range
                if end == -1:
                    end = total_lines
                selected = lines[start - 1:end]
                truncated = end < total_lines
            else:
                # Auto-truncate if too long (>500 lines)
                if total_lines > 500:
                    selected = lines[:250] + ["\n... [TRUNCATED] ...\n"] + lines[-250:]
                    truncated = True
                else:
                    selected = lines
                    truncated = False

            content = "".join(selected)
            return {
                "content": content,
                "total_lines": total_lines,
                "truncated": truncated,
                "path": path,
            }
        except FileNotFoundError:
            return {"error": f"File not found: {path}"}

    def _tool_view_truncated(
        self, path: str, start_line: int, end_line: int
    ) -> Dict:
        """
        View truncated section of a file.

        This is Feature #2 from the requirements:
        'View truncated section of xxx.py' - allows viewing specific sections
        of files that were previously truncated.
        """
        return self._tool_view(path, view_range=[start_line, end_line])

    def _tool_view_files(self, files: List[str]) -> Dict:
        """
        Batch view multiple files.

        Feature #3: 'Viewed 3 files' - supports viewing multiple files
        in a single tool call for efficiency.
        """
        results = {}
        for file_path in files:
            results[file_path] = self._tool_view(file_path)

        return {
            "files_viewed": len(files),
            "results": results,
        }

    def _tool_search(self, query: str, path: str = ".", regex: bool = False) -> Dict:
        """Search for code patterns using grep/ripgrep."""
        import subprocess
        try:
            cmd = ["grep", "-rn"]
            if regex:
                cmd.append("-E")
            cmd.extend([query, path])

            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=30
            )
            matches = result.stdout.strip().split("\n") if result.stdout.strip() else []
            return {
                "query": query,
                "matches": matches[:50],  # Limit to 50 results
                "total_matches": len(matches),
            }
        except Exception as e:
            return {"error": str(e)}

    def _tool_web_search(self, query: str, max_results: int = 10) -> Dict:
        """
        Web search integration.

        Feature #4: 'Searched the web' with query and N results.
        Returns structured search results with titles and URLs.
        """
        # In production, this calls a search API
        logger.info(f"Web search: '{query}' (max {max_results} results)")
        return {
            "query": query,
            "max_results": max_results,
            "results": [],  # Populated by actual search API
            "status": "placeholder - connect search API",
        }

    def _tool_fetch(self, url: str) -> Dict:
        """
        Fetch web page content.

        Feature #5: 'Fetched: <title>' - fetch and parse web content.
        """
        import urllib.request
        try:
            with urllib.request.urlopen(url, timeout=30) as response:
                content = response.read().decode("utf-8", errors="replace")
            return {
                "url": url,
                "content": content[:10000],  # Limit content size
                "status": response.status,
            }
        except Exception as e:
            return {"url": url, "error": str(e)}

    def _tool_edit(
        self, path: str, old_str: str, new_str: str
    ) -> Dict:
        """
        Edit a file by replacing a string.

        Feature #8-9, #14-15: File editing with diff tracking.
        Returns change summary in '+N -M' format.
        """
        try:
            with open(path, "r") as f:
                original = f.read()

            if old_str not in original:
                return {"error": f"String not found in {path}"}

            new_content = original.replace(old_str, new_str, 1)

            with open(path, "w") as f:
                f.write(new_content)

            # Calculate diff stats
            old_lines = old_str.count("\n") + 1
            new_lines = new_str.count("\n") + 1
            added = max(0, new_lines - old_lines) + new_lines
            removed = max(0, old_lines - new_lines) + old_lines

            return {
                "path": path,
                "changes": f"+{added} -{removed}",
                "status": "success",
            }
        except Exception as e:
            return {"error": str(e)}

    def _tool_str_replace(self, path: str, old_str: str, new_str: str = "") -> Dict:
        """Alias for edit with str_replace semantics."""
        return self._tool_edit(path, old_str, new_str)

    def _tool_execute(self, command: str, timeout: int = 60) -> Dict:
        """
        Execute a single shell command.

        Feature #8, #10-11: Run commands with output capture.
        """
        import subprocess
        try:
            result = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
            return {
                "command": command,
                "stdout": result.stdout[-5000:],  # Last 5000 chars
                "stderr": result.stderr[-2000:],
                "returncode": result.returncode,
            }
        except subprocess.TimeoutExpired:
            return {"command": command, "error": "timeout"}
        except Exception as e:
            return {"command": command, "error": str(e)}

    def _tool_execute_script(self, script: str, description: str = "") -> Dict:
        """
        Execute a multi-line script.

        Feature #6-7: 'Ran N commands' with script description.
        """
        import subprocess
        try:
            result = subprocess.run(
                ["bash", "-c", script],
                capture_output=True,
                text=True,
                timeout=120,
            )
            return {
                "description": description,
                "stdout": result.stdout[-5000:],
                "stderr": result.stderr[-2000:],
                "returncode": result.returncode,
            }
        except Exception as e:
            return {"description": description, "error": str(e)}

    def _tool_run_commands(self, commands: List[Dict[str, str]]) -> Dict:
        """
        Run multiple commands in sequence.

        Feature #6: 'Ran 7 commands' - batch command execution.
        Each command has a 'script' and optional 'description'.
        """
        results = []
        for i, cmd in enumerate(commands):
            script = cmd.get("script", cmd.get("command", ""))
            desc = cmd.get("description", f"Command {i+1}")
            result = self._tool_execute_script(script, desc)
            results.append(result)

            # Stop on failure if requested
            if result.get("returncode", 0) != 0 and cmd.get("stop_on_error", False):
                break

        return {
            "commands_run": len(results),
            "results": results,
        }

    def _tool_todo_write(self, todos: List[Dict[str, str]]) -> Dict:
        """
        Create or update TODO items for planning.

        Inspired by Claude Code's TodoWrite tool.
        """
        for todo_data in todos:
            existing = next(
                (t for t in self.todo_list if t.id == todo_data.get("id")),
                None
            )
            if existing:
                if "status" in todo_data:
                    existing.status = todo_data["status"]
                if "content" in todo_data:
                    existing.content = todo_data["content"]
            else:
                self.todo_list.append(TodoItem(
                    id=todo_data.get("id", f"todo_{len(self.todo_list)+1}"),
                    content=todo_data["content"],
                    status=todo_data.get("status", "pending"),
                    priority=todo_data.get("priority", "medium"),
                ))

        return {
            "total_todos": len(self.todo_list),
            "pending": sum(1 for t in self.todo_list if t.status == "pending"),
            "done": sum(1 for t in self.todo_list if t.status == "done"),
            "todos": [t.to_dict() for t in self.todo_list],
        }

    def _tool_sub_agent(
        self, task: str, subagent_type: str = "explore"
    ) -> Dict:
        """
        Spawn a sub-agent for exploration or specialized tasks.

        Feature: Sub-agent dispatch for tasks like codebase exploration.
        Runs in isolated context to prevent context pollution.
        """
        self.sub_agents_spawned += 1
        logger.info(f"Spawning sub-agent ({subagent_type}): {task[:100]}...")

        # Create a sub-loop with limited context
        sub_loop = AgenticLoop(
            validator=self.validator,
            model=self.model,
            max_turns=20,  # Limited turns for sub-agents
        )
        sub_result = sub_loop.run(task)

        return {
            "subagent_type": subagent_type,
            "task": task,
            "result_summary": sub_result.code[:2000] if sub_result.code else "",
            "turns_used": sub_result.total_turns,
        }

    def _tool_validate_statement(
        self, code_line: str, statement: str, method: str = "hybrid"
    ) -> Dict:
        """
        Validate a scientific statement derived from a code line.

        This is the SSG core - uses the validator to check if the
        statement is scientifically accurate.
        """
        if self.validator is None:
            return {"warning": "No validator configured", "valid": True}

        return self.validator.validate(
            code_line=code_line,
            statement=statement,
            method=method,
        )

    # ===== Internal Methods =====

    def _estimate_context_usage(self) -> float:
        """Estimate current context window usage as a fraction."""
        total_chars = sum(len(m.content) for m in self.messages)
        # Rough estimate: 4 chars per token
        estimated_tokens = total_chars / 4
        return estimated_tokens / self.max_context_tokens

    def _compress_context(self):
        """
        Compress context when approaching limits.

        Inspired by Claude Code's Compressor wU2 that triggers at ~92%.
        Summarizes older messages while keeping recent context.
        """
        self.state = LoopState.COMPRESSING
        self.context_compressions += 1
        logger.info(f"Compressing context (compression #{self.context_compressions})")

        # Keep last 10 messages, summarize the rest
        if len(self.messages) > 15:
            old_messages = self.messages[:-10]
            summary = self._summarize_messages(old_messages)
            self.messages = [
                Message(role="system", content=f"[Context Summary]\n{summary}")
            ] + self.messages[-10:]

    def _summarize_messages(self, messages: List[Message]) -> str:
        """Summarize a list of messages for context compression."""
        tool_calls = []
        key_findings = []

        for m in messages:
            if m.tool_calls:
                for tc in m.tool_calls:
                    tool_calls.append(f"- {tc.tool_name}({', '.join(f'{k}={v}' for k, v in tc.arguments.items())})")
            if m.role == "assistant" and m.content:
                # Keep first 200 chars of each assistant message
                key_findings.append(m.content[:200])

        return (
            f"Previous actions ({len(tool_calls)} tool calls):\n"
            + "\n".join(tool_calls[:20])
            + f"\n\nKey findings:\n"
            + "\n".join(key_findings[:10])
        )

    def _inject_todo_reminder(self):
        """Inject current TODO state as a system reminder (Claude Code pattern)."""
        todo_state = "\n".join(
            f"[{'✓' if t.status == 'done' else '○'}] {t.id}: {t.content} ({t.status})"
            for t in self.todo_list
        )
        self.messages.append(Message(
            role="system",
            content=f"[TODO Reminder]\n{todo_state}"
        ))

    def _extract_generated_code(self) -> str:
        """Extract the final generated code from the conversation."""
        # Look for code blocks in assistant messages (reverse order)
        for m in reversed(self.messages):
            if m.role == "assistant" and "```" in m.content:
                # Extract code between ``` markers
                parts = m.content.split("```")
                for i in range(1, len(parts), 2):
                    code = parts[i]
                    # Remove language identifier
                    if code.startswith("python\n"):
                        code = code[7:]
                    elif code.startswith("py\n"):
                        code = code[3:]
                    return code.strip()
        return ""

    def _generate_validation_report(self) -> Dict[str, Any]:
        """Generate a summary validation report."""
        validations = [
            tc for tc in self.tool_calls_log
            if tc.tool_name == "validate_statement"
        ]
        return {
            "total_validations": len(validations),
            "passed": sum(1 for v in validations if v.result and v.result.get("valid")),
            "failed": sum(1 for v in validations if v.result and not v.result.get("valid")),
            "errors": sum(1 for v in validations if v.error),
        }

    def _get_tool_definitions(self) -> List[Dict]:
        """Get tool definitions in Claude API format."""
        return [
            {
                "name": "view",
                "description": "View a file with optional line range",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string"},
                        "view_range": {
                            "type": "array",
                            "items": {"type": "integer"},
                            "description": "[start_line, end_line], use -1 for end of file",
                        },
                    },
                    "required": ["path"],
                },
            },
            {
                "name": "view_truncated_section",
                "description": "View a truncated section of a file",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string"},
                        "start_line": {"type": "integer"},
                        "end_line": {"type": "integer"},
                    },
                    "required": ["path", "start_line", "end_line"],
                },
            },
            {
                "name": "view_files",
                "description": "Batch view multiple files",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "files": {
                            "type": "array",
                            "items": {"type": "string"},
                        },
                    },
                    "required": ["files"],
                },
            },
            {
                "name": "search",
                "description": "Search for code patterns",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string"},
                        "path": {"type": "string", "default": "."},
                        "regex": {"type": "boolean", "default": False},
                    },
                    "required": ["query"],
                },
            },
            {
                "name": "web_search",
                "description": "Search the web for information",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string"},
                        "max_results": {"type": "integer", "default": 10},
                    },
                    "required": ["query"],
                },
            },
            {
                "name": "fetch",
                "description": "Fetch web page content",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "url": {"type": "string"},
                    },
                    "required": ["url"],
                },
            },
            {
                "name": "edit",
                "description": "Edit a file by replacing text",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string"},
                        "old_str": {"type": "string"},
                        "new_str": {"type": "string"},
                    },
                    "required": ["path", "old_str", "new_str"],
                },
            },
            {
                "name": "execute",
                "description": "Execute a shell command",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "command": {"type": "string"},
                        "timeout": {"type": "integer", "default": 60},
                    },
                    "required": ["command"],
                },
            },
            {
                "name": "execute_script",
                "description": "Execute a multi-line script",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "script": {"type": "string"},
                        "description": {"type": "string", "default": ""},
                    },
                    "required": ["script"],
                },
            },
            {
                "name": "run_commands",
                "description": "Run multiple commands in sequence",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "commands": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "script": {"type": "string"},
                                    "description": {"type": "string"},
                                },
                            },
                        },
                    },
                    "required": ["commands"],
                },
            },
            {
                "name": "todo_write",
                "description": "Create or update TODO items",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "todos": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "id": {"type": "string"},
                                    "content": {"type": "string"},
                                    "status": {"type": "string"},
                                    "priority": {"type": "string"},
                                },
                            },
                        },
                    },
                    "required": ["todos"],
                },
            },
            {
                "name": "sub_agent",
                "description": "Spawn a sub-agent for exploration",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "task": {"type": "string"},
                        "subagent_type": {"type": "string", "default": "explore"},
                    },
                    "required": ["task"],
                },
            },
            {
                "name": "validate_statement",
                "description": "Validate a scientific statement from a code line",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "code_line": {"type": "string"},
                        "statement": {"type": "string"},
                        "method": {
                            "type": "string",
                            "enum": ["code_exec", "llm_judge", "hybrid"],
                            "default": "hybrid",
                        },
                    },
                    "required": ["code_line", "statement"],
                },
            },
        ]
