"""
Master Agentic Loop (AgenticLoop) - The heart of Code SSG.
Inspired by Claude Code's nO master loop.

Core pattern: while(tool_call) → execute tool → feed results → repeat
Single-threaded, flat message history, transparent audit trail.

The loop integrates scientific verification at each step:
- After code edits: verify the change
- After test runs: verify the results
- After multi-step operations: verify the overall outcome
"""

import time
import json
from typing import Optional, Dict, Any, List, Callable
from enum import Enum

from core.tool_engine import ToolEngine
from core.context_manager import ContextManager, MessageRole
from core.message_queue import MessageQueue, MessageType, QueueMessage
from core.planner import Planner, TaskStatus
from tools.base_tool import ToolResult
from verification.verifier import (
    VerificationMode, ExecutionVerifier, LLMVerifier, HybridVerifier,
    VerificationReport, create_verifier
)


class LoopState(Enum):
    """Current state of the agentic loop."""
    IDLE = "idle"
    GATHERING = "gathering_context"      # Phase 1: Gather context
    ACTING = "taking_action"             # Phase 2: Take action
    VERIFYING = "verifying"              # Phase 3: Verify results
    WAITING_INPUT = "waiting_for_input"
    COMPLETE = "complete"
    ERROR = "error"


class AgenticLoop:
    """
    The master agentic loop.
    
    Implements Claude Code's three-phase workflow:
    1. Gather Context - read files, search, understand the codebase
    2. Take Action - edit files, run commands, make changes
    3. Verify Results - run tests, check output, verify scientifically
    
    Enhanced with EG-CFG's line-by-line verification:
    - Each code change is verified through execution
    - Each statement is checked for scientific correctness
    - Failed verifications trigger automatic retry/revert
    """

    MAX_ITERATIONS = 50  # Safety limit

    def __init__(
        self,
        tool_engine: ToolEngine,
        context_manager: ContextManager,
        message_queue: MessageQueue,
        planner: Planner,
        verification_mode: VerificationMode = VerificationMode.EXECUTION,
        on_tool_call: Optional[Callable] = None,
        on_state_change: Optional[Callable] = None,
        on_verification: Optional[Callable] = None,
    ):
        self.tool_engine = tool_engine
        self.context = context_manager
        self.queue = message_queue
        self.planner = planner
        self.verification_mode = verification_mode

        # Callbacks for UI integration
        self.on_tool_call = on_tool_call
        self.on_state_change = on_state_change
        self.on_verification = on_verification

        # State
        self._state = LoopState.IDLE
        self._iteration = 0
        self._verification_reports: List[VerificationReport] = []
        self._verifier = create_verifier(verification_mode)

    @property
    def state(self) -> LoopState:
        return self._state

    def _set_state(self, new_state: LoopState):
        """Update state and notify callback."""
        old_state = self._state
        self._state = new_state
        if self.on_state_change:
            self.on_state_change(old_state, new_state)

    def run(self, user_input: str) -> str:
        """
        Main entry point. Process a user request through the agentic loop.
        
        Returns the final response text.
        """
        self._set_state(LoopState.GATHERING)
        self._iteration = 0

        # Add user message to context
        self.context.add_user_message(user_input)

        # Determine actions needed (simplified - in production, use LLM)
        actions = self._plan_actions(user_input)

        # Execute the plan
        results = []
        for action in actions:
            if self._iteration >= self.MAX_ITERATIONS:
                results.append("⚠️ Maximum iterations reached. Stopping.")
                break

            self._iteration += 1

            # Check for user interrupts
            interrupt = self._check_interrupt()
            if interrupt:
                results.append(f"📩 User interrupt: {interrupt}")
                # Re-plan based on interrupt
                actions = self._plan_actions(interrupt)
                continue

            # Execute action
            self._set_state(LoopState.ACTING)
            result = self._execute_action(action)
            results.append(result.to_display())

            # Feed result back to context
            self.context.add_tool_result(action.get("tool", ""), result.output)

            # Verify if this was a code-modifying action
            if self._should_verify(action, result):
                self._set_state(LoopState.VERIFYING)
                verification = self._verify_action(action, result)
                if verification:
                    self._verification_reports.append(verification)
                    results.append(verification.to_display())

                    if self.on_verification:
                        self.on_verification(verification)

                    # If verification failed, attempt fix
                    if not verification.is_success():
                        fix_result = self._attempt_fix(action, verification)
                        if fix_result:
                            results.append(fix_result)

            # Update plan if active
            current_task = self.planner.get_current_task()
            if current_task:
                status = "completed" if result.success else "failed"
                self.planner.update_task(current_task.id, status, result.output[:100])
                self.context.add_plan_update(self.planner.get_plan_display())

        self._set_state(LoopState.COMPLETE)

        # Build final response
        final_response = self._build_response(results)
        self.context.add_assistant_message(final_response)

        return final_response

    def run_interactive(self):
        """
        Run in interactive mode (CLI).
        Reads from stdin, processes, and prints results.
        """
        print("🔬 Code SSG - Scientific Statement Generator")
        print("Type your request, or 'quit' to exit.\n")

        while True:
            try:
                user_input = input("You: ").strip()
                if user_input.lower() in ('quit', 'exit', 'q'):
                    print("Goodbye!")
                    break
                if not user_input:
                    continue

                response = self.run(user_input)
                print(f"\nAssistant: {response}\n")

            except KeyboardInterrupt:
                print("\n\nInterrupted. Type 'quit' to exit.")
            except Exception as e:
                print(f"\n❌ Error: {e}\n")

    def _plan_actions(self, user_input: str) -> List[Dict[str, Any]]:
        """
        Plan actions based on user input.
        In production, this uses LLM to generate a tool call sequence.
        Here we implement a rule-based planner for common patterns.
        """
        actions = []
        input_lower = user_input.lower()

        # File viewing
        if any(kw in input_lower for kw in ['view', 'look at', 'show', 'read']):
            # Extract file path (simplified)
            words = user_input.split()
            for w in words:
                if '.' in w and '/' in w or w.endswith('.py') or w.endswith('.js'):
                    actions.append({"tool": "view", "args": {"path": w}})

        # File editing
        if any(kw in input_lower for kw in ['edit', 'fix', 'change', 'replace', 'modify']):
            actions.append({"tool": "edit", "args": {"path": "", "old_str": "", "new_str": ""},
                            "needs_llm": True})

        # Command execution
        if any(kw in input_lower for kw in ['run', 'execute', 'test', 'build']):
            words = user_input.split()
            # Try to extract command
            for i, w in enumerate(words):
                if w in ('run', 'execute'):
                    cmd = " ".join(words[i+1:])
                    if cmd:
                        actions.append({"tool": "bash", "args": {"command": cmd, "label": cmd[:50]}})
                    break

        # Search
        if any(kw in input_lower for kw in ['search', 'find', 'look up']):
            query = user_input.replace('search', '').replace('find', '').replace('look up', '').strip()
            actions.append({"tool": "web_search", "args": {"query": query}})

        # Fetch URL
        if 'http' in input_lower:
            import re
            urls = re.findall(r'https?://\S+', user_input)
            for url in urls:
                actions.append({"tool": "web_fetch", "args": {"url": url}})

        # Default: if no actions identified, return a help message action
        if not actions:
            actions.append({"tool": "_respond", "args": {"message": user_input}})

        return actions

    def _execute_action(self, action: Dict[str, Any]) -> ToolResult:
        """Execute a single action through the tool engine."""
        tool_name = action.get("tool", "")
        args = action.get("args", {})

        # Notify callback
        if self.on_tool_call:
            self.on_tool_call(tool_name, args)

        if tool_name == "_respond":
            return ToolResult(
                tool_name="respond",
                success=True,
                output=f"Understood. Processing: {args.get('message', '')}",
            )

        return self.tool_engine.dispatch(tool_name, **args)

    def _should_verify(self, action: Dict[str, Any], result: ToolResult) -> bool:
        """Determine if this action's result should be scientifically verified."""
        tool_name = action.get("tool", "")
        # Verify after edits, bash commands, and test runs
        return tool_name in ("edit", "bash") and result.success

    def _verify_action(self, action: Dict[str, Any], result: ToolResult) -> Optional[VerificationReport]:
        """
        Run scientific verification on an action's result.
        This is where EG-CFG's execution-guided approach kicks in.
        """
        tool_name = action.get("tool", "")

        if tool_name == "edit":
            # Verify edited file
            path = action.get("args", {}).get("path", "")
            if path and path.endswith('.py'):
                try:
                    with open(path, 'r') as f:
                        code = f.read()
                    return self._verifier.verify(code, path)
                except Exception:
                    return None

        elif tool_name == "bash":
            # Check if the command was a test run
            cmd = action.get("args", {}).get("command", "")
            if any(kw in cmd for kw in ['test', 'pytest', 'unittest']):
                # Verify test results
                report = VerificationReport(
                    file_path="<test_run>",
                    total_statements=1,
                    mode=self.verification_mode,
                )
                from verification.verifier import StatementVerification, VerificationResult as VR
                if result.success:
                    report.passed = 1
                    report.statements.append(StatementVerification(
                        line_number=0,
                        statement=f"Test: {cmd}",
                        result=VR.PASS,
                        details="Tests passed",
                    ))
                else:
                    report.failed = 1
                    report.statements.append(StatementVerification(
                        line_number=0,
                        statement=f"Test: {cmd}",
                        result=VR.FAIL,
                        details=result.error or "Tests failed",
                    ))
                return report

        return None

    def _attempt_fix(self, action: Dict[str, Any], report: VerificationReport) -> Optional[str]:
        """
        Attempt to fix a failed verification.
        Inspired by Claude Code's iterative debug workflow (features #11, #12).
        """
        tool_name = action.get("tool", "")

        if tool_name == "edit":
            path = action.get("args", {}).get("path", "")
            # Try to revert
            edit_tool = self.tool_engine.get_tool("edit")
            if edit_tool and hasattr(edit_tool, 'revert'):
                revert_result = edit_tool.revert(path)
                if revert_result.success:
                    return f"⚠️ Verification failed. Reverted {path} to previous version."
                return f"⚠️ Verification failed but could not revert: {revert_result.error}"

        return f"⚠️ Verification failed. Manual intervention may be needed."

    def _check_interrupt(self) -> Optional[str]:
        """Check for user interrupts in the message queue."""
        msg = self.queue.get(timeout=0.01)
        if msg and msg.type == MessageType.USER_INTERRUPT:
            return msg.content
        return None

    def _build_response(self, results: List[str]) -> str:
        """Build the final response from all action results."""
        parts = []
        for r in results:
            if r:
                parts.append(r)

        if self._verification_reports:
            # Add overall verification summary
            total_passed = sum(r.passed for r in self._verification_reports)
            total_failed = sum(r.failed for r in self._verification_reports)
            parts.append(
                f"\n📊 Verification Summary: {total_passed} passed, {total_failed} failed"
            )

        # Add tool execution summary
        parts.append(f"\n{self.tool_engine.get_execution_summary()}")
        parts.append(f"\n{self.context.get_history_display()}")

        return "\n\n".join(parts) + "\n\nDone."

    def get_verification_reports(self) -> List[VerificationReport]:
        """Get all verification reports from this session."""
        return self._verification_reports
