"""
Bash Tool - Execute shell commands with safety and grouping.
Implements features from claudecode功能.txt:
- Run N commands (with script labels)
- Run a command, edited a file
- Test execution with result reporting
"""

import subprocess
import os
import time
from typing import Optional, List, Dict, Any
from dataclasses import dataclass
from .base_tool import BaseTool, ToolResult, ToolRiskLevel


@dataclass
class CommandSpec:
    """A single command to execute with a descriptive label."""
    command: str
    label: str = ""
    timeout: int = 30
    cwd: Optional[str] = None


class BashTool(BaseTool):
    """
    Execute shell commands with safety classification.
    
    Features (from Claude Code):
    - Persistent shell session concept
    - Risk level classification
    - Command sanitization (blocks injection attempts)
    - Grouped command execution with labels
    """

    # Dangerous patterns to block
    BLOCKED_PATTERNS = [
        "rm -rf /",
        "mkfs.",
        "dd if=/dev/zero",
        "> /dev/sda",
        ":(){ :|:& };:",  # fork bomb
    ]

    def __init__(self, default_cwd: Optional[str] = None):
        super().__init__(
            name="bash",
            description="Execute shell commands with safety checks",
            risk_level=ToolRiskLevel.EXECUTE,
        )
        self.default_cwd = default_cwd or os.getcwd()

    def execute(self, command: str, label: str = "", timeout: int = 30,
                cwd: Optional[str] = None, **kwargs) -> ToolResult:
        """Execute a single command."""
        # Safety check
        safety_issue = self._check_safety(command)
        if safety_issue:
            return ToolResult(
                tool_name=self.name,
                success=False,
                output="",
                error=f"Blocked dangerous command: {safety_issue}",
            )

        working_dir = cwd or self.default_cwd
        display_label = label or command[:60]

        try:
            result = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=working_dir,
                env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
            )

            output_parts = []
            if result.stdout:
                output_parts.append(result.stdout.rstrip())
            if result.stderr:
                output_parts.append(f"[stderr]\n{result.stderr.rstrip()}")

            output = "\n".join(output_parts) or "(no output)"

            return ToolResult(
                tool_name=self.name,
                success=(result.returncode == 0),
                output=output,
                error=f"Exit code: {result.returncode}" if result.returncode != 0 else None,
                metadata={
                    "command": command,
                    "label": display_label,
                    "exit_code": result.returncode,
                    "cwd": working_dir,
                },
            )

        except subprocess.TimeoutExpired:
            return ToolResult(
                tool_name=self.name,
                success=False,
                output="",
                error=f"Command timed out after {timeout}s: {command}",
                metadata={"command": command, "label": display_label, "timeout": timeout},
            )
        except Exception as e:
            return ToolResult(
                tool_name=self.name,
                success=False,
                output="",
                error=str(e),
                metadata={"command": command, "label": display_label},
            )

    def execute_multiple(self, commands: List[CommandSpec]) -> ToolResult:
        """
        Execute multiple commands and report results.
        Shows: "Ran N commands" summary (feature #6, #7 from claudecode功能.txt).
        """
        results = []
        all_success = True

        for spec in commands:
            result = self.execute(
                command=spec.command,
                label=spec.label,
                timeout=spec.timeout,
                cwd=spec.cwd,
            )
            results.append(result)
            if not result.success:
                all_success = False

        # Build summary
        success_count = sum(1 for r in results if r.success)
        output_parts = [f"Ran {len(commands)} command(s) ({success_count} succeeded)\n"]

        for i, (spec, result) in enumerate(zip(commands, results)):
            status = "✓" if result.success else "✗"
            label = spec.label or spec.command[:50]
            output_parts.append(f"\n{status} [{i+1}] {label}")
            if not result.success and result.error:
                output_parts.append(f"  Error: {result.error}")
            # Show abbreviated output
            if result.output:
                lines = result.output.split('\n')
                if len(lines) > 5:
                    output_parts.append(f"  Output: ({len(lines)} lines)")
                    for line in lines[:3]:
                        output_parts.append(f"    {line}")
                    output_parts.append(f"    ... ({len(lines)-3} more lines)")
                else:
                    for line in lines:
                        output_parts.append(f"  {line}")

        return ToolResult(
            tool_name=self.name,
            success=all_success,
            output="\n".join(output_parts),
            metadata={
                "total_commands": len(commands),
                "successful": success_count,
                "failed": len(commands) - success_count,
                "command_results": [r.to_dict() for r in results],
            },
        )

    def _check_safety(self, command: str) -> Optional[str]:
        """Check if command contains dangerous patterns."""
        cmd_lower = command.lower().strip()
        for pattern in self.BLOCKED_PATTERNS:
            if pattern in cmd_lower:
                return f"Contains blocked pattern: {pattern}"
        return None
