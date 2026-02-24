"""
Tool Engine - Orchestrates tool dispatch and execution.
Central registry for all tools, handles dispatching tool calls to the right tool.
Inspired by Claude Code's ToolEngine & Scheduler.
"""

from typing import Dict, Optional, List, Any
from tools.base_tool import BaseTool, ToolResult, ToolRiskLevel


class ToolEngine:
    """
    Central tool registry and dispatcher.
    
    All tools register here. When the agent loop decides to use a tool,
    it goes through the ToolEngine which:
    1. Validates the tool exists
    2. Checks permissions/risk level
    3. Dispatches to the correct tool
    4. Returns uniform ToolResult
    """

    def __init__(self, auto_approve_reads: bool = True):
        self._tools: Dict[str, BaseTool] = {}
        self._auto_approve_reads = auto_approve_reads
        self._execution_log: List[Dict[str, Any]] = []

    def register(self, tool: BaseTool):
        """Register a tool."""
        self._tools[tool.name] = tool

    def register_all(self, tools: List[BaseTool]):
        """Register multiple tools."""
        for tool in tools:
            self.register(tool)

    def get_tool(self, name: str) -> Optional[BaseTool]:
        """Get a registered tool by name."""
        return self._tools.get(name)

    def list_tools(self) -> List[Dict[str, Any]]:
        """List all registered tools with their schemas."""
        return [tool.get_schema() for tool in self._tools.values()]

    def dispatch(self, tool_name: str, **kwargs) -> ToolResult:
        """
        Dispatch a tool call.
        
        This is the main entry point for tool execution.
        The agentic loop calls this with the tool name and arguments.
        """
        tool = self._tools.get(tool_name)
        if tool is None:
            result = ToolResult(
                tool_name=tool_name,
                success=False,
                output="",
                error=f"Unknown tool: {tool_name}. Available: {list(self._tools.keys())}",
            )
            self._log_execution(tool_name, kwargs, result)
            return result

        # Permission check
        if not self._check_permission(tool):
            result = ToolResult(
                tool_name=tool_name,
                success=False,
                output="",
                error=f"Permission denied for {tool_name} (risk: {tool.risk_level.value})",
            )
            self._log_execution(tool_name, kwargs, result)
            return result

        # Execute
        result = tool.safe_execute(**kwargs)
        self._log_execution(tool_name, kwargs, result)
        return result

    def _check_permission(self, tool: BaseTool) -> bool:
        """Check if tool execution is permitted based on risk level."""
        if self._auto_approve_reads and tool.risk_level == ToolRiskLevel.READ_ONLY:
            return True
        # In a full implementation, this would prompt the user for confirmation
        # For now, auto-approve all tools
        return True

    def _log_execution(self, tool_name: str, args: Dict, result: ToolResult):
        """Log tool execution for audit trail."""
        self._execution_log.append({
            "tool": tool_name,
            "args": {k: str(v)[:200] for k, v in args.items()},
            "success": result.success,
            "duration_ms": result.duration_ms,
            "error": result.error,
        })

    def get_execution_log(self) -> List[Dict[str, Any]]:
        """Get the full execution audit trail."""
        return self._execution_log

    def get_execution_summary(self) -> str:
        """Get a human-readable execution summary."""
        total = len(self._execution_log)
        success = sum(1 for e in self._execution_log if e["success"])
        tools_used = set(e["tool"] for e in self._execution_log)

        parts = [f"Tool Execution Summary: {total} calls ({success} succeeded)"]
        parts.append(f"Tools used: {', '.join(sorted(tools_used))}")

        for entry in self._execution_log:
            status = "✓" if entry["success"] else "✗"
            parts.append(f"  {status} {entry['tool']} ({entry['duration_ms']:.0f}ms)")
            if entry["error"]:
                parts.append(f"    Error: {entry['error'][:100]}")

        return "\n".join(parts)
