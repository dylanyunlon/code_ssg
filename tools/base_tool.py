"""
Base tool interface for Code SSG.
All tools inherit from BaseTool and implement the execute() method.
Inspired by Claude Code's uniform JSON tool call → sandboxed execution → plain text result pattern.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, Optional
from enum import Enum
import time
import json


class ToolRiskLevel(Enum):
    """Risk classification for tools (inspired by Claude Code's permission system)."""
    READ_ONLY = "read_only"       # View, Glob, LS - no confirmation needed
    WRITE = "write"               # Edit, Write - needs confirmation
    EXECUTE = "execute"           # Bash - needs confirmation + risk assessment
    NETWORK = "network"           # Search, Fetch - needs confirmation
    DANGEROUS = "dangerous"       # Destructive operations - always confirm


@dataclass
class ToolResult:
    """Uniform result from any tool execution."""
    tool_name: str
    success: bool
    output: str
    error: Optional[str] = None
    duration_ms: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "tool_name": self.tool_name,
            "success": self.success,
            "output": self.output,
            "error": self.error,
            "duration_ms": self.duration_ms,
            "metadata": self.metadata,
        }

    def to_display(self) -> str:
        """Format for display in the UI."""
        if self.success:
            return self.output
        return f"Error ({self.tool_name}): {self.error}"


class BaseTool(ABC):
    """
    Base class for all tools in Code SSG.
    
    Design principles (from Claude Code):
    - JSON tool calls → sandboxed execution → plain text results
    - Each tool has a risk level for the permission system
    - Tools return uniform ToolResult objects
    """

    def __init__(self, name: str, description: str, risk_level: ToolRiskLevel = ToolRiskLevel.READ_ONLY):
        self.name = name
        self.description = description
        self.risk_level = risk_level

    @abstractmethod
    def execute(self, **kwargs) -> ToolResult:
        """Execute the tool with given arguments. Must be implemented by subclasses."""
        pass

    def safe_execute(self, **kwargs) -> ToolResult:
        """Execute with timing and error handling."""
        start = time.time()
        try:
            result = self.execute(**kwargs)
            result.duration_ms = (time.time() - start) * 1000
            return result
        except Exception as e:
            duration = (time.time() - start) * 1000
            return ToolResult(
                tool_name=self.name,
                success=False,
                output="",
                error=str(e),
                duration_ms=duration,
            )

    def get_schema(self) -> Dict[str, Any]:
        """Return JSON schema for this tool (for LLM tool calling)."""
        return {
            "name": self.name,
            "description": self.description,
            "risk_level": self.risk_level.value,
        }
