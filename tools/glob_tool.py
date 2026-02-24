"""
Glob Tool - File pattern matching.
Like Claude Code's Glob tool for wildcard searches.
"""

import glob as glob_module
import os
from typing import Optional
from .base_tool import BaseTool, ToolResult, ToolRiskLevel


class GlobTool(BaseTool):
    """Find files matching a glob pattern."""

    def __init__(self):
        super().__init__(
            name="glob",
            description="Find files matching a wildcard pattern",
            risk_level=ToolRiskLevel.READ_ONLY,
        )

    def execute(self, pattern: str, root: Optional[str] = None, **kwargs) -> ToolResult:
        search_root = root or os.getcwd()
        full_pattern = os.path.join(search_root, pattern)

        try:
            matches = sorted(glob_module.glob(full_pattern, recursive=True))
            # Filter out hidden files
            matches = [m for m in matches if not any(p.startswith('.') for p in m.split(os.sep))]

            if not matches:
                return ToolResult(
                    tool_name=self.name,
                    success=True,
                    output=f"No files matching: {pattern}",
                    metadata={"pattern": pattern, "count": 0},
                )

            output = f"Found {len(matches)} file(s) matching: {pattern}\n\n"
            output += "\n".join(f"  {m}" for m in matches[:100])
            if len(matches) > 100:
                output += f"\n  ... and {len(matches)-100} more"

            return ToolResult(
                tool_name=self.name,
                success=True,
                output=output,
                metadata={"pattern": pattern, "count": len(matches), "matches": matches[:100]},
            )
        except Exception as e:
            return ToolResult(
                tool_name=self.name, success=False, output="", error=str(e)
            )
