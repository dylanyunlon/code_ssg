"""
View Tool - Read files with truncation support.
Implements features #2 and #3 from claudecode功能.txt:
- View truncated section of xxx.py
- View multiple files at once
"""

import os
from typing import Optional, List, Dict, Any
from .base_tool import BaseTool, ToolResult, ToolRiskLevel


class ViewTool(BaseTool):
    """
    View file contents with intelligent truncation.
    
    Inspired by Claude Code's View tool:
    - Defaults to ~2000 lines
    - Supports line range viewing
    - Handles truncation with "View truncated section" capability
    """

    DEFAULT_MAX_LINES = 2000
    TRUNCATION_THRESHOLD = 200  # Show truncation notice after this many lines

    def __init__(self):
        super().__init__(
            name="view",
            description="View file contents with optional line range and truncation support",
            risk_level=ToolRiskLevel.READ_ONLY,
        )

    def execute(self, path: str, start_line: Optional[int] = None,
                end_line: Optional[int] = None, max_lines: Optional[int] = None,
                **kwargs) -> ToolResult:
        """
        View a file's contents.
        
        Args:
            path: File path to view
            start_line: Starting line number (1-indexed)
            end_line: Ending line number (1-indexed, -1 for end of file)
            max_lines: Maximum number of lines to show
        """
        if not os.path.exists(path):
            return ToolResult(
                tool_name=self.name,
                success=False,
                output="",
                error=f"File not found: {path}",
            )

        if os.path.isdir(path):
            return self._view_directory(path)

        return self._view_file(path, start_line, end_line, max_lines)

    def _view_file(self, path: str, start_line: Optional[int] = None,
                   end_line: Optional[int] = None, max_lines: Optional[int] = None) -> ToolResult:
        """View a single file with line range support."""
        try:
            with open(path, 'r', encoding='utf-8', errors='replace') as f:
                lines = f.readlines()
        except Exception as e:
            return ToolResult(
                tool_name=self.name,
                success=False,
                output="",
                error=f"Cannot read file: {e}",
            )

        total_lines = len(lines)
        max_l = max_lines or self.DEFAULT_MAX_LINES

        # Apply line range
        if start_line is not None:
            start_idx = max(0, start_line - 1)
        else:
            start_idx = 0

        if end_line is not None:
            if end_line == -1:
                end_idx = total_lines
            else:
                end_idx = min(total_lines, end_line)
        else:
            end_idx = total_lines

        selected_lines = lines[start_idx:end_idx]
        truncated = False
        truncated_from = None
        truncated_to = None

        # Apply max_lines truncation
        if len(selected_lines) > max_l:
            truncated = True
            truncated_from = start_idx + max_l + 1
            truncated_to = end_idx
            selected_lines = selected_lines[:max_l]

        # Format output with line numbers
        output_lines = []
        for i, line in enumerate(selected_lines):
            line_num = start_idx + i + 1
            output_lines.append(f"{line_num:6d} | {line.rstrip()}")

        output = "\n".join(output_lines)

        metadata = {
            "path": path,
            "total_lines": total_lines,
            "shown_lines": len(selected_lines),
            "start_line": start_idx + 1,
            "end_line": start_idx + len(selected_lines),
            "truncated": truncated,
        }

        if truncated:
            output += f"\n\n... [Truncated: lines {truncated_from}-{truncated_to} not shown]"
            output += f"\n    Use: view_truncated('{path}', {truncated_from}, {truncated_to})"
            metadata["truncated_from"] = truncated_from
            metadata["truncated_to"] = truncated_to

        return ToolResult(
            tool_name=self.name,
            success=True,
            output=output,
            metadata=metadata,
        )

    def _view_directory(self, path: str, max_depth: int = 2) -> ToolResult:
        """List directory contents up to max_depth levels."""
        entries = []
        self._walk_dir(path, entries, depth=0, max_depth=max_depth, prefix="")

        output = f"Directory: {path}\n" + "\n".join(entries)
        return ToolResult(
            tool_name=self.name,
            success=True,
            output=output,
            metadata={"path": path, "entry_count": len(entries)},
        )

    def _walk_dir(self, path: str, entries: list, depth: int, max_depth: int, prefix: str):
        """Recursively walk directory tree."""
        if depth >= max_depth:
            return

        try:
            items = sorted(os.listdir(path))
        except PermissionError:
            entries.append(f"{prefix}[Permission Denied]")
            return

        # Filter hidden files and node_modules
        items = [i for i in items if not i.startswith('.') and i != 'node_modules' and i != '__pycache__']

        for i, item in enumerate(items):
            is_last = (i == len(items) - 1)
            connector = "└── " if is_last else "├── "
            full_path = os.path.join(path, item)

            if os.path.isdir(full_path):
                entries.append(f"{prefix}{connector}{item}/")
                next_prefix = prefix + ("    " if is_last else "│   ")
                self._walk_dir(full_path, entries, depth + 1, max_depth, next_prefix)
            else:
                size = os.path.getsize(full_path)
                entries.append(f"{prefix}{connector}{item} ({self._format_size(size)})")

    @staticmethod
    def _format_size(size: int) -> str:
        for unit in ['B', 'KB', 'MB', 'GB']:
            if size < 1024:
                return f"{size:.0f}{unit}"
            size /= 1024
        return f"{size:.0f}TB"


class MultiViewTool(BaseTool):
    """
    View multiple files at once (feature #3 from claudecode功能.txt).
    Shows: "Viewed 3 files" summary.
    """

    def __init__(self):
        super().__init__(
            name="multi_view",
            description="View multiple files at once with summaries",
            risk_level=ToolRiskLevel.READ_ONLY,
        )
        self._view_tool = ViewTool()

    def execute(self, paths: List[str], max_lines_each: int = 100, **kwargs) -> ToolResult:
        """View multiple files, showing a summary header."""
        results = []
        for path in paths:
            result = self._view_tool.execute(path=path, max_lines=max_lines_each)
            results.append(result)

        # Build summary
        successful = sum(1 for r in results if r.success)
        output_parts = [f"Viewed {successful} file(s)\n"]

        for path, result in zip(paths, results):
            output_parts.append(f"\n{'='*60}")
            output_parts.append(f"File: {path}")
            output_parts.append(f"{'='*60}")
            output_parts.append(result.output if result.success else f"Error: {result.error}")

        return ToolResult(
            tool_name=self.name,
            success=all(r.success for r in results),
            output="\n".join(output_parts),
            metadata={
                "files_viewed": successful,
                "total_files": len(paths),
                "file_details": [r.metadata for r in results],
            },
        )
