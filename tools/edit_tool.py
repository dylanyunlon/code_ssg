"""
Edit Tool - Edit files with diff display.
Implements features from claudecode功能.txt:
- Edit files with +lines/-lines diff display
- Replace content in files  
- Revert changes when needed
"""

import os
import shutil
import difflib
from typing import Optional, Dict, Any
from .base_tool import BaseTool, ToolResult, ToolRiskLevel


class EditTool(BaseTool):
    """
    Edit files with surgical patches and full replacements.
    Shows colorized diffs like Claude Code.
    Maintains backup for revert capability.
    """

    def __init__(self, backup_dir: str = "/tmp/code_ssg_backups"):
        super().__init__(
            name="edit",
            description="Edit file content with str_replace or full write, showing diffs",
            risk_level=ToolRiskLevel.WRITE,
        )
        self.backup_dir = backup_dir
        os.makedirs(backup_dir, exist_ok=True)
        self._backup_registry: Dict[str, str] = {}  # path -> backup_path

    def execute(self, path: str, old_str: Optional[str] = None,
                new_str: Optional[str] = None, full_content: Optional[str] = None,
                **kwargs) -> ToolResult:
        """
        Edit a file.
        
        Modes:
        1. str_replace: Replace old_str with new_str (old_str must be unique)
        2. full_write: Write full_content to file (creates if not exists)
        """
        if full_content is not None:
            return self._full_write(path, full_content)
        elif old_str is not None:
            return self._str_replace(path, old_str, new_str or "")
        else:
            return ToolResult(
                tool_name=self.name,
                success=False,
                output="",
                error="Must provide either (old_str, new_str) or full_content",
            )

    def _backup(self, path: str):
        """Create backup before editing."""
        if os.path.exists(path):
            backup_path = os.path.join(
                self.backup_dir,
                path.replace("/", "_") + ".bak"
            )
            shutil.copy2(path, backup_path)
            self._backup_registry[path] = backup_path

    def _str_replace(self, path: str, old_str: str, new_str: str) -> ToolResult:
        """Replace a unique string in a file."""
        if not os.path.exists(path):
            return ToolResult(
                tool_name=self.name,
                success=False,
                output="",
                error=f"File not found: {path}",
            )

        try:
            with open(path, 'r', encoding='utf-8') as f:
                content = f.read()
        except Exception as e:
            return ToolResult(
                tool_name=self.name, success=False, output="", error=str(e)
            )

        # Check uniqueness
        count = content.count(old_str)
        if count == 0:
            return ToolResult(
                tool_name=self.name,
                success=False,
                output="",
                error=f"String not found in {path}",
            )
        if count > 1:
            return ToolResult(
                tool_name=self.name,
                success=False,
                output="",
                error=f"String appears {count} times in {path} (must be unique)",
            )

        # Backup and replace
        self._backup(path)
        new_content = content.replace(old_str, new_str, 1)

        with open(path, 'w', encoding='utf-8') as f:
            f.write(new_content)

        # Generate diff
        diff = self._generate_diff(content, new_content, path)
        added = diff.count('\n+') - 1  # exclude +++ header
        removed = diff.count('\n-') - 1  # exclude --- header

        return ToolResult(
            tool_name=self.name,
            success=True,
            output=f"Edited: {path}, +{max(0,added)}, -{max(0,removed)}\n\n{diff}",
            metadata={
                "path": path,
                "lines_added": max(0, added),
                "lines_removed": max(0, removed),
                "operation": "str_replace",
            },
        )

    def _full_write(self, path: str, content: str) -> ToolResult:
        """Write full content to a file."""
        is_new = not os.path.exists(path)

        if not is_new:
            self._backup(path)
            with open(path, 'r', encoding='utf-8') as f:
                old_content = f.read()
        else:
            old_content = ""
            # Create parent directories
            os.makedirs(os.path.dirname(path) or '.', exist_ok=True)

        with open(path, 'w', encoding='utf-8') as f:
            f.write(content)

        if is_new:
            line_count = content.count('\n') + (1 if content and not content.endswith('\n') else 0)
            return ToolResult(
                tool_name=self.name,
                success=True,
                output=f"Created: {path} (+{line_count} lines)",
                metadata={"path": path, "operation": "create", "lines_added": line_count},
            )

        diff = self._generate_diff(old_content, content, path)
        added = diff.count('\n+') - 1
        removed = diff.count('\n-') - 1

        return ToolResult(
            tool_name=self.name,
            success=True,
            output=f"Wrote: {path}, +{max(0,added)}, -{max(0,removed)}\n\n{diff}",
            metadata={
                "path": path,
                "lines_added": max(0, added),
                "lines_removed": max(0, removed),
                "operation": "full_write",
            },
        )

    def revert(self, path: str) -> ToolResult:
        """Revert a file to its backup (feature #14 from claudecode功能.txt)."""
        if path not in self._backup_registry:
            return ToolResult(
                tool_name=self.name,
                success=False,
                output="",
                error=f"No backup found for {path}",
            )

        backup_path = self._backup_registry[path]
        if not os.path.exists(backup_path):
            return ToolResult(
                tool_name=self.name,
                success=False,
                output="",
                error=f"Backup file missing: {backup_path}",
            )

        shutil.copy2(backup_path, path)
        del self._backup_registry[path]

        return ToolResult(
            tool_name=self.name,
            success=True,
            output=f"Reverted: {path} to previous version",
            metadata={"path": path, "operation": "revert"},
        )

    @staticmethod
    def _generate_diff(old_content: str, new_content: str, path: str) -> str:
        """Generate unified diff between old and new content."""
        old_lines = old_content.splitlines(keepends=True)
        new_lines = new_content.splitlines(keepends=True)

        diff = difflib.unified_diff(
            old_lines, new_lines,
            fromfile=f"a/{path}",
            tofile=f"b/{path}",
            lineterm="",
        )
        return "\n".join(diff)
