"""
Permission Manager — Claude Code-style permission gates.
===========================================================
Controls which tool operations are auto-approved vs require user confirmation.

Mirrors Claude Code's permission model:
  - Read operations (view, search, glob): auto-approved
  - Write operations (edit, str_replace): configurable
  - Execute operations (bash, script): configurable
  - Network operations (fetch, web_search): configurable

Location: core/permissions.py (NEW FILE — plan item 1.1.16)
"""

import logging
from enum import Enum
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Set

logger = logging.getLogger(__name__)


class PermissionLevel(Enum):
    """Permission levels for tool categories."""
    AUTO_APPROVE = "auto_approve"       # No confirmation needed
    PROMPT_ONCE = "prompt_once"         # Ask once, remember for session
    PROMPT_ALWAYS = "prompt_always"     # Always ask
    DENY = "deny"                       # Block entirely


class ToolCategory(Enum):
    """Tool categories matching Claude Code's classification."""
    READ = "read"           # view, search, glob, tree
    WRITE = "write"         # edit, str_replace, write_file
    EXECUTE = "execute"     # bash, execute, execute_script, run_commands
    NETWORK = "network"     # fetch, web_search
    AGENT = "agent"         # sub_agent spawning
    PLANNING = "planning"   # todo_write
    VALIDATE = "validate"   # validate_statement


# Default category mapping for tools
TOOL_CATEGORY_MAP: Dict[str, ToolCategory] = {
    # Read
    "view": ToolCategory.READ,
    "view_truncated_section": ToolCategory.READ,
    "view_files": ToolCategory.READ,
    "search": ToolCategory.READ,
    "glob": ToolCategory.READ,
    "ls": ToolCategory.READ,
    "tree": ToolCategory.READ,
    # Write
    "edit": ToolCategory.WRITE,
    "str_replace": ToolCategory.WRITE,
    "write_file": ToolCategory.WRITE,
    "multi_edit": ToolCategory.WRITE,
    # Execute
    "execute": ToolCategory.EXECUTE,
    "execute_script": ToolCategory.EXECUTE,
    "run_commands": ToolCategory.EXECUTE,
    "bash": ToolCategory.EXECUTE,
    # Network
    "fetch": ToolCategory.NETWORK,
    "web_search": ToolCategory.NETWORK,
    "web_fetch": ToolCategory.NETWORK,
    # Agent
    "sub_agent": ToolCategory.AGENT,
    # Planning
    "todo_write": ToolCategory.PLANNING,
    # Validate
    "validate_statement": ToolCategory.VALIDATE,
}

# Default permission levels per category
DEFAULT_PERMISSIONS: Dict[ToolCategory, PermissionLevel] = {
    ToolCategory.READ: PermissionLevel.AUTO_APPROVE,
    ToolCategory.WRITE: PermissionLevel.PROMPT_ONCE,
    ToolCategory.EXECUTE: PermissionLevel.PROMPT_ONCE,
    ToolCategory.NETWORK: PermissionLevel.PROMPT_ONCE,
    ToolCategory.AGENT: PermissionLevel.PROMPT_ALWAYS,
    ToolCategory.PLANNING: PermissionLevel.AUTO_APPROVE,
    ToolCategory.VALIDATE: PermissionLevel.AUTO_APPROVE,
}


@dataclass
class PermissionDecision:
    """Result of a permission check."""
    allowed: bool
    reason: str = ""
    category: Optional[ToolCategory] = None
    level: Optional[PermissionLevel] = None


class PermissionManager:
    """
    Manages tool execution permissions.
    
    Claude Code's permission model:
    - auto_approve_reads=True by default
    - Write/execute operations ask once per session
    - Sub-agent spawning always asks
    - Denied tools are blocked entirely
    
    Usage:
        pm = PermissionManager(auto_approve_reads=True)
        decision = pm.check_permission("execute", {"command": "ls"})
        if not decision.allowed:
            # Ask user for confirmation
            pm.grant_session_permission("execute")
    """

    def __init__(
        self,
        auto_approve_reads: bool = True,
        auto_approve_writes: bool = False,
        auto_approve_execute: bool = False,
        auto_approve_network: bool = False,
        custom_permissions: Optional[Dict[ToolCategory, PermissionLevel]] = None,
        prompt_callback: Optional[Callable[[str, str, dict], bool]] = None,
    ):
        self.permissions = dict(DEFAULT_PERMISSIONS)
        if custom_permissions:
            self.permissions.update(custom_permissions)

        # Apply convenience flags
        if auto_approve_reads:
            self.permissions[ToolCategory.READ] = PermissionLevel.AUTO_APPROVE
        if auto_approve_writes:
            self.permissions[ToolCategory.WRITE] = PermissionLevel.AUTO_APPROVE
        if auto_approve_execute:
            self.permissions[ToolCategory.EXECUTE] = PermissionLevel.AUTO_APPROVE
        if auto_approve_network:
            self.permissions[ToolCategory.NETWORK] = PermissionLevel.AUTO_APPROVE

        # Session-level grants (tools approved for this session)
        self._session_grants: Set[ToolCategory] = set()

        # Callback for prompting user (if None, auto-deny non-approved)
        self.prompt_callback = prompt_callback

        # Audit log
        self._audit_log: List[Dict] = []

    def check_permission(
        self, tool_name: str, arguments: dict = None
    ) -> PermissionDecision:
        """
        Check if a tool call is permitted.
        
        Returns PermissionDecision with allowed=True/False.
        """
        category = TOOL_CATEGORY_MAP.get(tool_name, ToolCategory.EXECUTE)
        level = self.permissions.get(category, PermissionLevel.PROMPT_ALWAYS)

        # Auto-approve
        if level == PermissionLevel.AUTO_APPROVE:
            decision = PermissionDecision(
                allowed=True, reason="auto_approved",
                category=category, level=level,
            )
            self._log(tool_name, arguments, decision)
            return decision

        # Deny
        if level == PermissionLevel.DENY:
            decision = PermissionDecision(
                allowed=False, reason=f"Tool category '{category.value}' is denied",
                category=category, level=level,
            )
            self._log(tool_name, arguments, decision)
            return decision

        # Prompt once — check session grants
        if level == PermissionLevel.PROMPT_ONCE and category in self._session_grants:
            decision = PermissionDecision(
                allowed=True, reason="session_granted",
                category=category, level=level,
            )
            self._log(tool_name, arguments, decision)
            return decision

        # Need user approval
        if self.prompt_callback:
            approved = self.prompt_callback(tool_name, category.value, arguments or {})
            if approved:
                if level == PermissionLevel.PROMPT_ONCE:
                    self._session_grants.add(category)
                decision = PermissionDecision(
                    allowed=True, reason="user_approved",
                    category=category, level=level,
                )
            else:
                decision = PermissionDecision(
                    allowed=False, reason="user_denied",
                    category=category, level=level,
                )
        else:
            # No callback — deny by default for non-auto-approved
            decision = PermissionDecision(
                allowed=False,
                reason=f"Tool '{tool_name}' requires '{level.value}' permission (category: {category.value})",
                category=category, level=level,
            )

        self._log(tool_name, arguments, decision)
        return decision

    def grant_session_permission(self, category_or_tool: str):
        """Grant permission for a category for the rest of this session."""
        # Try as category first
        try:
            cat = ToolCategory(category_or_tool)
        except ValueError:
            cat = TOOL_CATEGORY_MAP.get(category_or_tool)
        if cat:
            self._session_grants.add(cat)
            logger.info(f"Granted session permission for category: {cat.value}")

    def revoke_session_permission(self, category_or_tool: str):
        """Revoke a session grant."""
        try:
            cat = ToolCategory(category_or_tool)
        except ValueError:
            cat = TOOL_CATEGORY_MAP.get(category_or_tool)
        if cat:
            self._session_grants.discard(cat)

    def set_permission(self, category: ToolCategory, level: PermissionLevel):
        """Override permission level for a category."""
        self.permissions[category] = level

    def get_audit_log(self) -> List[Dict]:
        """Return the permission audit log."""
        return list(self._audit_log)

    def reset_session(self):
        """Clear all session grants."""
        self._session_grants.clear()

    def _log(self, tool_name: str, arguments: dict, decision: PermissionDecision):
        self._audit_log.append({
            "tool": tool_name,
            "category": decision.category.value if decision.category else "unknown",
            "allowed": decision.allowed,
            "reason": decision.reason,
        })