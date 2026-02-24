"""
Event Stream - Real-time SSE event streaming for agentic loop.
===============================================================
Provides Claude Code-style event publishing for:
- Tool execution progress (Ran 7 commands, Viewed 3 files, etc.)
- File change tracking (+N, -M format)
- Context compaction notifications
- Session lifecycle events

Location: core/event_stream.py (NEW FILE)
"""
import time
import asyncio
import logging
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class SSEEvent:
    """A Server-Sent Event."""
    type: str
    data: Dict[str, Any]
    timestamp: float = field(default_factory=time.time)
    session_id: Optional[str] = None
    sequence: int = 0


class EventStream:
    """
    Event streaming system for the agentic loop.
    
    Emits Claude Code-style events:
    - session_start, session_complete
    - thinking (model is processing)
    - tool_calls_start (with title like "Ran 7 commands")
    - tool_executing, tool_completed, tool_error
    - step_completed (with file changes: +N, -M)
    - context_compacting, context_compacted
    - text_response (final answer)
    """

    def __init__(self, max_history: int = 1000):
        self._history: List[SSEEvent] = []
        self._subscribers: List[asyncio.Queue] = []
        self._sequence = 0
        self._max_history = max_history

    def emit(self, event_type: str, data: Dict[str, Any],
             session_id: Optional[str] = None) -> Dict:
        """Emit an event and return it as a dict."""
        self._sequence += 1
        event = SSEEvent(
            type=event_type,
            data=data,
            session_id=session_id,
            sequence=self._sequence,
        )
        self._history.append(event)
        if len(self._history) > self._max_history:
            self._history = self._history[-self._max_history:]

        event_dict = {
            "type": event_type,
            "data": data,
            "timestamp": event.timestamp,
            "session_id": session_id,
            "sequence": self._sequence,
        }

        # Notify subscribers
        for queue in self._subscribers:
            try:
                queue.put_nowait(event_dict)
            except asyncio.QueueFull:
                logger.warning("Subscriber queue full, dropping event")

        return event_dict

    def subscribe(self) -> asyncio.Queue:
        queue = asyncio.Queue(maxsize=200)
        self._subscribers.append(queue)
        return queue

    def unsubscribe(self, queue: asyncio.Queue):
        if queue in self._subscribers:
            self._subscribers.remove(queue)

    async def listen(self, queue: asyncio.Queue):
        """Async generator for listening to events."""
        try:
            while True:
                event = await queue.get()
                yield event
        except asyncio.CancelledError:
            pass

    def get_history(self, since_sequence: int = 0) -> List[Dict]:
        return [
            {
                "type": e.type,
                "data": e.data,
                "timestamp": e.timestamp,
                "session_id": e.session_id,
                "sequence": e.sequence,
            }
            for e in self._history
            if e.sequence > since_sequence
        ]

    @staticmethod
    def format_step_title(tool_calls: List[Dict]) -> str:
        """
        Generate Claude Code-style step titles from tool calls.
        
        Examples:
        - "Ran 7 commands"
        - "Viewed 3 files"
        - "Ran a command, edited a file"
        - "Searched the web"
        - "Fetched: Anthropic's original take home assignment"
        """
        TOOL_CATEGORIES = {
            "execute": "command", "execute_script": "command",
            "run_commands": "command", "bash": "command",
            "view": "view", "view_truncated_section": "view",
            "view_files": "view", "glob": "view",
            "edit": "edit", "str_replace": "edit",
            "web_search": "search", "search": "search",
            "fetch": "fetch",
        }

        counts = {}
        for tc in tool_calls:
            name = tc.get("name", tc.get("tool_name", ""))
            cat = TOOL_CATEGORIES.get(name, "other")
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

        return ", ".join(parts) if parts else "Processing"

    @staticmethod
    def format_file_change(path: str, additions: int, deletions: int) -> str:
        """Format a file change like Claude Code: 'perf_takehome.py, +3, -4'"""
        import os
        name = os.path.basename(path)
        return f"{name}, +{additions}, -{deletions}"