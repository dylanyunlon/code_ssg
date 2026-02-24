"""
Context Manager for the Agentic Loop.

Handles context window management including:
- Token estimation
- Context compression (Compressor wU2 pattern from Claude Code)
- Long-term memory via CLAUDE.md / project memory files
- Message history compaction
"""

import json
import time
import logging
from typing import List, Dict, Optional
from pathlib import Path

logger = logging.getLogger(__name__)


class ContextManager:
    """
    Manages the context window for the agentic loop.

    Implements the Claude Code pattern:
    - Estimates token usage
    - Triggers compression at ~92% capacity
    - Summarizes old messages while preserving recent context
    - Maintains long-term memory in markdown files
    """

    CHARS_PER_TOKEN = 4  # Rough estimate
    COMPRESSION_THRESHOLD = 0.92
    KEEP_RECENT = 10  # Messages to keep during compression

    def __init__(
        self,
        max_tokens: int = 200000,
        memory_file: str = "CLAUDE.md",
    ):
        self.max_tokens = max_tokens
        self.memory_file = Path(memory_file)
        self.compressions_count = 0

    def estimate_tokens(self, messages: List[Dict]) -> int:
        """Estimate token count from messages."""
        total_chars = sum(len(str(m.get("content", ""))) for m in messages)
        return total_chars // self.CHARS_PER_TOKEN

    def usage_fraction(self, messages: List[Dict]) -> float:
        """Get current context usage as a fraction (0.0 to 1.0)."""
        tokens = self.estimate_tokens(messages)
        return tokens / self.max_tokens

    def needs_compression(self, messages: List[Dict]) -> bool:
        """Check if context compression is needed."""
        return self.usage_fraction(messages) > self.COMPRESSION_THRESHOLD

    def compress(self, messages: List[Dict]) -> List[Dict]:
        """
        Compress messages by summarizing older ones.

        Keeps the most recent messages and summarizes the rest.
        Returns a new message list with a summary prefix.
        """
        if len(messages) <= self.KEEP_RECENT + 5:
            return messages

        self.compressions_count += 1
        logger.info(f"Context compression #{self.compressions_count}")

        old = messages[:-self.KEEP_RECENT]
        recent = messages[-self.KEEP_RECENT:]

        summary = self._summarize(old)

        # Save summary to long-term memory
        self._append_to_memory(summary)

        compressed = [
            {
                "role": "system",
                "content": f"[Context Summary - Compression #{self.compressions_count}]\n{summary}",
            }
        ] + recent

        logger.info(
            f"Compressed {len(messages)} messages → {len(compressed)} "
            f"(saved ~{self.estimate_tokens(old)} tokens)"
        )

        return compressed

    def _summarize(self, messages: List[Dict]) -> str:
        """Summarize a list of messages."""
        tool_actions = []
        key_content = []
        files_modified = set()

        for m in messages:
            content = str(m.get("content", ""))
            role = m.get("role", "")

            if role == "assistant":
                # Extract key decisions
                if len(content) > 100:
                    key_content.append(content[:150] + "...")

            elif role == "tool_result":
                try:
                    result = json.loads(content)
                    if "path" in result:
                        files_modified.add(result["path"])
                    if "command" in result:
                        tool_actions.append(f"Ran: {result['command'][:80]}")
                    elif "query" in result:
                        tool_actions.append(f"Searched: {result['query']}")
                except (json.JSONDecodeError, TypeError):
                    pass

        parts = []
        if tool_actions:
            parts.append("Actions taken:\n" + "\n".join(f"  - {a}" for a in tool_actions[:15]))
        if files_modified:
            parts.append(f"Files modified: {', '.join(files_modified)}")
        if key_content:
            parts.append("Key findings:\n" + "\n".join(f"  - {k}" for k in key_content[:8]))

        return "\n\n".join(parts)

    def _append_to_memory(self, summary: str):
        """Append summary to the long-term memory file."""
        try:
            existing = ""
            if self.memory_file.exists():
                existing = self.memory_file.read_text()

            timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
            new_entry = f"\n\n## Session Summary [{timestamp}]\n\n{summary}"

            self.memory_file.write_text(existing + new_entry)
            logger.debug(f"Appended to memory file: {self.memory_file}")
        except Exception as e:
            logger.warning(f"Could not write to memory file: {e}")

    def load_memory(self) -> Optional[str]:
        """Load long-term memory from file."""
        if self.memory_file.exists():
            return self.memory_file.read_text()
        return None
