#!/usr/bin/env python3
"""
Code SSG - Scientific Statement Generator
Main entry point.

Usage:
    python main.py                          # Interactive mode
    python main.py "Fix the bug in app.py"  # Single task mode
    python main.py --verify code.py         # Verify a file
"""

import sys
import os
import argparse

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.agent_loop import AgenticLoop
from core.tool_engine import ToolEngine
from core.context_manager import ContextManager
from core.message_queue import MessageQueue
from core.planner import Planner
from tools import ViewTool, MultiViewTool, EditTool, BashTool, SearchTool, GlobTool
from tools.fetch_tool import FetchTool, WebSearchTool
from verification.verifier import VerificationMode, create_verifier
from ui.cli import (
    print_header, print_tool_call, print_state_change,
    print_verification, print_result
)


def setup_tool_engine(working_dir: str = None) -> ToolEngine:
    """Initialize the tool engine with all built-in tools."""
    engine = ToolEngine(auto_approve_reads=True)

    engine.register_all([
        ViewTool(),
        MultiViewTool(),
        EditTool(),
        BashTool(default_cwd=working_dir or os.getcwd()),
        SearchTool(),
        FetchTool(),
        WebSearchTool(),
        GlobTool(),
    ])

    return engine


def setup_agentic_loop(
    working_dir: str = None,
    verification_mode: str = "execution",
    max_tokens: int = 100000,
) -> AgenticLoop:
    """Set up the complete agentic loop."""

    # Initialize components
    tool_engine = setup_tool_engine(working_dir)
    context_manager = ContextManager(max_tokens=max_tokens)
    message_queue = MessageQueue()
    planner = Planner()

    # Set system prompt (like Claude Code's CLAUDE.md)
    context_manager.set_system_prompt(
        "You are Code SSG, a scientific code generation and verification agent. "
        "You gather context, take action, verify results, and repeat. "
        "Every code change is scientifically verified through execution and/or LLM judgment."
    )

    # Parse verification mode
    mode_map = {
        "execution": VerificationMode.EXECUTION,
        "llm": VerificationMode.LLM_JUDGE,
        "hybrid": VerificationMode.HYBRID,
        "trace": VerificationMode.TRACE,
    }
    v_mode = mode_map.get(verification_mode, VerificationMode.EXECUTION)

    # Create the loop
    loop = AgenticLoop(
        tool_engine=tool_engine,
        context_manager=context_manager,
        message_queue=message_queue,
        planner=planner,
        verification_mode=v_mode,
        on_tool_call=print_tool_call,
        on_state_change=print_state_change,
        on_verification=print_verification,
    )

    return loop


def verify_file(file_path: str, mode: str = "execution"):
    """Standalone file verification."""
    if not os.path.exists(file_path):
        print(f"❌ File not found: {file_path}")
        return

    with open(file_path, 'r') as f:
        code = f.read()

    mode_map = {
        "execution": VerificationMode.EXECUTION,
        "llm": VerificationMode.LLM_JUDGE,
        "hybrid": VerificationMode.HYBRID,
        "trace": VerificationMode.TRACE,
    }
    v_mode = mode_map.get(mode, VerificationMode.EXECUTION)
    verifier = create_verifier(v_mode)

    print(f"🔬 Verifying: {file_path} (mode: {mode})")
    report = verifier.verify(code, file_path)
    print(report.to_display())


def main():
    parser = argparse.ArgumentParser(
        description="Code SSG - Scientific Statement Generator"
    )
    parser.add_argument(
        "task", nargs="?", default=None,
        help="Task to execute (omit for interactive mode)"
    )
    parser.add_argument(
        "--verify", "-v", metavar="FILE",
        help="Verify a file's scientific correctness"
    )
    parser.add_argument(
        "--mode", "-m", default="execution",
        choices=["execution", "llm", "hybrid", "trace"],
        help="Verification mode (default: execution)"
    )
    parser.add_argument(
        "--dir", "-d", default=None,
        help="Working directory"
    )

    args = parser.parse_args()

    print_header()

    # Standalone verification mode
    if args.verify:
        verify_file(args.verify, args.mode)
        return

    # Set up the agentic loop
    loop = setup_agentic_loop(
        working_dir=args.dir,
        verification_mode=args.mode,
    )

    if args.task:
        # Single task mode
        response = loop.run(args.task)
        print_result(response)
    else:
        # Interactive mode
        loop.run_interactive()


if __name__ == "__main__":
    main()