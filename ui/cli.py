"""
CLI Interface for Code SSG.
Provides the command-line user experience with:
- Color-coded output
- Progress indicators
- Diff display
- Verification report display
"""

import sys
import os


# ANSI color codes
class Colors:
    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    RED = "\033[31m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    BLUE = "\033[34m"
    MAGENTA = "\033[35m"
    CYAN = "\033[36m"
    WHITE = "\033[37m"
    BG_RED = "\033[41m"
    BG_GREEN = "\033[42m"


def print_header():
    """Print the Code SSG banner."""
    print(f"""
{Colors.CYAN}{Colors.BOLD}╔══════════════════════════════════════════════════╗
║          🔬 Code SSG v1.0                        ║
║    Scientific Statement Generator                 ║
║    Agentic Loop + Execution-Guided Verification   ║
╚══════════════════════════════════════════════════╝{Colors.RESET}
""")


def print_tool_call(tool_name: str, args: dict):
    """Display a tool call."""
    args_display = ", ".join(f"{k}={repr(v)[:50]}" for k, v in args.items() if v)
    print(f"  {Colors.BLUE}⚡ {tool_name}{Colors.RESET}({args_display})")


def print_state_change(old_state, new_state):
    """Display state change."""
    state_colors = {
        "gathering_context": Colors.CYAN,
        "taking_action": Colors.YELLOW,
        "verifying": Colors.MAGENTA,
        "complete": Colors.GREEN,
        "error": Colors.RED,
    }
    color = state_colors.get(new_state.value, Colors.WHITE)
    print(f"  {color}▸ {new_state.value}{Colors.RESET}")


def print_verification(report):
    """Display verification report with colors."""
    if report.is_success():
        print(f"\n  {Colors.GREEN}✅ Verification PASSED{Colors.RESET}")
    else:
        print(f"\n  {Colors.RED}❌ Verification FAILED{Colors.RESET}")

    for sv in report.statements:
        if sv.result.value == "pass":
            icon = f"{Colors.GREEN}✅{Colors.RESET}"
        elif sv.result.value == "fail":
            icon = f"{Colors.RED}❌{Colors.RESET}"
        elif sv.result.value == "warning":
            icon = f"{Colors.YELLOW}⚠️{Colors.RESET}"
        else:
            icon = f"{Colors.DIM}⏭️{Colors.RESET}"

        print(f"    {icon} L{sv.line_number}: {sv.statement[:70]}")
        if sv.details and sv.result.value in ("fail", "error"):
            print(f"       {Colors.DIM}{sv.details[:100]}{Colors.RESET}")


def print_diff(diff_text: str):
    """Print a colorized diff."""
    for line in diff_text.split('\n'):
        if line.startswith('+') and not line.startswith('+++'):
            print(f"  {Colors.GREEN}{line}{Colors.RESET}")
        elif line.startswith('-') and not line.startswith('---'):
            print(f"  {Colors.RED}{line}{Colors.RESET}")
        elif line.startswith('@@'):
            print(f"  {Colors.CYAN}{line}{Colors.RESET}")
        else:
            print(f"  {line}")


def print_result(text: str):
    """Print a result with appropriate formatting."""
    for line in text.split('\n'):
        if line.startswith('✓') or line.startswith('✅'):
            print(f"  {Colors.GREEN}{line}{Colors.RESET}")
        elif line.startswith('✗') or line.startswith('❌'):
            print(f"  {Colors.RED}{line}{Colors.RESET}")
        elif line.startswith('⚠'):
            print(f"  {Colors.YELLOW}{line}{Colors.RESET}")
        elif line.startswith('📋') or line.startswith('📊'):
            print(f"  {Colors.BOLD}{line}{Colors.RESET}")
        elif line.startswith('Done'):
            print(f"  {Colors.GREEN}{Colors.BOLD}{line}{Colors.RESET}")
        else:
            print(f"  {line}")


def print_summary(execution_summary: str, context_info: str):
    """Print final summary."""
    print(f"\n{Colors.DIM}{'─'*50}")
    print(execution_summary)
    print(context_info)
    print(f"{'─'*50}{Colors.RESET}")
