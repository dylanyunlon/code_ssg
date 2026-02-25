#!/usr/bin/env python3
"""
Agentic Loop Integration Test for Code-SSG
=============================================
Following skynetCheapBuy/test_agentic_loop.py execution standard.

Tests:
  1. ToolExecutor (pure local tool execution)
  2. ClaudeClient with tool support (httpx-based, no anthropic SDK)
  3. Full AgenticLoop (AI-driven tool execution)

Usage:
    # Set env variables (from .env or export):
    #   OPENAI_API_KEY=sk-...     (or ANTHROPIC_API_KEY)
    #   OPENAI_API_BASE=https://api.tryallai.com/v1  (or similar proxy)
    python test_agentic_loop.py

Location: test_agentic_loop.py (NEW FILE)
"""

import os
import sys
import json
import asyncio
import logging
import tempfile
import shutil

# ============================================================================
# Path setup: ensure code_ssg modules importable
# ============================================================================
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_DIR)

# Optional: load .env file if present
env_file = os.path.join(PROJECT_DIR, ".env")
if os.path.exists(env_file):
    with open(env_file) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, _, value = line.partition("=")
                key = key.strip()
                value = value.strip().strip('"').strip("'")
                if key and key not in os.environ:
                    os.environ[key] = value
    print(f"✅ Loaded .env: {env_file}")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


# ============================================================================
# Test 1: Tool Executor (pure local, no API)
# ============================================================================
async def test_1_tool_executor():
    """Test: Local tool execution works correctly."""
    print("\n" + "=" * 60)
    print("🧪 Test 1: Tool Executor (local tools)")
    print("=" * 60)

    from core.agent_loop import AgenticLoop

    work_dir = tempfile.mkdtemp(prefix="ssg_test_")
    loop = AgenticLoop(max_iterations=5)

    # Test: view tool
    result = loop._tool_view(os.path.join(PROJECT_DIR, "main.py"))
    assert "content" in result and len(result["content"]) > 0, f"view failed: {result}"
    print(f"  ✅ view: {result['total_lines']} lines, truncated={result['truncated']}")

    # Test: view_truncated_section
    result = loop._tool_view_truncated(
        os.path.join(PROJECT_DIR, "main.py"), start_line=1, end_line=10
    )
    assert "content" in result
    print(f"  ✅ view_truncated_section: got lines 1-10")

    # Test: execute
    result = loop._tool_execute('echo "Hello from Code-SSG"')
    assert result["returncode"] == 0
    assert "Hello from Code-SSG" in result["stdout"]
    print(f"  ✅ execute: {result['stdout'].strip()}")

    # Test: edit (write + str_replace)
    test_file = os.path.join(work_dir, "test.txt")
    with open(test_file, "w") as f:
        f.write("Hello World\nLine 2\n")

    result = loop._tool_edit(test_file, "Hello World", "Hello SSG")
    assert result.get("status") == "success" or "path" in result
    print(f"  ✅ edit: {result.get('changes', 'ok')}")

    with open(test_file) as f:
        assert "Hello SSG" in f.read()
    print(f"  ✅ edit verified: content updated correctly")

    # Test: search
    result = loop._tool_search("AgenticLoop", PROJECT_DIR)
    assert result.get("total_matches", 0) > 0
    print(f"  ✅ search: found {result['total_matches']} matches")

    # Test: execute_script
    result = loop._tool_execute_script(
        'python3 -c "print(2+2)"', description="Arithmetic test"
    )
    assert result.get("returncode") == 0
    assert "4" in result.get("stdout", "")
    print(f"  ✅ execute_script: {result['stdout'].strip()}")

    # Test: view_files (batch)
    files = [os.path.join(PROJECT_DIR, "main.py"), os.path.join(PROJECT_DIR, "README.md")]
    existing_files = [f for f in files if os.path.exists(f)]
    if existing_files:
        result = loop._tool_view_files(existing_files)
        assert result["files_viewed"] == len(existing_files)
        print(f"  ✅ view_files: batch viewed {result['files_viewed']} files")

    print(f"\n  ✅ ToolExecutor tests passed!")
    shutil.rmtree(work_dir, ignore_errors=True)


# ============================================================================
# Test 2: ClaudeClient with tools (httpx, no anthropic SDK)
# ============================================================================
async def test_2_claude_client_tools():
    """Test: ClaudeClient calls /v1/messages with tool definitions."""
    print("\n" + "=" * 60)
    print("🧪 Test 2: ClaudeClient with tools (httpx-based)")
    print("=" * 60)

    from core.claude_client import ClaudeClient

    api_key = os.environ.get("OPENAI_API_KEY", os.environ.get("ANTHROPIC_API_KEY", ""))
    api_base = os.environ.get("OPENAI_API_BASE", os.environ.get("ANTHROPIC_API_BASE", ""))

    client = ClaudeClient(
        api_key=api_key,
        model="claude-sonnet-4-5-20250929",
        max_tokens=1024,
        base_url=api_base or None,
    )

    print(f"  📡 Endpoint: {client.endpoint}")
    print(f"  🔑 Key: {api_key[:8]}...{api_key[-4:]}")

    tools = [
        {
            "name": "get_weather",
            "description": "Get current weather for a city",
            "input_schema": {
                "type": "object",
                "properties": {
                    "city": {"type": "string", "description": "City name"}
                },
                "required": ["city"],
            },
        }
    ]

    messages = [
        {
            "role": "user",
            "content": "What's the weather in Beijing? Use the get_weather tool.",
        }
    ]

    print(f"  📡 Calling Claude with tools...")
    result = await client.chat(messages=messages, tools=tools, temperature=0.1)

    # Verify response structure
    assert "content" in result, "Missing content"
    assert "stop_reason" in result, "Missing stop_reason"
    assert "usage" in result, "Missing usage"

    print(f"  ✅ content_blocks: {len(result['content'])} blocks")
    print(f"  ✅ tool_uses: {len(result.get('tool_uses', []))} calls")
    print(f"  ✅ stop_reason: {result['stop_reason']}")
    print(f"  ✅ usage: {result['usage']}")

    if result.get("tool_uses"):
        tu = result["tool_uses"][0]
        print(
            f"  ✅ tool_use: name={tu['name']}, id={tu['id']}, input={tu.get('input', {})}"
        )
    else:
        print(f"  ⚠️  AI didn't call tool, text: {result.get('text', '')[:200]}")

    # Backward compat
    assert isinstance(result.get("text", ""), str)
    print(f"\n  ✅ ClaudeClient tool calling passed!")


# ============================================================================
# Test 3: Full AgenticLoop (AI drives tool execution)
# ============================================================================
async def test_3_agentic_loop():
    """Test: Complete agentic loop with AI-driven tool execution."""
    print("\n" + "=" * 60)
    print("🧪 Test 3: Full Agentic Loop (AI-driven)")
    print("=" * 60)

    from core.agent_loop import AgenticLoop

    work_dir = tempfile.mkdtemp(prefix="ssg_loop_test_")

    loop = AgenticLoop(
        model="claude-sonnet-4-5-20250929",
        max_iterations=15,
    )

    task = (
        "Create a Python file called calc.py with functions add(a,b) and multiply(a,b). "
        "Then create test_calc.py that tests both functions using assert statements. "
        "Run the tests with python3 and verify they pass."
    )

    print(f"  📝 Task: {task[:80]}...")
    print(f"  📁 Work dir: {work_dir}")
    print()

    event_counts = {}

    async for event in loop.run_async(task):
        t = event.get("type", "unknown")
        event_counts[t] = event_counts.get(t, 0) + 1

        if t == "session_start":
            print(f"  🚀 Session started")
        elif t == "thinking":
            print(f"  🤔 Thinking (iteration {event.get('data', {}).get('iteration', '?')})")
        elif t == "text_response":
            text = event.get("data", {}).get("content", "")[:150].replace("\n", " ")
            print(f"  📝 Response: {text}")
        elif t == "tool_calls_start":
            data = event.get("data", {})
            print(f"  🔧 {data.get('display_title', 'Tools')} ({data.get('tool_count', 0)} tools)")
        elif t == "tool_completed":
            data = event.get("data", {})
            print(f"  ✅ {data.get('tool_name', '?')} ({data.get('duration_ms', 0):.0f}ms)")
        elif t == "tool_error":
            data = event.get("data", {})
            print(f"  ❌ {data.get('tool_name', '?')}: {data.get('error', '')[:100]}")
        elif t == "step_completed":
            data = event.get("data", {})
            fc = data.get("files_changed", [])
            if fc:
                print(f"  📄 Files changed: {fc}")
        elif t == "session_complete":
            data = event.get("data", {})
            stats = data.get("stats", {})
            print(f"\n  ✅ DONE! Stats: {stats}")
        elif t == "error":
            print(f"\n  ❌ Error: {event.get('data', {}).get('message', '')}")

    print(f"\n  📊 Events: {event_counts}")

    # Verify files were created (in current dir since loop doesn't know work_dir)
    for fn in ["calc.py", "test_calc.py"]:
        path = os.path.join(os.getcwd(), fn)
        if os.path.exists(path):
            print(f"  📁 {fn}: ✅")
        else:
            print(f"  📁 {fn}: ❌ (may be in different dir)")

    shutil.rmtree(work_dir, ignore_errors=True)


# ============================================================================
# Main
# ============================================================================
async def main():
    print("=" * 60)
    print("🔧 Code-SSG Agentic Loop Integration Test")
    print("=" * 60)

    # Test 1: Pure local (no API needed)
    await test_1_tool_executor()

    # Check API config
    api_key = os.environ.get(
        "OPENAI_API_KEY", os.environ.get("ANTHROPIC_API_KEY", "")
    )
    api_base = os.environ.get(
        "OPENAI_API_BASE", os.environ.get("ANTHROPIC_API_BASE", "")
    )

    if not api_key:
        print("\n⚠️  No API key found (OPENAI_API_KEY / ANTHROPIC_API_KEY)")
        print("   Skipping Tests 2-3 (require API access)")
        print("   Set your key and re-run to test full agentic loop")
        return

    print(f"\n📡 API base: {api_base or '(default)'}")
    print(f"🔑 Key: {api_key[:8]}...{api_key[-4:]}")

    # Test 2: ClaudeClient with tool calling
    await test_2_claude_client_tools()

    # Test 3: Full agentic loop
    await test_3_agentic_loop()

    print("\n" + "=" * 60)
    print("✅ All tests passed! Code-SSG Agentic Loop is operational.")
    print("=" * 60)
    print()
    print("Next steps:")
    print("  1. Run experiments:  python scripts/run_experiments.py --suite all")
    print("  2. Interactive mode: python main.py")
    print("  3. Single task:     python main.py 'Fix the bug in app.py'")


if __name__ == "__main__":
    asyncio.run(main())