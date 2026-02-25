
"""
Environment Loader
===================
Loads .env configuration file, same pattern as skynetCheapBuy/test_agentic_loop.py.
Does NOT depend on python-dotenv - uses pure Python parsing.

Location: core/env_loader.py (NEW FILE)
"""

import os
from pathlib import Path


def load_env(env_path: str = None) -> dict:
    """
    Load environment variables from a .env file.
    
    Search order:
    1. Explicit path
    2. .env in current directory
    3. .env in project root (parent of core/)
    """
    loaded = {}
    
    # Find .env file
    if env_path and os.path.exists(env_path):
        target = env_path
    else:
        # Try current dir, then project root
        candidates = [
            os.path.join(os.getcwd(), ".env"),
            os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env"),
        ]
        target = None
        for c in candidates:
            if os.path.exists(c):
                target = c
                break
    
    if not target:
        return loaded
    
    with open(target) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#') or '=' not in line:
                continue
            key, _, value = line.partition('=')
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = value
                loaded[key] = value
    
    return loaded


def get_api_config() -> dict:
    """Get API configuration from environment."""
    return {
        "api_key": os.environ.get("OPENAI_API_KEY") or os.environ.get("ANTHROPIC_API_KEY", ""),
        "api_base": os.environ.get("OPENAI_API_BASE") or os.environ.get("ANTHROPIC_API_BASE", "https://api.tryallai.com/v1"),
        "default_model": os.environ.get("DEFAULT_MODEL", "claude-sonnet-4-5-20250929"),
    }