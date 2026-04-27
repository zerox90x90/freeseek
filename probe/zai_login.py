"""Playwright login for chat.z.ai. Opens Chromium, user signs in (or continues
as guest), captures the Bearer token + cookies.

Usage: python -m probe.zai_login
State: ~/.zai-proxy/state.json
"""
import asyncio

from app.zai.auth import _launch_login


if __name__ == "__main__":
    asyncio.run(_launch_login())
