"""Playwright login: opens Chromium, user signs in, captures userToken + cookies.

Usage: python -m probe.login
"""
import asyncio
import json
import os
from pathlib import Path

from playwright.async_api import async_playwright

STATE_DIR = Path(os.path.expanduser("~/.deepseek-proxy"))
STATE_FILE = STATE_DIR / "state.json"
PROFILE_DIR = STATE_DIR / "chromium-profile"


async def login() -> dict:
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    PROFILE_DIR.mkdir(parents=True, exist_ok=True)

    async with async_playwright() as p:
        ctx = await p.chromium.launch_persistent_context(
            str(PROFILE_DIR),
            headless=False,
            viewport={"width": 1200, "height": 800},
        )
        page = ctx.pages[0] if ctx.pages else await ctx.new_page()
        await page.goto("https://chat.deepseek.com/")

        print("Log in to DeepSeek in the opened browser. Waiting for userToken...")

        # Poll localStorage until userToken present. Tolerate navigation races.
        token = None
        for _ in range(600):  # 10 min max
            try:
                token_json = await page.evaluate(
                    "() => window.localStorage.getItem('userToken')"
                )
            except Exception:
                await asyncio.sleep(1)
                continue
            if token_json:
                try:
                    val = json.loads(token_json).get("value")
                    if val:
                        token = val
                        break
                except (KeyError, json.JSONDecodeError):
                    pass
            await asyncio.sleep(1)
        if not token:
            await ctx.close()
            raise TimeoutError("userToken not found after 10 min")

        cookies = await ctx.cookies()
        state = {"userToken": token, "cookies": cookies}
        STATE_FILE.write_text(json.dumps(state, indent=2))
        print(f"Saved state to {STATE_FILE}")
        print(f"Token: {token[:20]}...")
        print(f"Cookies: {len(cookies)} ({[c['name'] for c in cookies]})")

        await ctx.close()
        return state


def load_state() -> dict | None:
    if not STATE_FILE.exists():
        return None
    return json.loads(STATE_FILE.read_text())


if __name__ == "__main__":
    asyncio.run(login())
