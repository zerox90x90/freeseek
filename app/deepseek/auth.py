"""Load persisted DeepSeek web session. Re-login via Playwright on demand."""
import asyncio
import json
from typing import Any

from playwright.async_api import async_playwright

from app.config import PROFILE_DIR, STATE_FILE

_lock = asyncio.Lock()


def _read_state() -> dict[str, Any] | None:
    if not STATE_FILE.exists():
        return None
    try:
        return json.loads(STATE_FILE.read_text())
    except json.JSONDecodeError:
        return None


def _write_state(state: dict[str, Any]) -> None:
    STATE_FILE.write_text(json.dumps(state, indent=2))


async def _launch_login() -> dict[str, Any]:
    PROFILE_DIR.mkdir(parents=True, exist_ok=True)
    async with async_playwright() as p:
        ctx = await p.chromium.launch_persistent_context(
            str(PROFILE_DIR),
            headless=False,
            viewport={"width": 1200, "height": 800},
        )
        page = ctx.pages[0] if ctx.pages else await ctx.new_page()
        await page.goto("https://chat.deepseek.com/")
        print("[auth] waiting for userToken in browser...")

        token: str | None = None
        for _ in range(600):
            try:
                raw = await page.evaluate(
                    "() => window.localStorage.getItem('userToken')"
                )
            except Exception:
                await asyncio.sleep(1)
                continue
            if raw:
                try:
                    val = json.loads(raw).get("value")
                    if val:
                        token = val
                        break
                except (KeyError, json.JSONDecodeError):
                    pass
            await asyncio.sleep(1)
        if not token:
            await ctx.close()
            raise TimeoutError("userToken not captured in 10 min")
        cookies = await ctx.cookies()
        await ctx.close()

    state = {"userToken": token, "cookies": cookies}
    _write_state(state)
    print(f"[auth] saved state, token={token[:16]}... cookies={len(cookies)}")
    return state


async def get_state(force_refresh: bool = False) -> dict[str, Any]:
    async with _lock:
        if not force_refresh:
            state = _read_state()
            if state:
                return state
        return await _launch_login()


def cookies_dict(state: dict[str, Any]) -> dict[str, str]:
    return {c["name"]: c["value"] for c in state["cookies"]}
