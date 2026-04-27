"""Launch Chromium logged into chat.z.ai, intercept POST traffic, dump
headers/body/URL to ~/.zai-proxy/capture.json.

Usage:
  .venv/bin/python -m probe.zai_capture                # one completion, exit
  .venv/bin/python -m probe.zai_capture --turns 3      # capture multi-turn
                                                       # (writes continuation
                                                       #  capture for RE)

In multi-turn mode, send N messages in the SAME chat in the opened browser.
The script accumulates every /api/* POST + its response body so we can see how
the front-end chains chat_id, message_id, and parent_id across turns.
"""
from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

from playwright.async_api import async_playwright

from app.config import ZAI_PROFILE_DIR, ZAI_STATE_DIR

CAPTURE_FILE = ZAI_STATE_DIR / "capture.json"
CONTINUATION_FILE = Path(__file__).parent / "zai_continuation_capture.json"


async def main(turns: int) -> None:
    done = asyncio.Event()
    completion_count = 0
    captures: list = []

    async with async_playwright() as p:
        ctx = await p.chromium.launch_persistent_context(
            str(ZAI_PROFILE_DIR),
            headless=False,
            channel="chrome",
            viewport=None,
            args=[
                "--disable-blink-features=AutomationControlled",
                "--no-first-run",
                "--no-default-browser-check",
            ],
            ignore_default_args=["--enable-automation"],
        )
        await ctx.add_init_script(
            """
            Object.defineProperty(navigator, 'webdriver', {get: () => undefined});
            window.chrome = window.chrome || {runtime: {}};
            Object.defineProperty(navigator, 'plugins', {get: () => [1,2,3,4,5]});
            Object.defineProperty(navigator, 'languages', {get: () => ['en-US','en']});
            """
        )
        page = ctx.pages[0] if ctx.pages else await ctx.new_page()

        def on_request(req):
            nonlocal completion_count
            if req.method != "POST":
                return
            if "z.ai" not in req.url:
                return
            if "/api/" not in req.url:
                return
            try:
                post_data = req.post_data
            except Exception:
                post_data = None
            try:
                body = json.loads(post_data) if post_data else None
            except (TypeError, json.JSONDecodeError):
                body = post_data
            entry = {
                "turn": completion_count + 1,
                "url": req.url,
                "method": req.method,
                "headers": dict(req.headers),
                "body": body,
            }
            captures.append(entry)
            print(f"[capture #{len(captures)}] POST {req.url[:140]}")
            if isinstance(body, dict):
                interesting = {
                    k: body[k] for k in body
                    if k in ("chat_id", "id", "current_user_message_id",
                             "current_user_message_parent_id", "model")
                }
                if interesting:
                    print(f"  ids: {interesting}")
                else:
                    print(f"  body keys: {list(body.keys())}")
            if "/chat/completions" in req.url:
                completion_count += 1
                if completion_count >= turns:
                    done.set()

        page.on("request", on_request)

        await page.goto("https://chat.z.ai/")
        print(
            f"Send {turns} message{'s' if turns != 1 else ''} in the SAME chat. "
            "Waiting for POST /api/v2/chat/completions ..."
        )
        try:
            await asyncio.wait_for(done.wait(), timeout=900)
        except asyncio.TimeoutError:
            print(
                f"[capture] timed out after {completion_count}/{turns} turns; "
                "writing what we have"
            )

        # Always write to capture.json (single-turn artifact, overwrites OK).
        CAPTURE_FILE.write_text(json.dumps(captures, indent=2, default=str))
        print(f"[capture] wrote {len(captures)} entries to {CAPTURE_FILE}")
        # If multi-turn, also write to the continuation reference file.
        if turns > 1:
            CONTINUATION_FILE.write_text(
                json.dumps(captures, indent=2, default=str)
            )
            print(f"[capture] continuation reference -> {CONTINUATION_FILE}")
        await ctx.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--turns", type=int, default=1,
        help="Stop after N /chat/completions POSTs (default 1).",
    )
    args = parser.parse_args()
    asyncio.run(main(args.turns))
