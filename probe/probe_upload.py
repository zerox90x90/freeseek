"""Probe /file/upload_file with PoW header."""
import asyncio
import sys
from pathlib import Path

import httpx

from probe.login import load_state
from probe.pow import solve_challenge

BASE = "https://chat.deepseek.com/api/v0"


def headers(token: str, pow_resp: str | None = None) -> dict:
    h = {
        "authorization": f"Bearer {token}",
        "x-app-version": "20241129.1",
        "x-client-platform": "web",
        "x-client-version": "1.0.0-always",
        "user-agent": (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 Chrome/131.0.0.0 Safari/537.36"
        ),
        "accept": "*/*",
        "origin": "https://chat.deepseek.com",
        "referer": "https://chat.deepseek.com/",
    }
    if pow_resp:
        h["x-ds-pow-response"] = pow_resp
    return h


async def get_pow(c: httpx.AsyncClient, state: dict, target: str) -> str:
    r = await c.post(
        f"{BASE}/chat/create_pow_challenge",
        headers=headers(state["userToken"]),
        cookies={x["name"]: x["value"] for x in state["cookies"]},
        json={"target_path": target},
    )
    r.raise_for_status()
    ch = r.json()["data"]["biz_data"]["challenge"]
    ch["target_path"] = target
    return solve_challenge(ch)


async def main():
    file_path = Path(sys.argv[1] if len(sys.argv) > 1 else "/tmp/hello.txt")
    if not file_path.exists():
        file_path.write_text("Hello from probe\nLine 2\n")

    state = load_state()
    cookies = {c["name"]: c["value"] for c in state["cookies"]}
    async with httpx.AsyncClient(http2=True, timeout=60) as c:
        for target in ["/api/v0/file/upload_file", "/file/upload_file"]:
            print(f"\n--- PoW target={target} ---")
            try:
                pow_resp = await get_pow(c, state, target)
                print(f"pow ok, len={len(pow_resp)}")
            except Exception as e:
                print(f"pow err: {e}")
                continue
            r = await c.post(
                f"{BASE}/file/upload_file",
                headers=headers(state["userToken"], pow_resp),
                cookies=cookies,
                files={"file": (file_path.name, file_path.read_bytes(), "text/plain")},
            )
            print(f"status={r.status_code}")
            print(r.text[:600])


if __name__ == "__main__":
    asyncio.run(main())
