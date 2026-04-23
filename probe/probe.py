"""End-to-end probe: create session, solve PoW, stream completion. Prints tokens.

Usage: python -m probe.probe "your prompt here"
"""
import asyncio
import json
import sys

import httpx

from probe.login import load_state
from probe.pow import solve_challenge

BASE = "https://chat.deepseek.com/api/v0"


def headers(token: str, pow_resp: str | None = None) -> dict:
    h = {
        "authorization": f"Bearer {token}",
        "content-type": "application/json",
        "x-app-version": "20241129.1",
        "x-client-platform": "web",
        "x-client-version": "1.0.0-always",
        "user-agent": (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/131.0.0.0 Safari/537.36"
        ),
        "accept": "*/*",
        "origin": "https://chat.deepseek.com",
        "referer": "https://chat.deepseek.com/",
    }
    if pow_resp:
        h["x-ds-pow-response"] = pow_resp
    return h


def cookies_dict(state: dict) -> dict:
    return {c["name"]: c["value"] for c in state["cookies"]}


async def create_session(client: httpx.AsyncClient, state: dict) -> str:
    r = await client.post(
        f"{BASE}/chat_session/create",
        headers=headers(state["userToken"]),
        cookies=cookies_dict(state),
        json={"character_id": None},
    )
    r.raise_for_status()
    data = r.json()
    print(f"[session.create] {data}")
    return data["data"]["biz_data"]["id"]


async def get_pow(client: httpx.AsyncClient, state: dict, target: str) -> str:
    r = await client.post(
        f"{BASE}/chat/create_pow_challenge",
        headers=headers(state["userToken"]),
        cookies=cookies_dict(state),
        json={"target_path": target},
    )
    r.raise_for_status()
    data = r.json()
    challenge = data["data"]["biz_data"]["challenge"]
    challenge["target_path"] = target
    return solve_challenge(challenge)


async def stream_completion(
    client: httpx.AsyncClient,
    state: dict,
    session_id: str,
    prompt: str,
    thinking: bool,
    search: bool,
) -> None:
    target = "/api/v0/chat/completion"
    pow_resp = await get_pow(client, state, target)
    print(f"[pow] solved, len={len(pow_resp)}")

    body = {
        "chat_session_id": session_id,
        "parent_message_id": None,
        "prompt": prompt,
        "ref_file_ids": [],
        "thinking_enabled": thinking,
        "search_enabled": search,
    }
    async with client.stream(
        "POST",
        f"{BASE}/chat/completion",
        headers=headers(state["userToken"], pow_resp),
        cookies=cookies_dict(state),
        json=body,
        timeout=120.0,
    ) as r:
        if r.status_code != 200:
            body = await r.aread()
            print(f"[HTTP {r.status_code}] {body.decode(errors='replace')[:500]}")
            return
        print(f"[stream] connected {r.status_code} ct={r.headers.get('content-type')}")
        event = None
        current_path = None
        msg_ids = {}
        async for line in r.aiter_lines():
            if not line:
                continue
            if line.startswith("event:"):
                event = line[6:].strip()
                continue
            if not line.startswith("data:"):
                continue
            try:
                chunk = json.loads(line[5:].strip())
            except json.JSONDecodeError:
                continue

            if event == "ready":
                msg_ids = {
                    "request": chunk.get("request_message_id"),
                    "response": chunk.get("response_message_id"),
                }
                print(f"[ready] {msg_ids}")
                event = None
                continue
            if event == "finish":
                print(f"\n[finish] msg_id={msg_ids.get('response')}")
                break
            if event in ("update_session", "title", "close"):
                event = None
                continue
            event = None  # data line without event = content update

            # JSON-patch style content frames
            p = chunk.get("p")
            v = chunk.get("v")
            op = chunk.get("o")  # APPEND | SET (omitted = continue last path)

            if p:
                current_path = p
            if current_path == "response/content" and isinstance(v, str):
                print(v, end="", flush=True)
            elif current_path == "response/thinking_content" and isinstance(v, str):
                print(f"\033[90m{v}\033[0m", end="", flush=True)
            elif current_path and "search" in current_path:
                print(f"\n[search {current_path}] {v}")
            elif current_path == "response/status":
                print(f"\n[status] {v}")
            elif p is None and isinstance(v, dict) and "response" in v:
                # initial snapshot
                pass


async def main():
    prompt = sys.argv[1] if len(sys.argv) > 1 else "Say hi in one word."
    thinking = "--think" in sys.argv
    search = "--search" in sys.argv

    state = load_state()
    if not state:
        print("No state. Run: python -m probe.login")
        sys.exit(1)

    async with httpx.AsyncClient(http2=True) as client:
        sid = await create_session(client, state)
        print(f"[session] {sid}")
        await stream_completion(client, state, sid, prompt, thinking, search)


if __name__ == "__main__":
    asyncio.run(main())
