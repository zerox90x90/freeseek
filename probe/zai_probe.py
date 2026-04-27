"""End-to-end probe against chat.z.ai. Uses saved state from zai_login
and the proxy's ZaiClient (so tests the real signature + chats/new path).

Usage:
  python -m probe.zai_probe "hi"
  python -m probe.zai_probe "search the web for claude 5" --search
  python -m probe.zai_probe "think carefully" --think
  python -m probe.zai_probe "..." --model glm-5.1
  python -m probe.zai_probe --dump          # just dump /api/models
  python -m probe.zai_probe --tools         # round-trip a synthetic tool call
  python -m probe.zai_probe --continue      # multi-turn cache hit smoke test
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys

import httpx

from app.config import ZAI_BASE_URL
from app.zai import auth
from app.zai.client import ZaiClient, _base_headers
from app.tools.inject import tool_system_block
from app.tools.parser import ToolCallParser


async def dump_models() -> None:
    state = await auth.get_state()
    token = state["token"]
    headers = _base_headers(token, signature="")
    headers.pop("x-signature", None)
    async with httpx.AsyncClient(http2=True, timeout=30.0) as http:
        r = await http.get(
            f"{ZAI_BASE_URL}/api/models", headers=headers,
            cookies=auth.cookies_dict(state),
        )
        print(f"[models] HTTP {r.status_code} ct={r.headers.get('content-type')}")
        try:
            print(json.dumps(r.json(), indent=2)[:4000])
        except Exception:
            print(r.text[:4000])


async def run(prompt: str, thinking: bool, search: bool, model: str | None) -> None:
    state = await auth.get_state()
    token = state["token"]
    user_id = state.get("user_id") or ""
    print(f"[auth] token={token[:16]}... user_id={user_id or '(anon)'}")

    client = ZaiClient()
    try:
        session = await client.create_session()
        print(f"[probe] session={session} model={model or 'default'} "
              f"thinking={thinking} search={search}")
        async for ev in client.stream_completion(
            session_id=session, prompt=prompt,
            thinking=thinking, search=search, model=model,
        ):
            t = ev.get("type")
            if t == "thinking":
                print(f"\033[90m{ev['text']}\033[0m", end="", flush=True)
            elif t == "content":
                print(ev["text"], end="", flush=True)
            elif t == "search_status":
                print(f"\n[search] {ev['status']}")
            elif t == "search_results":
                print(f"\n[search:results] {len(ev['results'])} hits")
            elif t == "done":
                print(f"\n[done] message_id={ev.get('message_id')}")
                return
    finally:
        await client.aclose()


async def run_tools(model: str | None) -> None:
    """Send a tool schema and confirm the parser sees streaming arg deltas
    before the envelope closes. Verifies XML-mode streaming end-to-end."""
    tools = [{
        "name": "set_color",
        "description": "Pick a color the user named.",
        "parameters": {
            "type": "object",
            "properties": {"color": {"type": "string"}},
            "required": ["color"],
        },
    }]
    system = tool_system_block(tools)
    user_prompt = (
        "Call the set_color tool with color='periwinkle'. "
        "Reply with ONLY the tool-call block in the EXACT format the system "
        "prompt specifies. No prose, no explanation, no other output."
    )
    full_prompt = f"{system}\n\n{user_prompt}"

    client = ZaiClient()
    parser = ToolCallParser()
    saw_start = False
    saw_end = False
    final_args: dict | None = None
    deltas: list[str] = []
    raw_chunks: list[str] = []
    try:
        session = await client.create_session()
        print(f"[probe:tools] session={session} model={model or 'default'}")
        async for ev in client.stream_completion(
            session_id=session, prompt=full_prompt,
            thinking=False, search=False, model=model,
        ):
            t = ev.get("type")
            if t == "content":
                raw_chunks.append(ev["text"])
                for pev in parser.feed(ev["text"]):
                    pt = pev["type"]
                    if pt == "tool_call_start":
                        saw_start = True
                        print(f"\n[start] name={pev['name']}")
                    elif pt == "tool_call_arg_delta":
                        deltas.append(pev["delta"])
                        print(f"[arg_delta] {pev['delta']!r}")
                    elif pt == "tool_call_end":
                        saw_end = True
                        final_args = pev["arguments"]
                        print(f"[end] args={pev['arguments']}")
            elif t == "done":
                break
        for pev in parser.flush():
            print(f"[flush] {pev}")

        raw = "".join(raw_chunks)
        print(f"\n[raw model output, {len(raw)} chars]\n{raw}\n")

        parse_ok = saw_start and saw_end and isinstance(final_args, dict)
        stream_ok = parse_ok and len(deltas) >= 2
        if not parse_ok:
            print(
                f"[probe:tools] parsing=FAIL "
                f"(saw_start={saw_start} saw_end={saw_end} args={final_args}). "
                "GLM didn't produce a tool-call shape on this turn. Retry, "
                "or run inside Claude Code where the schema is reinforced."
            )
            return
        quality = "streaming" if stream_ok else "buffered"
        print(
            f"[probe:tools] parsing=OK ({quality}: {len(deltas)} delta"
            f"{'s' if len(deltas) != 1 else ''}, args={final_args})"
        )
    finally:
        await client.aclose()


async def run_continue(model: str | None) -> None:
    """Two turns in one cache window. Confirms ZaiClient yields a real
    chat_id on done, and (when ZAI_CONTINUATION=1) reuses it on turn 2."""
    client = ZaiClient()
    try:
        # Turn 1: fresh chat
        session = await client.create_session()
        print(f"[probe:continue] turn 1 session={session}")
        chat_id_1: str | None = None
        async for ev in client.stream_completion(
            session_id=session, prompt="Reply with the single word: alpha",
            thinking=False, search=False, model=model,
        ):
            if ev.get("type") == "content":
                print(ev["text"], end="", flush=True)
            elif ev.get("type") == "done":
                chat_id_1 = ev.get("session_id")
                parent = ev.get("message_id")
                print(f"\n[turn 1 done] chat_id={chat_id_1} parent={parent}")
                break
        if not chat_id_1:
            print("[probe:continue] FAIL: no session_id on done event")
            return

        # Turn 2: pass the real chat_id back as session, with parent
        async for ev in client.stream_completion(
            session_id=chat_id_1, parent_message_id=parent,
            prompt="Reply with the single word: bravo",
            thinking=False, search=False, model=model,
        ):
            if ev.get("type") == "content":
                print(ev["text"], end="", flush=True)
            elif ev.get("type") == "done":
                chat_id_2 = ev.get("session_id")
                print(f"\n[turn 2 done] chat_id={chat_id_2}")
                if chat_id_2 == chat_id_1:
                    print("[probe:continue] OK — chat_id reused across turns")
                else:
                    print(
                        "[probe:continue] continuation NOT used "
                        "(set ZAI_CONTINUATION=1 after multi-turn capture confirms shape)"
                    )
                break
    finally:
        await client.aclose()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("prompt", nargs="?", default="Say hi in one word.")
    p.add_argument("--think", action="store_true")
    p.add_argument("--search", action="store_true")
    p.add_argument("--dump", action="store_true")
    p.add_argument("--tools", action="store_true",
                   help="round-trip a tool call; verify streaming arg deltas")
    p.add_argument("--continue", dest="continue_", action="store_true",
                   help="multi-turn smoke; verify chat_id reused on turn 2")
    p.add_argument("--model", default=None)
    args = p.parse_args()

    if args.dump:
        asyncio.run(dump_models())
        return
    if args.tools:
        asyncio.run(run_tools(args.model))
        return
    if args.continue_:
        asyncio.run(run_continue(args.model))
        return
    asyncio.run(run(args.prompt, args.think, args.search, args.model))


if __name__ == "__main__":
    main()
