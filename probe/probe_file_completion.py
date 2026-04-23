"""End-to-end: upload file, poll status, then ask a question about it."""
import asyncio
import sys
import time
from pathlib import Path

import httpx

from probe.login import load_state
from probe.probe_upload import get_pow, headers
from probe.probe import BASE, cookies_dict, create_session, stream_completion


async def upload(c: httpx.AsyncClient, state: dict, path: Path) -> str:
    pow_resp = await get_pow(c, state, "/api/v0/file/upload_file")
    r = await c.post(
        f"{BASE}/file/upload_file",
        headers=headers(state["userToken"], pow_resp),
        cookies=cookies_dict(state),
        files={"file": (path.name, path.read_bytes(), "text/plain")},
    )
    r.raise_for_status()
    return r.json()["data"]["biz_data"]["id"]


async def main():
    path = Path("/tmp/notes.txt")
    path.write_text(
        "Secret codeword: PINEAPPLE-47. Due date: 2026-05-03. Stakeholder: Jane Doe."
    )

    state = load_state()
    async with httpx.AsyncClient(http2=True, timeout=120) as c:
        file_id = await upload(c, state, path)
        print(f"[upload] {file_id}")
        time.sleep(2)

        sid = await create_session(c, state)
        print(f"[session] {sid}")

        # Use the standalone probe streaming helper but with ref_file_ids patched in.
        # Re-implementing here to pass ref_file_ids.
        from probe.probe import get_pow as gp
        import json as _json

        pow2 = await gp(c, state, "/api/v0/chat/completion")
        body = {
            "chat_session_id": sid,
            "parent_message_id": None,
            "prompt": "What's the codeword and due date in the uploaded file?",
            "ref_file_ids": [file_id],
            "thinking_enabled": False,
            "search_enabled": False,
        }
        from probe.probe import headers as probe_headers

        async with c.stream(
            "POST",
            f"{BASE}/chat/completion",
            headers=probe_headers(state["userToken"], pow2),
            cookies=cookies_dict(state),
            json=body,
        ) as r:
            print(f"[stream] {r.status_code}")
            async for line in r.aiter_lines():
                if line.startswith("data:"):
                    try:
                        obj = _json.loads(line[5:].strip())
                    except Exception:
                        continue
                    if obj.get("p") == "response/content" or (
                        obj.get("p") is None and isinstance(obj.get("v"), str)
                    ):
                        print(obj.get("v", ""), end="", flush=True)
                elif line.startswith("event: finish"):
                    print("\n[done]")
                    break


if __name__ == "__main__":
    asyncio.run(main())
