"""File upload passthrough to chat.deepseek.com /file/upload_file."""
from __future__ import annotations

import asyncio
import json
from typing import Any

import httpx

RATE_LIMIT_CODE = 7

from app.config import BASE_URL, STATE_DIR
from app.deepseek import auth
from app.deepseek.client import _headers  # reuse header builder
from app.deepseek.pow import solve_challenge

FILES_MAP = STATE_DIR / "files.json"
_lock = asyncio.Lock()


def _load() -> dict[str, dict]:
    if not FILES_MAP.exists():
        return {}
    try:
        return json.loads(FILES_MAP.read_text())
    except json.JSONDecodeError:
        return {}


def _save(data: dict[str, dict]) -> None:
    tmp = FILES_MAP.with_suffix(".tmp")
    tmp.write_text(json.dumps(data))
    tmp.replace(FILES_MAP)


async def _pow(http: httpx.AsyncClient, state: dict, target: str) -> str:
    r = await http.post(
        f"{BASE_URL}/chat/create_pow_challenge",
        headers=_headers(state["userToken"]),
        cookies=auth.cookies_dict(state),
        json={"target_path": target},
    )
    r.raise_for_status()
    ch = r.json()["data"]["biz_data"]["challenge"]
    ch["target_path"] = target
    return solve_challenge(ch)


async def upload(http: httpx.AsyncClient, filename: str, content: bytes, mime: str) -> dict[str, Any]:
    """Upload to DeepSeek; return {id, file_name, file_size, status} from upstream."""
    state = await auth.get_state()
    target = "/api/v0/file/upload_file"
    last_body: dict | None = None
    for attempt in range(4):
        pow_resp = await _pow(http, state, target)
        # Drop application/json content-type so httpx can set multipart/form-data itself
        hdrs = {k: v for k, v in _headers(state["userToken"], pow_resp).items() if k != "content-type"}
        r = await http.post(
            f"{BASE_URL}/file/upload_file",
            headers=hdrs,
            cookies=auth.cookies_dict(state),
            files={"file": (filename, content, mime)},
        )
        if r.status_code == 429:
            await asyncio.sleep(2 ** attempt + 1)
            continue
        r.raise_for_status()
        body = r.json()
        last_body = body
        biz_code = (body.get("data") or {}).get("biz_code")
        if biz_code == RATE_LIMIT_CODE:
            await asyncio.sleep(2 ** attempt + 1)
            continue
        biz = (body.get("data") or {}).get("biz_data")
        if not biz:
            raise RuntimeError(f"upload failed: {body}")
        return biz
    raise RuntimeError(f"upload rate-limited after retries: {last_body}")


async def store_mapping(openai_id: str, info: dict[str, Any], content: bytes | None = None) -> None:
    async with _lock:
        data = _load()
        if content is not None:
            # Cache original bytes so we can inline small text files as fallback
            try:
                info = {**info, "text": content.decode("utf-8")}
            except UnicodeDecodeError:
                pass
        data[openai_id] = info
        _save(data)


async def get_mapping(openai_id: str) -> dict[str, Any] | None:
    async with _lock:
        return _load().get(openai_id)


async def list_mappings() -> list[dict[str, Any]]:
    async with _lock:
        return list(_load().values())


async def delete_mapping(openai_id: str) -> bool:
    async with _lock:
        data = _load()
        if openai_id not in data:
            return False
        del data[openai_id]
        _save(data)
        return True
