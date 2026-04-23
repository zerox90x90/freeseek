"""Disk-backed cache: conversation-prefix hash -> {session_id, parent_message_id}.

Lets multi-turn clients reuse DeepSeek's own context across requests instead of
replaying the whole history each call.

Reserved keys (never treated as prefix hashes):
  _aliases: {name: prefix_hash}  — human-friendly session pins for API resume
  _last:    prefix_hash          — most recently written prefix hash, for
                                    quick "pin current conversation" requests
"""
from __future__ import annotations

import asyncio
import hashlib
import json
from typing import Any

from app.config import STATE_DIR

SESSIONS_FILE = STATE_DIR / "sessions.json"
_RESERVED = ("_aliases", "_last")
_lock = asyncio.Lock()


def _load() -> dict[str, Any]:
    if not SESSIONS_FILE.exists():
        return {}
    try:
        return json.loads(SESSIONS_FILE.read_text())
    except json.JSONDecodeError:
        return {}


def _save(data: dict[str, Any]) -> None:
    tmp = SESSIONS_FILE.with_suffix(".tmp")
    tmp.write_text(json.dumps(data))
    tmp.replace(SESSIONS_FILE)


def _prefix_keys(data: dict[str, Any]) -> list[str]:
    return [k for k in data.keys() if k not in _RESERVED]


def hash_turns(turns: list[tuple[str, str]]) -> str:
    """Stable hash over [(role, content), ...]."""
    h = hashlib.sha256()
    for role, content in turns:
        h.update(role.encode())
        h.update(b"\x00")
        h.update(content.encode())
        h.update(b"\x01")
    return h.hexdigest()


async def get(prefix_hash: str) -> dict[str, Any] | None:
    if prefix_hash in _RESERVED:
        return None
    async with _lock:
        entry = _load().get(prefix_hash)
        if not isinstance(entry, dict):
            return None
        if "session_id" not in entry:
            return None
        return entry


async def put(prefix_hash: str, session_id: str, parent_message_id: int | None) -> None:
    if prefix_hash in _RESERVED:
        raise ValueError(f"reserved key: {prefix_hash}")
    async with _lock:
        data = _load()
        data[prefix_hash] = {"session_id": session_id, "parent_message_id": parent_message_id}
        data["_last"] = prefix_hash
        keys = _prefix_keys(data)
        if len(keys) > 500:
            aliases = data.get("_aliases", {}) or {}
            pinned = set(aliases.values()) if isinstance(aliases, dict) else set()
            to_drop: list[str] = []
            for k in keys:
                if len(to_drop) >= 100:
                    break
                if k in pinned:
                    continue
                to_drop.append(k)
            for k in to_drop:
                del data[k]
        _save(data)


async def delete(prefix_hash: str) -> bool:
    async with _lock:
        data = _load()
        removed = data.pop(prefix_hash, None) is not None
        aliases = data.get("_aliases")
        if isinstance(aliases, dict):
            for name in [n for n, h in aliases.items() if h == prefix_hash]:
                del aliases[name]
        if removed:
            _save(data)
        return removed


async def clear_all() -> int:
    async with _lock:
        data = _load()
        count = len(_prefix_keys(data))
        _save({})
        return count


async def put_alias(name: str, prefix_hash: str | None = None) -> str:
    """Pin `name` to `prefix_hash`, or to the last-written prefix if omitted.
    Returns the resolved prefix_hash. Raises KeyError if no prefix available."""
    async with _lock:
        data = _load()
        resolved = prefix_hash or data.get("_last")
        if not resolved or resolved in _RESERVED:
            raise KeyError("no prefix hash available to alias")
        if resolved not in data:
            raise KeyError(f"unknown prefix hash: {resolved}")
        aliases = data.setdefault("_aliases", {})
        if not isinstance(aliases, dict):
            aliases = {}
            data["_aliases"] = aliases
        aliases[name] = resolved
        _save(data)
        return resolved


async def get_alias(name: str) -> dict[str, Any] | None:
    async with _lock:
        data = _load()
        aliases = data.get("_aliases", {}) or {}
        if not isinstance(aliases, dict):
            return None
        prefix_hash = aliases.get(name)
        if not prefix_hash:
            return None
        entry = data.get(prefix_hash)
        if not isinstance(entry, dict) or "session_id" not in entry:
            return None
        return {**entry, "prefix_hash": prefix_hash}


async def delete_alias(name: str, drop_entry: bool = True) -> bool:
    async with _lock:
        data = _load()
        aliases = data.get("_aliases", {}) or {}
        if not isinstance(aliases, dict) or name not in aliases:
            return False
        prefix_hash = aliases.pop(name)
        if drop_entry:
            data.pop(prefix_hash, None)
        _save(data)
        return True


async def list_all() -> dict[str, Any]:
    async with _lock:
        data = _load()
        aliases = data.get("_aliases", {}) or {}
        prefixes = _prefix_keys(data)
        return {
            "entries": len(prefixes),
            "aliases": aliases if isinstance(aliases, dict) else {},
            "last": data.get("_last"),
        }
