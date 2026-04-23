"""Anthropic Messages API /v1/messages — stream + non-stream."""
from __future__ import annotations

import json
import time
import uuid
from typing import Any, AsyncIterator, Literal

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, ConfigDict

from app.config import PROXY_API_KEY
from app.deepseek import files as ds_files
from app.deepseek import sessions
from app.deepseek.client import DeepSeekClient
from app.routes.openai_chat import (
    flatten_prefix,
    prepend_system,
    resolve_session,
    _cache_turn,
)
from app.tools.inject import normalize_anthropic_tools, tool_system_block
from app.tools.parser import ToolCallParser
from app.tools.prune import prune_tool_result

router = APIRouter()


def _estimate_tokens(text: str) -> int:
    # Rough 1 token ≈ 4 chars; good enough for Claude Code usage display.
    if not text:
        return 0
    return max(1, len(text) // 4)


def _require_key(request: Request):
    if not PROXY_API_KEY:
        return
    got = request.headers.get("x-api-key") or request.headers.get(
        "authorization", ""
    ).removeprefix("Bearer ").strip()
    if got != PROXY_API_KEY:
        raise HTTPException(401, "invalid api key")


# ---- schema ----

class AnthropicMessage(BaseModel):
    model_config = ConfigDict(extra="ignore")
    role: Literal["user", "assistant"]
    content: str | list[dict]


class AnthropicRequest(BaseModel):
    # Claude Code sends many optional fields we don't need to act on.
    # `extra=ignore` keeps the proxy from 422'ing on new/beta params.
    model_config = ConfigDict(extra="ignore")
    model: str
    messages: list[AnthropicMessage]
    system: str | list[dict] | None = None
    max_tokens: int = 1024
    stream: bool = False
    temperature: float | None = None
    top_p: float | None = None
    top_k: int | None = None
    stop_sequences: list[str] | None = None
    metadata: dict | None = None
    service_tier: str | None = None
    tool_choice: dict | None = None
    tools: list[dict] | None = None
    thinking: dict | None = None  # {"type":"enabled", "budget_tokens":...}


def _anthropic_system_text(system: Any) -> str:
    if system is None:
        return ""
    if isinstance(system, str):
        return system
    parts = []
    for p in system:
        if isinstance(p, dict) and p.get("type") == "text":
            parts.append(p.get("text", ""))
    return "\n\n".join(parts)


def _flatten_block(block: dict) -> str:
    t = block.get("type")
    if t == "text":
        return block.get("text", "")
    if t == "tool_use":
        return (
            f"<tool_call>\n"
            f"{json.dumps({'name': block.get('name', ''), 'arguments': block.get('input', {})})}"
            f"\n</tool_call>"
        )
    if t == "tool_result":
        content = block.get("content", "")
        if isinstance(content, list):
            parts: list[str] = []
            for c in content:
                if not isinstance(c, dict):
                    continue
                ct = c.get("type")
                if ct == "text":
                    parts.append(c.get("text", ""))
                elif ct == "image":
                    parts.append("[image omitted]")
            content = "\n".join(p for p in parts if p)
        err = " error" if block.get("is_error") else ""
        tid = block.get("tool_use_id", "")
        return f"<tool_result id=\"{tid}\"{err}>\n{prune_tool_result(content)}\n</tool_result>"
    if t == "thinking" or t == "redacted_thinking":
        return ""  # don't replay thinking to the model
    if t == "image":
        return "[image omitted]"
    if t == "document":
        # File is passed separately via ref_file_ids; hint in prompt.
        src = block.get("source", {})
        if isinstance(src, dict) and src.get("type") == "file":
            return "[See attached document.]"
        return "[document omitted]"
    return ""


async def _collect_ref_files(req: "AnthropicRequest") -> list[str]:
    out: list[str] = []
    for m in req.messages:
        if not isinstance(m.content, list):
            continue
        for b in m.content:
            if not isinstance(b, dict):
                continue
            if b.get("type") == "document":
                src = b.get("source", {})
                fid = src.get("file_id") if isinstance(src, dict) else None
                if not fid:
                    continue
                info = await ds_files.get_mapping(fid)
                if info and info.get("deepseek_file_id"):
                    out.append(info["deepseek_file_id"])
                elif fid.startswith("file-"):
                    out.append(fid)
    return out


def _anthropic_to_turns(req: AnthropicRequest) -> list[tuple[str, str]]:
    turns: list[tuple[str, str]] = []
    sys_text = _anthropic_system_text(req.system)
    if sys_text:
        turns.append(("system", sys_text))
    for m in req.messages:
        if isinstance(m.content, str):
            turns.append((m.role, m.content))
            continue
        parts = [_flatten_block(b) for b in m.content if isinstance(b, dict)]
        text = "\n".join(p for p in parts if p).strip()
        if text:
            turns.append((m.role, text))
    return turns


# ---- SSE helpers ----

def _sse(event: str, data: dict) -> bytes:
    return f"event: {event}\ndata: {json.dumps(data, separators=(',', ':'))}\n\n".encode()


@router.post("/v1/messages/count_tokens")
async def count_tokens(req: AnthropicRequest, _: None = Depends(_require_key)):
    # Claude Code calls this to size prompts before sending. Estimate from
    # the flattened prompt since the upstream doesn't expose a tokenizer.
    turns = _anthropic_to_turns(req)
    body = "\n\n".join(t for _, t in turns)
    tools_text = json.dumps(req.tools or [])
    return {"input_tokens": _estimate_tokens(body) + _estimate_tokens(tools_text)}


@router.post("/v1/messages")
async def messages(req: AnthropicRequest, request: Request, _: None = Depends(_require_key)):
    # Search: :search suffix on model name; thinking: client-requested or reasoner model
    search = req.model.endswith(":search")
    base_model = req.model[:-len(":search")] if search else req.model
    is_reasoner = "reasoner" in base_model or "thinking" in base_model or "opus" in base_model
    thinking_enabled = bool(req.thinking and req.thinking.get("type") == "enabled") or is_reasoner

    turns = _anthropic_to_turns(req)

    tools_norm = normalize_anthropic_tools(req.tools or [])
    # tool_choice: {"type":"any"} ~ "required"; {"type":"tool","name":...} forces one
    force_note = ""
    if isinstance(req.tool_choice, dict):
        tc_type = req.tool_choice.get("type")
        if tc_type == "none":
            tools_norm = []
        elif tc_type == "any":
            force_note = "\n\nYou MUST call one of the tools above. Do not answer directly."
        elif tc_type == "tool" and req.tool_choice.get("name"):
            force_note = (
                f"\n\nYou MUST call the `{req.tool_choice['name']}` tool. "
                "Do not answer directly."
            )
    if tools_norm:
        block = tool_system_block(tools_norm) + force_note
        turns = prepend_system(turns, block)

    # Inline cached text-file content into the last user turn (reliability)
    file_text_parts: list[str] = []
    for m in req.messages:
        if not isinstance(m.content, list):
            continue
        for b in m.content:
            if not isinstance(b, dict):
                continue
            if b.get("type") == "document":
                src = b.get("source", {})
                fid = src.get("file_id") if isinstance(src, dict) else None
                if not fid:
                    continue
                info = await ds_files.get_mapping(fid)
                if info and info.get("text"):
                    file_text_parts.append(
                        f"--- FILE: {info.get('filename', fid)} ---\n{info['text']}\n--- END FILE ---"
                    )
    if file_text_parts and turns:
        ft = "\n\n".join(file_text_parts)
        role, content = turns[-1]
        if role == "user":
            turns = turns[:-1] + [(role, f"{ft}\n\n{content}")]
        else:
            turns = turns + [("user", ft)]

    client: DeepSeekClient = request.app.state.ds
    alias = request.headers.get("x-ds-session")
    session_id, parent_id, prompt = await resolve_session(client, turns, alias=alias)
    # Cache-hit => prompt is just the new user turn, which means DeepSeek no
    # longer sees the tool contract we injected into the flattened transcript.
    # Re-prepend it so the <tool_call> protocol survives across turns.
    if tools_norm and turns and prompt == turns[-1][1]:
        prompt = tool_system_block(tools_norm) + force_note + "\n\n" + prompt
    ref_file_ids = await _collect_ref_files(req)
    has_tools = bool(tools_norm)

    if req.stream:
        return StreamingResponse(
            _stream_anthropic(
                client, session_id, parent_id, turns, prompt, thinking_enabled, search,
                req.model, has_tools, ref_file_ids,
            ),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )
    return JSONResponse(
        await _buffered_anthropic(
            client, session_id, parent_id, turns, prompt, thinking_enabled, search,
            req.model, has_tools, ref_file_ids,
        )
    )


async def _stream_anthropic(
    client: DeepSeekClient,
    session_id: str,
    parent_id: int | None,
    turns: list[tuple[str, str]],
    prompt: str,
    thinking: bool,
    search: bool,
    model: str,
    has_tools: bool,
    ref_file_ids: list[str] | None = None,
) -> AsyncIterator[bytes]:
    msg_id = f"msg_{uuid.uuid4().hex[:24]}"
    input_tokens = _estimate_tokens(prompt)
    yield _sse(
        "message_start",
        {
            "type": "message_start",
            "message": {
                "id": msg_id,
                "type": "message",
                "role": "assistant",
                "content": [],
                "model": model,
                "stop_reason": None,
                "stop_sequence": None,
                "usage": {
                    "input_tokens": input_tokens,
                    "cache_creation_input_tokens": 0,
                    "cache_read_input_tokens": 0,
                    "output_tokens": 0,
                },
            },
        },
    )
    # Anthropic clients expect periodic `ping` events to keep the connection
    # alive during long reasoner waits. One right after start + a background
    # beat every ~10s.
    yield _sse("ping", {"type": "ping"})
    last_ping = time.time()

    parser = ToolCallParser() if has_tools else None
    block_idx = -1
    current_block: str | None = None  # "thinking" | "text" | "tool_use"
    assistant_buf: list[str] = []
    tool_calls: list[dict] = []
    stop_reason = "end_turn"
    final_msg_id: int | None = None

    def open_block(kind: str, extra: dict) -> bytes:
        nonlocal block_idx, current_block
        block_idx += 1
        current_block = kind
        return _sse(
            "content_block_start",
            {"type": "content_block_start", "index": block_idx, "content_block": extra},
        )

    def close_block() -> bytes:
        nonlocal current_block
        if current_block is None:
            return b""
        out = _sse("content_block_stop", {"type": "content_block_stop", "index": block_idx})
        current_block = None
        return out

    try:
        async for ev in client.stream_completion(
            session_id=session_id,
            prompt=prompt,
            parent_message_id=parent_id,
            thinking=thinking,
            search=search,
            ref_file_ids=ref_file_ids,
        ):
            if ev["type"] == "thinking":
                if current_block != "thinking":
                    if current_block:
                        yield close_block()
                    yield open_block("thinking", {"type": "thinking", "thinking": ""})
                yield _sse(
                    "content_block_delta",
                    {
                        "type": "content_block_delta",
                        "index": block_idx,
                        "delta": {"type": "thinking_delta", "thinking": ev["text"]},
                    },
                )
                continue
            if ev["type"] == "done":
                final_msg_id = ev.get("message_id")
                if parser:
                    for pev in parser.flush():
                        if pev["type"] == "text":
                            if current_block != "text":
                                if current_block:
                                    yield close_block()
                                yield open_block("text", {"type": "text", "text": ""})
                            assistant_buf.append(pev["text"])
                            yield _sse(
                                "content_block_delta",
                                {
                                    "type": "content_block_delta",
                                    "index": block_idx,
                                    "delta": {"type": "text_delta", "text": pev["text"]},
                                },
                            )
                if current_block:
                    yield close_block()
                if tool_calls:
                    stop_reason = "tool_use"
                output_tokens = _estimate_tokens("".join(assistant_buf))
                yield _sse(
                    "message_delta",
                    {
                        "type": "message_delta",
                        "delta": {"stop_reason": stop_reason, "stop_sequence": None},
                        "usage": {"output_tokens": output_tokens},
                    },
                )
                yield _sse("message_stop", {"type": "message_stop"})
                break
            if ev["type"] != "content":
                # Keep connection warm on search_status / idle frames.
                now_t = time.time()
                if now_t - last_ping > 10:
                    last_ping = now_t
                    yield _sse("ping", {"type": "ping"})
                continue
            now_t = time.time()
            if now_t - last_ping > 10:
                last_ping = now_t
                yield _sse("ping", {"type": "ping"})

            text = ev["text"]
            if not parser:
                if current_block == "thinking":
                    yield close_block()
                if current_block != "text":
                    yield open_block("text", {"type": "text", "text": ""})
                assistant_buf.append(text)
                yield _sse(
                    "content_block_delta",
                    {
                        "type": "content_block_delta",
                        "index": block_idx,
                        "delta": {"type": "text_delta", "text": text},
                    },
                )
                continue

            for pev in parser.feed(text):
                if pev["type"] == "text":
                    if current_block == "thinking":
                        yield close_block()
                    if current_block != "text":
                        yield open_block("text", {"type": "text", "text": ""})
                    assistant_buf.append(pev["text"])
                    yield _sse(
                        "content_block_delta",
                        {
                            "type": "content_block_delta",
                            "index": block_idx,
                            "delta": {"type": "text_delta", "text": pev["text"]},
                        },
                    )
                elif pev["type"] == "tool_call_start":
                    if current_block:
                        yield close_block()
                    yield open_block(
                        "tool_use",
                        {
                            "type": "tool_use",
                            "id": pev["id"],
                            "name": pev["name"],
                            "input": {},
                        },
                    )
                elif pev["type"] == "tool_call_arg_delta":
                    if current_block != "tool_use":
                        continue
                    yield _sse(
                        "content_block_delta",
                        {
                            "type": "content_block_delta",
                            "index": block_idx,
                            "delta": {
                                "type": "input_json_delta",
                                "partial_json": pev["delta"],
                            },
                        },
                    )
                elif pev["type"] == "tool_call_end":
                    if current_block == "tool_use":
                        yield close_block()
                    tool_calls.append(pev)
    except Exception as e:
        yield _sse(
            "error",
            {"type": "error", "error": {"type": "upstream_error", "message": str(e)}},
        )
    else:
        await _cache_turn(turns, "".join(assistant_buf), session_id, final_msg_id)


async def _buffered_anthropic(
    client: DeepSeekClient,
    session_id: str,
    parent_id: int | None,
    turns: list[tuple[str, str]],
    prompt: str,
    thinking: bool,
    search: bool,
    model: str,
    has_tools: bool,
    ref_file_ids: list[str] | None = None,
) -> dict:
    content_parts: list[str] = []
    thinking_parts: list[str] = []
    tool_calls: list[dict] = []
    message_id: int | None = None
    parser = ToolCallParser() if has_tools else None

    async for ev in client.stream_completion(
        session_id=session_id,
        prompt=prompt,
        parent_message_id=parent_id,
        thinking=thinking,
        search=search,
        ref_file_ids=ref_file_ids,
    ):
        if ev["type"] == "thinking":
            thinking_parts.append(ev["text"])
        elif ev["type"] == "content":
            if not parser:
                content_parts.append(ev["text"])
            else:
                for pev in parser.feed(ev["text"]):
                    if pev["type"] == "text":
                        content_parts.append(pev["text"])
                    elif pev["type"] == "tool_call":
                        tool_calls.append(pev)
        elif ev["type"] == "done":
            message_id = ev["message_id"]
            break
    if parser:
        for pev in parser.flush():
            if pev["type"] == "text":
                content_parts.append(pev["text"])

    content = "".join(content_parts)
    await _cache_turn(turns, content, session_id, message_id)

    blocks: list[dict] = []
    if thinking_parts:
        blocks.append({"type": "thinking", "thinking": "".join(thinking_parts)})
    if content:
        blocks.append({"type": "text", "text": content})
    for tc in tool_calls:
        blocks.append(
            {"type": "tool_use", "id": tc["id"], "name": tc["name"], "input": tc["arguments"]}
        )

    input_tokens = _estimate_tokens(prompt)
    output_tokens = _estimate_tokens(content) + _estimate_tokens("".join(thinking_parts))
    return {
        "id": f"msg_{uuid.uuid4().hex[:24]}",
        "type": "message",
        "role": "assistant",
        "content": blocks,
        "model": model,
        "stop_reason": "tool_use" if tool_calls else "end_turn",
        "stop_sequence": None,
        "usage": {
            "input_tokens": input_tokens,
            "cache_creation_input_tokens": 0,
            "cache_read_input_tokens": 0,
            "output_tokens": output_tokens,
        },
    }
