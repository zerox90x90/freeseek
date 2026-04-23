"""Streaming parser that filters <tool_call>...</tool_call> blocks out of the
text stream and emits structured ToolCall events.

Input: incremental text chunks from DeepSeek (the model's reply text).
Output: iterator of events:
    {"type": "text", "text": str}                      # text OUTSIDE tool_call tags
    {"type": "tool_call_start", "id": str, "name": str}
    {"type": "tool_call_arg_delta", "id": str, "delta": str}
    {"type": "tool_call_end", "id": str, "name": str, "arguments": dict}
    {"type": "tool_call", "id": str, "name": str, "arguments": dict}  # legacy composite, emitted after end

The parser emits `tool_call_start` as soon as the `"name":"..."` field is
readable inside the envelope, then streams the raw bytes of the `arguments`
JSON value as `tool_call_arg_delta` events as they arrive. This lets the
Anthropic route forward `input_json_delta` frames to the client without
waiting for the full tool call to be buffered.
"""
from __future__ import annotations

import json
import re
import uuid
from typing import Any, Iterator

OPEN = "<tool_call>"
CLOSE = "</tool_call>"

_NAME_RE = re.compile(r'"name"\s*:\s*"([^"\\]*(?:\\.[^"\\]*)*)"')
_ARGS_RE = re.compile(r'"arguments"\s*:\s*')


class ToolCallParser:
    def __init__(self) -> None:
        self._buf = ""
        self._in_tag = False
        self._raw_buf = ""
        self._name: str | None = None
        self._call_id: str | None = None
        self._args_started = False
        self._args_done = False
        self._args_depth = 0
        self._args_in_string = False
        self._args_escape = False

    def feed(self, chunk: str) -> Iterator[dict[str, Any]]:
        """Feed raw text; yield parsed events."""
        self._buf += chunk
        while True:
            if self._in_tag:
                idx = self._buf.find(CLOSE)
                if idx < 0:
                    # Don't consume the last len(CLOSE)-1 chars: they might be
                    # the start of the close tag.
                    safe = max(0, len(self._buf) - (len(CLOSE) - 1))
                    if safe > 0:
                        yield from self._consume_inside(self._buf[:safe])
                        self._buf = self._buf[safe:]
                    return
                yield from self._consume_inside(self._buf[:idx])
                self._buf = self._buf[idx + len(CLOSE):]
                yield from self._finish_tool_call()
                self._reset_tag_state()
                continue

            # Not in tag. Look for next OPEN.
            open_idx = self._buf.find(OPEN)
            if open_idx < 0:
                safe = max(0, len(self._buf) - (len(OPEN) - 1))
                if safe > 0:
                    yield {"type": "text", "text": self._buf[:safe]}
                    self._buf = self._buf[safe:]
                return
            if open_idx > 0:
                yield {"type": "text", "text": self._buf[:open_idx]}
            self._buf = self._buf[open_idx + len(OPEN):]
            self._in_tag = True
            self._call_id = f"call_{uuid.uuid4().hex[:16]}"

    def flush(self) -> Iterator[dict[str, Any]]:
        """Flush any remaining buffered text at end-of-stream."""
        if self._in_tag:
            return  # unterminated; drop
        if self._buf:
            yield {"type": "text", "text": self._buf}
            self._buf = ""

    # ---- internals ----

    def _consume_inside(self, piece: str) -> Iterator[dict[str, Any]]:
        if not piece:
            return
        prev_raw_len = len(self._raw_buf)
        self._raw_buf += piece

        if self._name is None:
            m = _NAME_RE.search(self._raw_buf)
            if m:
                self._name = m.group(1)
                yield {
                    "type": "tool_call_start",
                    "id": self._call_id,
                    "name": self._name,
                }

        if self._name is None or self._args_done:
            return

        if not self._args_started:
            m = _ARGS_RE.search(self._raw_buf)
            if not m:
                return
            self._args_started = True
            remainder = self._raw_buf[m.end():]
            yield from self._emit_args_chars(remainder)
            return

        # args already streaming; only emit the new piece
        new_piece = self._raw_buf[prev_raw_len:]
        yield from self._emit_args_chars(new_piece)

    def _emit_args_chars(self, chars: str) -> Iterator[dict[str, Any]]:
        out: list[str] = []
        for ch in chars:
            if self._args_done:
                break
            if self._args_escape:
                self._args_escape = False
                out.append(ch)
                continue
            if self._args_in_string:
                if ch == "\\":
                    self._args_escape = True
                    out.append(ch)
                    continue
                if ch == '"':
                    self._args_in_string = False
                    out.append(ch)
                    if self._args_depth == 0:
                        # primitive-string arg ended
                        self._args_done = True
                    continue
                out.append(ch)
                continue
            # not in string
            if ch == '"':
                self._args_in_string = True
                out.append(ch)
                continue
            if ch in "{[":
                self._args_depth += 1
                out.append(ch)
                continue
            if ch in "}]":
                if self._args_depth == 0:
                    # this `}` closes the envelope, not the args value
                    self._args_done = True
                    break
                self._args_depth -= 1
                out.append(ch)
                if self._args_depth == 0:
                    self._args_done = True
                continue
            out.append(ch)
        if out:
            yield {
                "type": "tool_call_arg_delta",
                "id": self._call_id,
                "delta": "".join(out),
            }

    def _finish_tool_call(self) -> Iterator[dict[str, Any]]:
        parsed = _parse_tool_call(self._raw_buf)
        if parsed is None:
            return
        # Ensure a start event was emitted even if name extraction failed earlier.
        if self._name is None:
            yield {
                "type": "tool_call_start",
                "id": self._call_id,
                "name": parsed["name"],
            }
        yield {
            "type": "tool_call_end",
            "id": self._call_id,
            "name": parsed["name"],
            "arguments": parsed["arguments"],
        }
        # Legacy composite event for routes that still consume the full call at once.
        yield {
            "type": "tool_call",
            "id": self._call_id,
            "name": parsed["name"],
            "arguments": parsed["arguments"],
        }

    def _reset_tag_state(self) -> None:
        self._in_tag = False
        self._raw_buf = ""
        self._name = None
        self._call_id = None
        self._args_started = False
        self._args_done = False
        self._args_depth = 0
        self._args_in_string = False
        self._args_escape = False


def _parse_tool_call(raw: str) -> dict[str, Any] | None:
    """Parse '{"name":..., "arguments":{...}}' into a ToolCall event."""
    m = re.search(r"\{.*\}", raw, flags=re.DOTALL)
    if not m:
        return None
    try:
        obj = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
    name = obj.get("name")
    if not isinstance(name, str):
        return None
    args = obj.get("arguments", {})
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except json.JSONDecodeError:
            pass
    return {
        "type": "tool_call",
        "id": f"call_{uuid.uuid4().hex[:16]}",
        "name": name,
        "arguments": args if isinstance(args, dict) else {},
    }
