"""Truncate oversized tool_result payloads before they hit upstream.

Large bash output and file dumps blow up the prompt and can trip upstream
timeouts. We keep head + tail so the model sees both how the command started
and how it ended; the middle becomes a visible marker so the model knows data
was elided rather than malformed.
"""
from __future__ import annotations

import os

DEFAULT_MAX_BYTES = int(os.environ.get("DS_TOOL_RESULT_MAX_BYTES", "24000"))
_HEAD_RATIO = 0.75


def prune_tool_result(text: str, max_bytes: int | None = None) -> str:
    limit = max_bytes if max_bytes is not None else DEFAULT_MAX_BYTES
    if limit <= 0:
        return text
    data = text.encode("utf-8", errors="replace")
    if len(data) <= limit:
        return text
    marker_tmpl = "\n\n[... truncated {n} bytes ...]\n\n"
    # Reserve room for the marker itself.
    sample_marker = marker_tmpl.format(n=len(data))
    budget = max(1, limit - len(sample_marker.encode()))
    head_bytes = int(budget * _HEAD_RATIO)
    tail_bytes = budget - head_bytes
    dropped = len(data) - head_bytes - tail_bytes
    head = data[:head_bytes].decode("utf-8", errors="replace")
    tail = data[-tail_bytes:].decode("utf-8", errors="replace") if tail_bytes > 0 else ""
    return head + marker_tmpl.format(n=dropped) + tail
