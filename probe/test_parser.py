"""Unit tests for ToolCallParser. Run:

    .venv/bin/python -m probe.test_parser
    BACKEND=zai .venv/bin/python -m probe.test_parser

Backend-aware tests: when BACKEND=zai we expect the parser to recover
tool calls even when Z.AI's stream filter has stripped the surrounding
tags (so we receive only bare JSON, possibly with leading garbage chars).
"""
from __future__ import annotations

import json
import sys
from typing import Iterable

from app.config import BACKEND
from app.tools.parser import OPEN, CLOSE, ToolCallParser, serialize_tool_calls


# ---- helpers ----

def feed_chunks(chunks: Iterable[str]) -> list[dict]:
    p = ToolCallParser()
    out: list[dict] = []
    for c in chunks:
        out.extend(list(p.feed(c)))
    out.extend(list(p.flush()))
    return out


def starts(events: list[dict]) -> list[dict]:
    return [e for e in events if e["type"] == "tool_call_start"]


def deltas(events: list[dict]) -> list[str]:
    return [e["delta"] for e in events if e["type"] == "tool_call_arg_delta"]


def ends(events: list[dict]) -> list[dict]:
    return [e for e in events if e["type"] == "tool_call_end"]


def texts(events: list[dict]) -> str:
    return "".join(e["text"] for e in events if e["type"] == "text")


def assert_eq(a, b, msg=""):
    if a != b:
        raise AssertionError(f"{msg}\n  expected: {b!r}\n  actual:   {a!r}")


# ---- shared cases (both backends) ----

def test_clean_envelope_single_chunk():
    body = '{"name":"set_color","arguments":{"color":"red"}}'
    text = f"hi {OPEN}\n{body}\n{CLOSE} bye"
    events = feed_chunks([text])
    assert_eq(starts(events)[0]["name"], "set_color", "name")
    assert_eq(ends(events)[0]["arguments"], {"color": "red"}, "args dict")
    assert_eq(json.loads("".join(deltas(events))), {"color": "red"}, "streamed JSON")
    assert_eq(texts(events), "hi  bye", "surrounding text")


def test_clean_envelope_chunked():
    chunks = [
        f"prefix {OPEN}",
        '\n{"na', 'me":"Wri',
        'te","argu', 'ments":{"file_path":"/tmp/x',
        '","content":"hello"}}\n',
        f"{CLOSE} suffix",
    ]
    events = feed_chunks(chunks)
    assert_eq(len(starts(events)), 1, "one start")
    assert_eq(ends(events)[0]["arguments"]["file_path"], "/tmp/x", "file_path")
    streamed = "".join(deltas(events))
    assert_eq(json.loads(streamed)["content"], "hello", "streamed content")


def test_multiple_tool_calls():
    body = (
        f"{OPEN}\n" + '{"name":"A","arguments":{"x":1}}' + f"\n{CLOSE}"
        f"{OPEN}\n" + '{"name":"B","arguments":{"y":2}}' + f"\n{CLOSE}"
    )
    events = feed_chunks([body])
    names = [s["name"] for s in starts(events)]
    assert_eq(names, ["A", "B"], "both names")
    args = [e["arguments"] for e in ends(events)]
    assert_eq(args, [{"x": 1}, {"y": 2}], "both arg dicts")


def test_no_tool_call_passes_through():
    events = feed_chunks(["hello world ", "this is plain text"])
    assert_eq(texts(events), "hello world this is plain text", "plain")
    assert_eq(starts(events), [], "no starts")


def test_serialize_round_trip():
    tcs = [
        {"name": "foo", "arguments": {"a": 1, "b": "x"}},
        {"name": "bar", "arguments": {}},
    ]
    text = serialize_tool_calls(tcs)
    events = feed_chunks([text])
    assert_eq([s["name"] for s in starts(events)], ["foo", "bar"], "round-trip names")
    assert_eq(
        [e["arguments"] for e in ends(events)],
        [{"a": 1, "b": "x"}, {}],
        "round-trip args",
    )


# ---- Z.AI-only cases (bare-JSON fallback after upstream tag strip) ----

def test_zai_bare_json_recovered():
    """Z.AI strips <tool_call> tags. Parser must recover the bare JSON."""
    if BACKEND != "zai":
        return  # skip on deepseek; bare JSON is text there
    text = '{"name":"set_color","arguments":{"color":"periwinkle"}}'
    events = feed_chunks([text])
    assert_eq(starts(events)[0]["name"], "set_color", "bare JSON name")
    assert_eq(
        ends(events)[0]["arguments"],
        {"color": "periwinkle"},
        "bare JSON args",
    )


def test_zai_bare_json_with_leading_garbage():
    """Sometimes Z.AI leaves a single leftover char from the stripped tag."""
    if BACKEND != "zai":
        return
    text = 'E{"name":"set_color","arguments":{"color":"red"}}'
    events = feed_chunks([text])
    assert_eq(starts(events)[0]["name"], "set_color", "bare JSON despite prefix")
    assert_eq(ends(events)[0]["arguments"], {"color": "red"}, "args")


def test_zai_bare_json_chunked():
    if BACKEND != "zai":
        return
    chunks = [
        '{"name":', '"Write",',
        '"arguments":', '{"file_path":"/tmp/y",',
        '"content":"hi"}}',
    ]
    events = feed_chunks(chunks)
    assert_eq(starts(events)[0]["name"], "Write", "chunked name")
    assert_eq(
        ends(events)[0]["arguments"],
        {"file_path": "/tmp/y", "content": "hi"},
        "chunked args",
    )


def test_zai_bare_json_with_surrounding_text():
    if BACKEND != "zai":
        return
    text = 'I will call: {"name":"X","arguments":{"a":1}} done.'
    events = feed_chunks([text])
    assert_eq(starts(events)[0]["name"], "X", "bare JSON inside prose")
    # Surrounding text outside the JSON should still appear:
    surrounding = texts(events)
    assert "I will call" in surrounding, f"prefix missing in: {surrounding!r}"
    assert "done" in surrounding, f"suffix missing in: {surrounding!r}"


def test_zai_bare_json_streams_incrementally():
    """Bare-mode must stream args deltas as chunks arrive, not buffer-then-dump."""
    if BACKEND != "zai":
        return
    p = ToolCallParser()
    chunks = [
        '{"name":"Edit",',
        '"arguments":{"file_path"',
        ':"/tmp/a","old',
        '":"foo","new":"bar"}}',
    ]
    seen_start_before_args = False
    deltas_seen: list[str] = []
    end_seen = False
    for c in chunks:
        for ev in p.feed(c):
            if ev["type"] == "tool_call_start":
                if not deltas_seen:
                    seen_start_before_args = True
            elif ev["type"] == "tool_call_arg_delta":
                deltas_seen.append(ev["delta"])
            elif ev["type"] == "tool_call_end":
                end_seen = True
    for ev in p.flush():
        if ev["type"] == "tool_call_end":
            end_seen = True
    assert seen_start_before_args, "tool_call_start should fire before any arg_delta"
    assert end_seen, "tool_call_end must fire"
    assert len(deltas_seen) >= 2, f"want streaming deltas, got {len(deltas_seen)}: {deltas_seen}"
    streamed = "".join(deltas_seen)
    parsed = json.loads(streamed)
    assert_eq(
        parsed,
        {"file_path": "/tmp/a", "old": "foo", "new": "bar"},
        "streamed JSON reconstructs",
    )


def test_zai_plain_text_not_misparsed():
    """Bare-JSON fallback must not eat ordinary JSON-looking objects in text."""
    if BACKEND != "zai":
        return
    # Object without `name` + `arguments` keys → must stay as text.
    text = 'Here is data: {"foo":"bar","count":3} ok.'
    events = feed_chunks([text])
    assert_eq(starts(events), [], "no spurious tool call")
    assert_eq(texts(events), text, "passthrough preserved")


# ---- runner ----

def main() -> int:
    tests = [v for k, v in globals().items() if k.startswith("test_") and callable(v)]
    failures: list[tuple[str, str]] = []
    for fn in tests:
        try:
            fn()
            print(f"[OK]   {fn.__name__}")
        except AssertionError as e:
            failures.append((fn.__name__, str(e)))
            print(f"[FAIL] {fn.__name__}: {e}")
        except Exception as e:
            failures.append((fn.__name__, repr(e)))
            print(f"[ERR]  {fn.__name__}: {e!r}")
    print()
    print(
        f"backend={BACKEND}: {len(tests) - len(failures)}/{len(tests)} passed "
        f"(envelope: OPEN={OPEN!r} CLOSE={CLOSE!r})"
    )
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
