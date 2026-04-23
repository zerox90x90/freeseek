"""Build a system-prompt contract that makes DeepSeek emit tool calls."""
from __future__ import annotations

import json
from typing import Any


TOOL_SYSTEM_TEMPLATE = """You have access to the following tools. To call a tool, emit EXACTLY this format — no code fences, no XML escaping, no commentary inside the tags:

<tool_call>
{{"name": "tool_name", "arguments": {{"arg1": "value1"}}}}
</tool_call>

Rules:
- One JSON object per <tool_call> block. You may emit multiple <tool_call> blocks in one reply to call several tools.
- `arguments` MUST be a JSON object (use {{}} when a tool needs none), never a string.
- Do NOT wrap <tool_call>...</tool_call> in markdown code fences (```), do NOT indent, do NOT prefix with "Tool:" or similar.
- After emitting tool calls, stop. Wait for tool results in the next user turn before continuing. Do not guess tool output.
- If you do NOT need a tool, answer the user directly in plain text without any <tool_call> tags.
- If the previous user message contains <tool_result id="...">...</tool_result> blocks, those are outputs of tools you previously called. READ them silently, then continue the user's original task — either by emitting more <tool_call> blocks or by writing your final answer. DO NOT repeat, echo, quote, or paraphrase the <tool_result> content back at the user.
- Never start your reply with "User:", "Assistant:", "[tool_result", "<tool_result", "[USER]", "[ASSISTANT]", or any transcript marker. Your reply is the NEXT assistant turn; do not continue or imitate the transcript format.

Available tools (JSON schema):
{tools_json}
"""


def normalize_openai_tools(tools: list[dict]) -> list[dict]:
    """Accept OpenAI tool format: {type:"function", function:{name, description, parameters}}."""
    out = []
    for t in tools:
        if t.get("type") == "function" and "function" in t:
            fn = t["function"]
            out.append(
                {
                    "name": fn.get("name", ""),
                    "description": fn.get("description", ""),
                    "parameters": fn.get("parameters", {}),
                }
            )
    return out


def normalize_anthropic_tools(tools: list[dict]) -> list[dict]:
    """Anthropic tool format: {name, description, input_schema}."""
    return [
        {
            "name": t.get("name", ""),
            "description": t.get("description", ""),
            "parameters": t.get("input_schema", {}),
        }
        for t in tools
    ]


def tool_system_block(tools: list[dict[str, Any]]) -> str:
    if not tools:
        return ""
    return TOOL_SYSTEM_TEMPLATE.format(tools_json=json.dumps(tools, indent=2))
