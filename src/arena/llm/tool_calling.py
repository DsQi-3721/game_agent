"""Dual-protocol tool calling.

Models that support native function-calling (OpenAI/Gemini/DeepSeek-chat) use the
provider's ``tools=`` path. Models that don't (many vLLM deployments, DeepSeek-R1)
use a text protocol: tool docs are injected into the prompt and the model emits a
``Tool: <name> {json args}`` line we parse. Both produce a uniform list of
``ToolInvocation`` and a uniform way to append results, so the ReAct loop is
protocol-agnostic.
"""

from __future__ import annotations

import json
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from arena.augment.base import Tool
from arena.llm.client import LLMResponse


@dataclass
class ToolInvocation:
    name: str
    arguments: dict
    id: str = ""


class ToolProtocol(ABC):
    @abstractmethod
    def complete_kwargs(self, tools: list[Tool]) -> dict: ...

    @abstractmethod
    def tool_docs(self, tools: list[Tool]) -> str: ...

    @abstractmethod
    def extract_calls(self, resp: LLMResponse) -> list[ToolInvocation]: ...

    @abstractmethod
    def append_turn(self, messages: list[dict], resp: LLMResponse,
                    calls: list[ToolInvocation], results: list[str]) -> None: ...


class NativeToolProtocol(ToolProtocol):
    def complete_kwargs(self, tools):
        return {"tools": [t.schema for t in tools]} if tools else {}

    def tool_docs(self, tools):
        return ""

    def extract_calls(self, resp):
        return [ToolInvocation(tc.name, tc.arguments, tc.id) for tc in resp.tool_calls]

    def append_turn(self, messages, resp, calls, results):
        # Re-create the assistant tool-call message, then one tool message per result.
        messages.append({
            "role": "assistant",
            "content": resp.text or None,
            "tool_calls": [
                {"id": c.id or f"call_{i}", "type": "function",
                 "function": {"name": c.name, "arguments": json.dumps(c.arguments)}}
                for i, c in enumerate(calls)
            ],
        })
        for i, (c, r) in enumerate(zip(calls, results)):
            messages.append({"role": "tool", "tool_call_id": c.id or f"call_{i}",
                             "name": c.name, "content": r})


_TOOL_LINE = re.compile(r"Tool:\s*([a-zA-Z0-9_]+)\s*(\{.*\})?", re.IGNORECASE)

# Wording of the ReAct agent; other callers (e.g. an executor that must finish with
# a <results> block) pass their own hints.
DEFAULT_FINAL_HINT = "After you have enough info, output your final `Action:` line instead of a Tool line."
DEFAULT_CONTINUE_HINT = "Now continue (call another tool or give your Action)."


class TextToolProtocol(ToolProtocol):
    def __init__(self, final_hint: str = DEFAULT_FINAL_HINT,
                 continue_hint: str = DEFAULT_CONTINUE_HINT):
        self.final_hint = final_hint
        self.continue_hint = continue_hint

    def complete_kwargs(self, tools):
        return {}

    def tool_docs(self, tools):
        if not tools:
            return ""
        lines = ["You may call tools to gather information before acting. To call a tool, "
                 "write a line:\n  Tool: <name> {\"arg\": \"value\"}\nAvailable tools:"]
        for t in tools:
            fn = t.schema.get("function", {})
            params = list(fn.get("parameters", {}).get("properties", {}))
            lines.append(f"  - {t.name}({', '.join(params)}): {fn.get('description', '')}")
        lines.append(self.final_hint)
        return "\n".join(lines)

    def extract_calls(self, resp):
        calls = []
        for m in _TOOL_LINE.finditer(resp.text or ""):
            name = m.group(1)
            try:
                args = json.loads(m.group(2)) if m.group(2) else {}
            except Exception:  # noqa: BLE001
                args = {}
            calls.append(ToolInvocation(name, args))
        return calls

    def append_turn(self, messages, resp, calls, results):
        messages.append({"role": "assistant", "content": resp.text})
        obs = "\n".join(f"[{c.name}] -> {r}" for c, r in zip(calls, results))
        messages.append({"role": "user", "content": f"Tool results:\n{obs}\n\n{self.continue_hint}"})


def make_protocol(supports_tools: bool, **text_hints) -> ToolProtocol:
    """``text_hints`` (final_hint / continue_hint) customise the text protocol only."""
    return NativeToolProtocol() if supports_tools else TextToolProtocol(**text_hints)
