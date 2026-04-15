"""
BaseAgent — the abstract foundation every Storibot agent inherits from.

The Managed Agents API integration lives here:
  • System-prompt caching via cache_control (saves tokens on every call)
  • Tool-use agent loop that runs until stop_reason == "end_turn"
  • A mandatory `submit_result` tool that forces Claude to return structured
    JSON rather than free text — 100% reliable structured output
  • Subclasses only need to define agent_id, system_prompt, result_schema,
    additional_tools, and execute()
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

import anthropic

from session import SessionState


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------


class BaseAgent(ABC):
    """Abstract base for all Storibot agents.

    Concrete agents implement:
        agent_id        – stable string identifier
        system_prompt   – static prompt (cached server-side automatically)
        result_schema   – JSON Schema for the submit_result tool input
        additional_tools – extra tool definitions the agent may call
        execute()       – maps (action, payload, session) → result dict
    """

    MODEL = "claude-sonnet-4-6"
    MAX_TOKENS = 4096
    MAX_LOOP_ITERATIONS = 8

    def __init__(self, client: anthropic.Anthropic) -> None:
        self.client = client

    # ---------------------------------------------------------------------------
    # Abstract interface
    # ---------------------------------------------------------------------------

    @property
    @abstractmethod
    def agent_id(self) -> str:
        ...

    @property
    @abstractmethod
    def system_prompt(self) -> str:
        ...

    @property
    @abstractmethod
    def result_schema(self) -> Dict[str, Any]:
        """JSON Schema object for the submit_result tool's `result` property."""
        ...

    @property
    def additional_tools(self) -> List[Dict[str, Any]]:
        """Override to add agent-specific tools alongside submit_result."""
        return []

    @abstractmethod
    def execute(
        self,
        action: str,
        payload: Dict[str, Any],
        session: Optional[SessionState],
    ) -> Dict[str, Any]:
        """Run the agent for the given action and payload.

        Must return a plain dict; the orchestrator attaches _meta to it.
        """
        ...

    # ---------------------------------------------------------------------------
    # Managed Agents API — tool definitions
    # ---------------------------------------------------------------------------

    @property
    def _submit_result_tool(self) -> Dict[str, Any]:
        """The `submit_result` tool forces structured output from Claude.

        By making this the only way to terminate the loop Claude cannot
        skip the schema — we get reliable JSON on every call.
        """
        return {
            "name": "submit_result",
            "description": (
                "Call this tool ONCE with your final structured result when you have "
                "finished your analysis. This is the only way to complete your task."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "result": self.result_schema,
                },
                "required": ["result"],
            },
        }

    @property
    def _all_tools(self) -> List[Dict[str, Any]]:
        return [self._submit_result_tool] + self.additional_tools

    # ---------------------------------------------------------------------------
    # Managed Agents API — core loop
    # ---------------------------------------------------------------------------

    def _run(self, user_message: str) -> tuple[Dict[str, Any], int]:
        """Execute the Managed Agents API loop.

        Returns (result_dict, total_tokens_used).

        The loop:
          1. Send user message with cached system prompt + tools
          2. If Claude calls submit_result → capture and return
          3. If Claude calls another tool → dispatch to _handle_tool()
          4. Continue until submit_result is called or max iterations
        """
        messages: List[Dict[str, Any]] = [{"role": "user", "content": user_message}]
        total_tokens = 0

        for _ in range(self.MAX_LOOP_ITERATIONS):
            response = self.client.messages.create(
                model=self.MODEL,
                max_tokens=self.MAX_TOKENS,
                # Cache the (static) system prompt — saves input tokens on every
                # subsequent call to the same agent within a process lifetime.
                system=[
                    {
                        "type": "text",
                        "text": self.system_prompt,
                        "cache_control": {"type": "ephemeral"},
                    }
                ],
                tools=self._all_tools,
                messages=messages,
            )

            total_tokens += response.usage.input_tokens + response.usage.output_tokens

            # --- handle stop conditions -----------------------------------
            if response.stop_reason == "end_turn":
                # Claude answered in plain text — extract and wrap it
                text = next(
                    (b.text for b in response.content if hasattr(b, "text")), ""
                )
                return self._parse_text_fallback(text), total_tokens

            if response.stop_reason == "tool_use":
                # Append assistant turn
                messages.append({"role": "assistant", "content": response.content})

                tool_results: List[Dict[str, Any]] = []
                for block in response.content:
                    if block.type != "tool_use":
                        continue

                    if block.name == "submit_result":
                        # Structured output received — done
                        result = block.input.get("result", {})
                        result["_total_tokens"] = total_tokens
                        return result, total_tokens

                    # Dispatch to agent-specific tool handler
                    tool_output = self._handle_tool(block.name, block.input)
                    tool_results.append(
                        {
                            "type": "tool_result",
                            "tool_use_id": block.id,
                            "content": json.dumps(tool_output),
                        }
                    )

                messages.append({"role": "user", "content": tool_results})

        # Exhausted iterations without a submit_result call
        return {"error": "Agent loop exhausted without a final result"}, total_tokens

    # ---------------------------------------------------------------------------
    # Helpers
    # ---------------------------------------------------------------------------

    def _handle_tool(self, tool_name: str, tool_input: Dict[str, Any]) -> Any:
        """Override in subclasses to handle additional tools."""
        return {"error": f"Unknown tool: {tool_name}"}

    def _parse_text_fallback(self, text: str) -> Dict[str, Any]:
        """Best-effort parse when Claude returns plain text instead of a tool call."""
        try:
            if "```json" in text:
                start = text.index("```json") + 7
                end = text.index("```", start)
                return json.loads(text[start:end].strip())
            if "```" in text:
                start = text.index("```") + 3
                end = text.index("```", start)
                return json.loads(text[start:end].strip())
            return json.loads(text)
        except (json.JSONDecodeError, ValueError):
            return {"raw_response": text}

    def _build_prompt(self, action: str, payload: Dict[str, Any]) -> str:
        """Produce the user-turn message sent to Claude."""
        payload_json = json.dumps(payload, indent=2, default=str)
        return f"Action: {action}\n\nPayload:\n{payload_json}"
