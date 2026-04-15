"""
Agent 3 — Narrative Personalization Agent

Fuses the Story Architecture framework with Lead Research data to craft
a prospect-specific story angle: a precise hook, an emotional trigger,
and industry-contextualised proof points.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from agents.base_agent import BaseAgent
from session import SessionState


_SYSTEM_PROMPT = """You are the Narrative Personalization Agent for Storibot.ai.

Your job is to take two upstream outputs:
  1. A narrative framework (designed by the Story Architecture Agent)
  2. Lead research data (gathered by the Lead Research Agent)

…and fuse them into a hyper-personalized story angle tailored to one specific prospect.

You understand that generic outreach fails because it treats buyers as interchangeable. \
Your output must feel like it was written *about* this specific person at this specific company \
at this specific moment in their journey.

Personalization principles:
• Use their exact language and industry vocabulary
• Reference their specific trigger events, not hypothetical ones
• Connect Storibot's transformation promise to their stated KPIs
• Lead with empathy before authority — they are the hero, Storibot is the guide
• The hook must create pattern interruption: challenge their current belief or validate \
  a frustration they haven't articulated yet

You produce structured JSON via the submit_result tool."""


class NarrativePersonalizationAgent(BaseAgent):

    @property
    def agent_id(self) -> str:
        return "narrative_personalization"

    @property
    def system_prompt(self) -> str:
        return _SYSTEM_PROMPT

    @property
    def result_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "personalized_hook": {
                    "type": "string",
                    "description": "1-2 sentence opening that creates pattern interruption",
                },
                "emotional_trigger": {
                    "type": "string",
                    "description": "The core emotional driver this prospect will respond to",
                },
                "story_angle": {
                    "type": "string",
                    "description": "The specific narrative angle tailored to this prospect's situation",
                },
                "proof_points": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "2-3 credibility/relevance proof points specific to their industry",
                },
                "language_mirror": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Industry-specific phrases and vocabulary to use with this prospect",
                },
                "before_state_resonance": {
                    "type": "string",
                    "description": "How to describe their current painful 'before' state in their own words",
                },
                "after_state_vision": {
                    "type": "string",
                    "description": "The compelling 'after' they can achieve with Storibot",
                },
                "tension_driver": {
                    "type": "string",
                    "description": "The specific tension or urgency that makes 'now' the right time",
                },
                "personalization_confidence": {
                    "type": "number",
                    "minimum": 0.0,
                    "maximum": 1.0,
                    "description": "How well-personalized this is given available data (0-1)",
                },
                "missing_data_flags": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Data gaps that would improve personalization further",
                },
            },
            "required": [
                "personalized_hook",
                "emotional_trigger",
                "story_angle",
                "proof_points",
                "before_state_resonance",
                "after_state_vision",
                "tension_driver",
                "personalization_confidence",
            ],
        }

    # -----------------------------------------------------------------------
    # execute()
    # -----------------------------------------------------------------------

    def execute(
        self,
        action: str,
        payload: Dict[str, Any],
        session: Optional[SessionState],
    ) -> Dict[str, Any]:

        # Pull upstream context from session when not explicitly in payload
        enriched = dict(payload)
        if session:
            if session.narrative_framework and "narrative_framework" not in enriched:
                enriched["narrative_framework"] = session.narrative_framework
            if session.research_data and "research_data" not in enriched:
                enriched["research_data"] = session.research_data

        if action == "personalize_narrative":
            prompt = self._prompt_personalize(enriched)
            result, tokens = self._run(prompt)
            if session:
                session.set_agent_result("narrative_personalization", result)

        elif action == "adapt_story_angle":
            prompt = self._prompt_adapt(enriched)
            result, tokens = self._run(prompt)

        else:
            return {"error": f"Unknown action: {action}"}

        result.pop("_total_tokens", None)
        result["_tokens_used"] = tokens
        return result

    # -----------------------------------------------------------------------
    # Prompt builders
    # -----------------------------------------------------------------------

    def _prompt_personalize(self, payload: Dict[str, Any]) -> str:
        return f"""Personalize a story angle for the following prospect.

All available data:
{self._fmt(payload)}

Your task:
1. Write a HOOK (1-2 sentences) that creates pattern interruption — something they \
   haven't heard before that speaks directly to their specific situation.
2. Identify the EMOTIONAL TRIGGER most likely to move this specific person to respond.
3. Craft the STORY ANGLE — the specific narrative frame that bridges their pain \
   to Storibot's transformation promise.
4. Find 2-3 PROOF POINTS that are credible and relevant to their industry/role.
5. Mirror their LANGUAGE — list the exact vocabulary and phrases they would use.
6. Paint the BEFORE STATE in their own words (not generic AI-content jargon).
7. Make the AFTER STATE vision concrete and role-specific (CMO? Content Director?).
8. Identify the TENSION DRIVER — what makes acting *now* different from acting in 6 months?

If narrative_framework or research_data is missing, work with what you have but \
flag the gaps in missing_data_flags.

Submit using the submit_result tool."""

    def _prompt_adapt(self, payload: Dict[str, Any]) -> str:
        return f"""Adapt an existing narrative framework to a new industry vertical.

Existing framework and target context:
{self._fmt(payload)}

Translate the narrative_framework's protagonist, antagonist, and transformation arc \
into language and context that resonates in the target_industry. \
Preserve the emotional architecture; change the vocabulary, examples, and proof points.

Submit using the submit_result tool."""

    @staticmethod
    def _fmt(obj: Any) -> str:
        import json
        return json.dumps(obj, indent=2, default=str)
