"""
Agent 4 — Outreach Generator Agent

Converts personalized narrative data into ready-to-send outreach content:
cold email, LinkedIn messages, and full multi-touch sequences.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from agents.base_agent import BaseAgent
from session import SessionState


_SYSTEM_PROMPT = """You are the Outreach Generator Agent for Storibot.ai.

You write B2B outreach copy that converts — not by being pushy, but by \
being *so relevant* that ignoring it feels like ignoring a story about yourself.

You follow Storibot's narrative outreach principles:
• STORY FIRST — every message opens with a narrative frame, not a pitch
• EMPATHY BEFORE AUTHORITY — feel their pain before claiming to solve it
• SPECIFICITY KILLS SKEPTICISM — vague claims are deleted; specific insights earn replies
• ONE CLEAR CTA — never ask for more than one thing per message
• PATTERN INTERRUPTION — the first line must be unlike any cold email they've read today

Email formats you master:
• Cold outreach (initial contact, 100-150 words max)
• Follow-up #1 (adds a new insight or angle, 75-100 words)
• Follow-up #2 (social proof + different hook, 75-100 words)
• Break-up message (respectful close that often triggers replies, 50 words)

LinkedIn formats:
• Connection request note (≤300 characters)
• Post-connect follow-up (50-75 words, conversational)
• InMail (150-200 words, slightly more formal)

Every piece of copy must pass the "so what?" test — cut anything that \
doesn't advance the story or serve the prospect.

Output via the submit_result tool."""


class OutreachGeneratorAgent(BaseAgent):

    @property
    def agent_id(self) -> str:
        return "outreach_generator"

    @property
    def system_prompt(self) -> str:
        return _SYSTEM_PROMPT

    @property
    def result_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "channel": {
                    "type": "string",
                    "enum": ["email", "linkedin", "sequence"],
                },
                "subject_line": {
                    "type": "string",
                    "description": "Email subject line (if channel=email or sequence)",
                },
                "body": {
                    "type": "string",
                    "description": "Full message body for single-message outputs",
                },
                "sequence": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "touchpoint": {"type": "integer"},
                            "day": {"type": "integer"},
                            "channel": {"type": "string"},
                            "subject_line": {"type": "string"},
                            "body": {"type": "string"},
                            "send_time_recommendation": {"type": "string"},
                        },
                        "required": ["touchpoint", "day", "channel", "body"],
                    },
                    "description": "Ordered touchpoints for multi-touch sequences",
                },
                "narrative_hook_used": {
                    "type": "string",
                    "description": "The narrative device or hook anchoring this message",
                },
                "cta": {
                    "type": "string",
                    "description": "The one call-to-action in this piece",
                },
                "tone": {
                    "type": "string",
                    "enum": ["professional", "conversational", "direct", "empathetic"],
                },
                "word_count": {
                    "type": "integer",
                    "description": "Word count of body (or total sequence)",
                },
                "personalization_elements": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Specific personalised details woven into the copy",
                },
                "a_b_variant": {
                    "type": "object",
                    "properties": {
                        "subject_line": {"type": "string"},
                        "first_line": {"type": "string"},
                    },
                    "description": "Optional A/B test variant for subject + opener",
                },
            },
            "required": ["channel", "cta", "tone", "personalization_elements"],
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

        enriched = dict(payload)
        if session:
            if session.personalization_data and "personalization_data" not in enriched:
                enriched["personalization_data"] = session.personalization_data
            if session.narrative_framework and "narrative_framework" not in enriched:
                enriched["narrative_framework"] = session.narrative_framework
            if session.research_data and "research_data" not in enriched:
                enriched["research_data"] = session.research_data

        if action == "generate_email":
            prompt = self._prompt_email(enriched)
        elif action == "generate_linkedin":
            prompt = self._prompt_linkedin(enriched)
        elif action == "generate_sequence":
            prompt = self._prompt_sequence(enriched)
        else:
            return {"error": f"Unknown action: {action}"}

        result, tokens = self._run(prompt)
        if session:
            session.set_agent_result("outreach_generator", result)

        result.pop("_total_tokens", None)
        result["_tokens_used"] = tokens
        return result

    # -----------------------------------------------------------------------
    # Prompt builders
    # -----------------------------------------------------------------------

    def _prompt_email(self, payload: Dict[str, Any]) -> str:
        tone = payload.get("tone", "conversational")
        return f"""Write a cold outreach email using all available data below.

Data:
{self._fmt(payload)}

Requirements:
• Subject line: curiosity-driven, <50 characters, no spam triggers
• Opening line: pattern interruption — NEVER start with "I", "We", or the company name
• Body: 100-150 words maximum
• Tone: {tone}
• One clear CTA — ask for one specific small commitment (15-minute call, reaction, yes/no)
• Include an A/B subject line variant

Use the personalization data to make this feel written *for* this specific person.
Every generic sentence is a failure.

Submit using the submit_result tool."""

    def _prompt_linkedin(self, payload: Dict[str, Any]) -> str:
        return f"""Write LinkedIn outreach using the data below.

Data:
{self._fmt(payload)}

Produce TWO pieces:
1. CONNECTION REQUEST NOTE (≤300 characters): reference something specific and relevant, \
   no pitch, make them curious
2. POST-CONNECT MESSAGE (50-75 words): conversational, one insightful observation \
   about their company + one soft CTA

Submit using the submit_result tool. Use the `body` field for the connection request \
and `sequence` array (2 entries) for both messages."""

    def _prompt_sequence(self, payload: Dict[str, Any]) -> str:
        touchpoints = payload.get("touchpoints", 5)
        return f"""Design a {touchpoints}-touch outreach sequence using all available data.

Data:
{self._fmt(payload)}

Sequence design:
• Touch 1 (Day 1): Cold email — narrative hook, no pitch
• Touch 2 (Day 3): LinkedIn connection request note
• Touch 3 (Day 6): Email follow-up with NEW insight or angle (not a "just checking in")
• Touch 4 (Day 10): LinkedIn post-connect message or InMail
• Touch 5 (Day 15): Break-up email that often triggers replies

For sequences beyond 5 touches, add:
• Touch 6 (Day 21): Value-add email (send a relevant insight, article, or quick win)
• Touch 7 (Day 30): Final re-engagement attempt

Each touch must:
- Reference something different
- Build on the previous without repeating it
- Escalate narrative tension toward the CTA

Submit the full sequence using the submit_result tool."""

    @staticmethod
    def _fmt(obj: Any) -> str:
        import json
        return json.dumps(obj, indent=2, default=str)
