"""
Agent 5 — Qualification Agent

Deep qualification scoring that goes beyond the simple A/B/C rating in Lead Research.
Uses MEDDIC + Narrative Readiness to evaluate buying readiness, detect timing signals,
and prioritize lists of leads.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from agents.base_agent import BaseAgent
from session import SessionState


_SYSTEM_PROMPT = """You are the Qualification Agent for Storibot.ai.

You perform rigorous B2B lead qualification using a hybrid framework:

MEDDIC FRAMEWORK:
  M — Metrics: What measurable outcome does the buyer need? (engagement lift, content ROI)
  E — Economic Buyer: Who controls the budget? Do we have access to them?
  D — Decision Criteria: What will they use to evaluate solutions?
  D — Decision Process: What steps must happen before they can buy?
  I — Identify Pain: How severe and acknowledged is the content/narrative pain?
  C — Champion: Is there an internal advocate who benefits from this change?

NARRATIVE READINESS SCORE (proprietary):
  • Does their culture value storytelling and authentic brand voice?
  • Have they publicly discussed content differentiation as a priority?
  • Are they already investing in premium content (case studies, video, editorial)?
  • Do their job postings signal content quality investment?
  • Is a senior content leader (VP/CMO level) actively shaping strategy?

TIMING SIGNALS (buying window detection):
  • Funding events (Series A/B/C = content investment mandate)
  • New CMO/VP Marketing (new leader = new strategy = new vendor evaluations)
  • Product launches (need to tell that story well)
  • Competitive pressure (rivals are outpublishing them)
  • Board pressure on content ROI or brand metrics
  • Recent hiring of content leadership

DISQUALIFICATION FLAGS:
  • Too small (<20 employees, no content team budget)
  • No digital content presence at all
  • Purely transactional B2C business model
  • No budget signals (bootstrapped, no funding, cost-cutting news)
  • Leadership completely resistant to creative/narrative approaches

Be honest about disqualifiers — a well-scoped pipeline beats a bloated one.
Output via the submit_result tool."""


class QualificationAgent(BaseAgent):

    @property
    def agent_id(self) -> str:
        return "qualification"

    @property
    def system_prompt(self) -> str:
        return _SYSTEM_PROMPT

    @property
    def result_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "company_name": {"type": "string"},
                "overall_qualification_score": {
                    "type": "string",
                    "enum": ["highly_qualified", "qualified", "marginal", "disqualified"],
                },
                "meddic": {
                    "type": "object",
                    "properties": {
                        "metrics": {"type": "string"},
                        "economic_buyer_access": {
                            "type": "string",
                            "enum": ["confirmed", "likely", "unclear", "unlikely"],
                        },
                        "decision_criteria": {"type": "array", "items": {"type": "string"}},
                        "decision_process": {"type": "string"},
                        "identified_pain_severity": {
                            "type": "string",
                            "enum": ["critical", "significant", "moderate", "latent"],
                        },
                        "champion_identified": {"type": "boolean"},
                        "champion_details": {"type": "string"},
                    },
                    "required": [
                        "metrics",
                        "economic_buyer_access",
                        "identified_pain_severity",
                        "champion_identified",
                    ],
                },
                "narrative_readiness_score": {
                    "type": "number",
                    "minimum": 0.0,
                    "maximum": 1.0,
                },
                "narrative_readiness_signals": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Specific evidence of narrative/storytelling culture",
                },
                "timing_signals": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "signal": {"type": "string"},
                            "urgency": {
                                "type": "string",
                                "enum": ["high", "medium", "low"],
                            },
                            "action_recommended": {"type": "string"},
                        },
                        "required": ["signal", "urgency"],
                    },
                },
                "disqualification_flags": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Any flags that reduce fit or buying likelihood",
                },
                "recommended_priority": {
                    "type": "string",
                    "enum": ["immediate", "this_week", "this_month", "nurture", "drop"],
                },
                "priority_reasoning": {"type": "string"},
                "estimated_deal_size": {
                    "type": "string",
                    "enum": ["enterprise", "mid_market", "smb", "unknown"],
                },
                "buying_window_estimate": {
                    "type": "string",
                    "description": "Estimated timeframe for a buying decision",
                },
                "next_best_action": {
                    "type": "string",
                    "description": "The single most important action to advance this account",
                },
            },
            "required": [
                "company_name",
                "overall_qualification_score",
                "meddic",
                "narrative_readiness_score",
                "timing_signals",
                "recommended_priority",
                "priority_reasoning",
                "next_best_action",
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

        enriched = dict(payload)
        if session:
            if session.research_data and "research_data" not in enriched:
                enriched["research_data"] = session.research_data
            if session.narrative_framework and "narrative_framework" not in enriched:
                enriched["narrative_framework"] = session.narrative_framework

        if action == "qualify_lead":
            prompt = self._prompt_qualify(enriched)
            result, tokens = self._run(prompt)
            if session:
                session.set_agent_result("qualification", result)

        elif action == "prioritize_list":
            result, tokens = self._handle_prioritize(payload)

        elif action == "identify_timing_signals":
            prompt = self._prompt_timing(enriched)
            result, tokens = self._run(prompt)

        else:
            return {"error": f"Unknown action: {action}"}

        result.pop("_total_tokens", None)
        result["_tokens_used"] = tokens
        return result

    # -----------------------------------------------------------------------
    # Prompt builders
    # -----------------------------------------------------------------------

    def _prompt_qualify(self, payload: Dict[str, Any]) -> str:
        return f"""Perform a full MEDDIC + Narrative Readiness qualification on this lead.

All available data:
{self._fmt(payload)}

Evaluate:
1. MEDDIC — be specific about each dimension, especially pain severity and champion status
2. NARRATIVE READINESS — look for cultural signals that this company values storytelling
3. TIMING SIGNALS — identify any current events that open or close the buying window
4. DISQUALIFICATION FLAGS — be honest; flag anything that reduces fit
5. OVERALL SCORE — highly_qualified / qualified / marginal / disqualified
6. PRIORITY — when should we act? immediate / this_week / this_month / nurture / drop
7. NEXT BEST ACTION — one concrete step to advance or test this account

Submit using the submit_result tool."""

    def _prompt_timing(self, payload: Dict[str, Any]) -> str:
        return f"""Identify buying-window timing signals from this company's recent data.

Data:
{self._fmt(payload)}

Look for:
• Funding announcements (round type, size, mandate)
• Leadership changes (especially CMO, VP Marketing, Chief Content Officer)
• Product launches or major announcements
• Job postings that signal content investment
• Competitive moves or market pressure events
• Earnings calls or public statements about content strategy

Rate each signal's urgency and recommend the action it warrants.

Submit using the submit_result tool. Use the timing_signals array as the primary output; \
fill other required fields with best estimates."""

    def _handle_prioritize(
        self, payload: Dict[str, Any]
    ) -> tuple[Dict[str, Any], int]:
        leads: List[Dict[str, Any]] = payload.get("leads", [])
        criteria = payload.get("scoring_criteria", "default MEDDIC + narrative readiness")

        prompt = f"""Prioritize this list of {len(leads)} leads by qualification score.

Scoring criteria: {criteria}

Leads:
{self._fmt(leads)}

For each lead provide:
• overall_qualification_score
• recommended_priority
• priority_reasoning (1 sentence)
• next_best_action

Return as a ranked list (highest priority first).

Submit using the submit_result tool with company_name set to "BATCH" and the \
prioritized list in timing_signals (as signal text per lead) plus your reasoning \
in priority_reasoning."""

        result, tokens = self._run(prompt)
        return result, tokens

    @staticmethod
    def _fmt(obj: Any) -> str:
        import json
        return json.dumps(obj, indent=2, default=str)
