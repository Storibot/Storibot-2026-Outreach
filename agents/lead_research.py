"""
Agent 2 — Lead Research Agent  [PLATFORM FUNCTIONAL MILESTONE]

Deep B2B company research: content pain points, trigger events,
decision-maker mapping, and A/B/C account scoring.

With Agents 1 and 2 live the platform can run a full
Story Architecture → Research pipeline end-to-end.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from agents.base_agent import BaseAgent
from session import SessionState


_SYSTEM_PROMPT = """You are the Lead Research Agent for Storibot.ai — a narrative intelligence \
platform that transforms AI-generated content into character-driven stories using Hollywood \
storytelling frameworks.

Your mission: Perform deep B2B intelligence research on companies to identify how they struggle \
with the $47–50 billion content waste crisis, and evaluate their fit for Storibot's solution.

You have expertise in:
• B2B SaaS sales intelligence and ICP analysis
• Content marketing and AI-content adoption patterns
• Buying signals: funding rounds, product launches, leadership changes, hiring surges
• MEDDIC/BANT qualification frameworks
• Identifying narrative-readiness — how open is this company to storytelling as a strategy?

Account scoring:
  A — Perfect fit: strong budget signals, active content pain, narrative-ready culture, buying window open
  B — Good fit: ICP match, pain exists, narrative potential clear, timing uncertain
  C — Acceptable fit: partial ICP match, content pain latent, nurture play

Always produce specific, actionable intelligence.  Generic observations are worthless — \
every insight must be tied to THIS company's specific situation."""


class LeadResearchAgent(BaseAgent):

    @property
    def agent_id(self) -> str:
        return "lead_research"

    @property
    def system_prompt(self) -> str:
        return _SYSTEM_PROMPT

    @property
    def result_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "company_name": {"type": "string"},
                "industry": {"type": "string"},
                "account_score": {
                    "type": "string",
                    "enum": ["A", "B", "C"],
                },
                "score_reasoning": {"type": "string"},
                "confidence_score": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                "content_pain_points": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "2-3 specific content/narrative challenges this company faces",
                },
                "narrative_readiness": {
                    "type": "string",
                    "enum": ["high", "medium", "low"],
                    "description": "How open is this company to storytelling as strategy?",
                },
                "decision_makers": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "title": {"type": "string"},
                            "likely_name_or_type": {"type": "string"},
                            "kpis": {"type": "array", "items": {"type": "string"}},
                            "key_pain": {"type": "string"},
                        },
                        "required": ["title", "kpis", "key_pain"],
                    },
                },
                "trigger_events": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Recent events that create an opening for outreach",
                },
                "ai_tools_detected": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "AI/content tools the company likely uses",
                },
                "narrative_opportunity": {
                    "type": "string",
                    "description": "Specific Hollywood storytelling application for this company",
                },
                "recommended_outreach_angle": {
                    "type": "string",
                    "description": "Personalized hook to lead with in the first touch",
                },
                "buying_window": {
                    "type": "string",
                    "enum": ["open", "likely_open", "unclear", "likely_closed"],
                },
                "competitive_intelligence": {
                    "type": "string",
                    "description": "What competing narrative/content approaches they may already use",
                },
            },
            "required": [
                "company_name",
                "industry",
                "account_score",
                "score_reasoning",
                "confidence_score",
                "content_pain_points",
                "narrative_readiness",
                "decision_makers",
                "trigger_events",
                "narrative_opportunity",
                "recommended_outreach_angle",
                "buying_window",
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

        # Enrich payload with narrative framework if available in session
        narrative_context = ""
        if session and session.narrative_framework:
            narrative_context = (
                "\n\nNarrative framework already designed for this company:\n"
                + self._fmt(session.narrative_framework)
            )

        if action == "research_company":
            prompt = self._prompt_research(payload, narrative_context)
            result, tokens = self._run(prompt)
            if session:
                session.set_agent_result("lead_research", result)
                if payload.get("company_name"):
                    session.company_context = session.company_context or {}
                    session.company_context.update(
                        {k: v for k, v in payload.items()}
                    )

        elif action == "score_account":
            prompt = self._prompt_score(payload)
            result, tokens = self._run(prompt)

        elif action == "batch_research":
            result, tokens = self._handle_batch(payload, session)

        else:
            return {"error": f"Unknown action: {action}"}

        result.pop("_total_tokens", None)
        result["_tokens_used"] = tokens
        return result

    # -----------------------------------------------------------------------
    # Prompt builders
    # -----------------------------------------------------------------------

    def _prompt_research(
        self, payload: Dict[str, Any], narrative_context: str
    ) -> str:
        return f"""Perform a full research pass on this company.

Company data:
{self._fmt(payload)}
{narrative_context}

Provide:
1. Content pain points (2-3 specific challenges, not generic)
2. Decision makers (titles, KPIs, key pains)
3. Trigger events visible in their recent news / growth patterns
4. AI / content tools they likely use
5. Narrative intelligence opportunity — specific Hollywood storytelling use case
6. Recommended first outreach angle
7. Account score (A/B/C) with clear reasoning
8. Narrative readiness assessment
9. Buying window estimate

Be specific to THIS company.  Avoid generic B2B filler.

Submit your research using the submit_result tool."""

    def _prompt_score(self, payload: Dict[str, Any]) -> str:
        return f"""Fast-path account scoring for:
{self._fmt(payload)}

Return account_score (A/B/C), score_reasoning, confidence_score, buying_window, \
and the single best recommended_outreach_angle.

Fill remaining fields with concise placeholder values.

Submit using the submit_result tool."""

    def _handle_batch(
        self,
        payload: Dict[str, Any],
        session: Optional[SessionState],
    ) -> tuple[Dict[str, Any], int]:
        companies: List[Dict[str, Any]] = payload.get("companies", [])
        results = []
        total_tokens = 0

        for company in companies:
            prompt = self._prompt_research(company, "")
            result, tokens = self._run(prompt)
            result.pop("_total_tokens", None)
            results.append(result)
            total_tokens += tokens

        return {
            "batch_results": results,
            "total_companies": len(companies),
            "successful": len([r for r in results if "error" not in r]),
        }, total_tokens

    @staticmethod
    def _fmt(obj: Any) -> str:
        import json
        return json.dumps(obj, indent=2, default=str)
