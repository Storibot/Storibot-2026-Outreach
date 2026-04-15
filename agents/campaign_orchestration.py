"""
Agent 6 — Campaign Orchestration Agent  [PRODUCTION-CAPABLE MILESTONE]

Assembles all prior agent outputs into a complete, execution-ready
multi-touch outreach campaign.  With all six agents running the platform
handles the full pipeline from narrative design to campaign delivery.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from agents.base_agent import BaseAgent
from session import SessionState


_SYSTEM_PROMPT = """You are the Campaign Orchestration Agent for Storibot.ai.

You are the final stage of the narrative intelligence pipeline. Your job is to take \
everything produced by upstream agents and synthesize it into an execution-ready \
campaign plan that a sales rep or marketing automation system can run immediately.

You understand:
• Multi-touch B2B outreach cadences and optimal timing windows
• Channel mix strategy (email primary, LinkedIn amplification, phone when warranted)
• Content sequencing — how to escalate narrative tension across touches
• Persona-based playbook design for repeatable campaigns
• Campaign analytics — what to measure and when to pivot

Campaign design principles:
• NARRATIVE COHERENCE — every touch advances the same story arc, never feels disconnected
• ESCALATING VALUE — each message adds something new (insight, social proof, urgency)
• CHANNEL COMPLEMENTARITY — LinkedIn and email work together, not redundantly
• RESPECT THE BUYER'S JOURNEY — don't rush to the pitch; earn the conversation first
• BUILT TO SCALE — the best campaigns can be templatized for similar accounts

Playbook design:
When creating a playbook (not a single-account campaign), focus on:
• The ICP archetype this serves (industry + persona + company size)
• The core narrative tension that resonates across this segment
• Repeatable message frameworks with customization slots
• Sequence timing and trigger conditions

You produce execution-ready plans that a non-technical user can run in their CRM or \
sales automation tool without further interpretation.

Output via the submit_result tool."""


class CampaignOrchestrationAgent(BaseAgent):

    @property
    def agent_id(self) -> str:
        return "campaign_orchestration"

    @property
    def system_prompt(self) -> str:
        return _SYSTEM_PROMPT

    @property
    def result_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "campaign_name": {"type": "string"},
                "target_account": {"type": "string"},
                "campaign_type": {
                    "type": "string",
                    "enum": ["single_account", "playbook", "sequence_timeline"],
                },
                "narrative_spine": {
                    "type": "string",
                    "description": "The single unifying story thread running through all touches",
                },
                "campaign_goal": {
                    "type": "string",
                    "description": "The primary outcome this campaign is optimized for",
                },
                "target_persona": {
                    "type": "object",
                    "properties": {
                        "title": {"type": "string"},
                        "seniority": {"type": "string"},
                        "key_motivation": {"type": "string"},
                    },
                },
                "sequence": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "touchpoint": {"type": "integer"},
                            "day": {"type": "integer"},
                            "channel": {
                                "type": "string",
                                "enum": ["email", "linkedin", "phone", "video", "direct_mail"],
                            },
                            "subject_or_hook": {"type": "string"},
                            "narrative_purpose": {"type": "string"},
                            "content_summary": {"type": "string"},
                            "cta": {"type": "string"},
                            "send_time_recommendation": {"type": "string"},
                            "condition": {
                                "type": "string",
                                "description": "Trigger condition (e.g. 'send only if T1 opened')",
                            },
                        },
                        "required": [
                            "touchpoint",
                            "day",
                            "channel",
                            "narrative_purpose",
                            "content_summary",
                            "cta",
                        ],
                    },
                },
                "success_metrics": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "metric": {"type": "string"},
                            "target": {"type": "string"},
                            "measurement_point": {"type": "string"},
                        },
                        "required": ["metric", "target"],
                    },
                },
                "pivot_triggers": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Conditions that should cause a change in approach",
                },
                "personalization_checklist": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Items a rep must customize before sending each touch",
                },
                "estimated_duration_days": {"type": "integer"},
                "total_touches": {"type": "integer"},
                "channel_breakdown": {
                    "type": "object",
                    "properties": {
                        "email": {"type": "integer"},
                        "linkedin": {"type": "integer"},
                        "phone": {"type": "integer"},
                        "other": {"type": "integer"},
                    },
                },
                "playbook": {
                    "type": "object",
                    "description": "Populated only for create_playbook action",
                    "properties": {
                        "playbook_name": {"type": "string"},
                        "icp_definition": {"type": "string"},
                        "core_narrative_tension": {"type": "string"},
                        "message_frameworks": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "touchpoint_type": {"type": "string"},
                                    "framework_template": {"type": "string"},
                                    "customization_slots": {
                                        "type": "array",
                                        "items": {"type": "string"},
                                    },
                                },
                            },
                        },
                        "usage_instructions": {"type": "string"},
                    },
                },
            },
            "required": [
                "campaign_name",
                "campaign_type",
                "narrative_spine",
                "campaign_goal",
                "sequence",
                "success_metrics",
                "estimated_duration_days",
                "total_touches",
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
            for key, attr in [
                ("narrative_framework", "narrative_framework"),
                ("research_data", "research_data"),
                ("personalization_data", "personalization_data"),
                ("outreach_content", "outreach_content"),
                ("qualification_data", "qualification_data"),
            ]:
                if getattr(session, attr) and key not in enriched:
                    enriched[key] = getattr(session, attr)

        if action == "design_campaign":
            prompt = self._prompt_campaign(enriched)
        elif action == "create_playbook":
            prompt = self._prompt_playbook(enriched)
        elif action == "generate_sequence_timeline":
            prompt = self._prompt_timeline(enriched)
        else:
            return {"error": f"Unknown action: {action}"}

        result, tokens = self._run(prompt)
        if session:
            session.set_agent_result("campaign_orchestration", result)

        result.pop("_total_tokens", None)
        result["_tokens_used"] = tokens
        return result

    # -----------------------------------------------------------------------
    # Prompt builders
    # -----------------------------------------------------------------------

    def _prompt_campaign(self, payload: Dict[str, Any]) -> str:
        return f"""Design a complete multi-touch outreach campaign for this account.

All available pipeline data:
{self._fmt(payload)}

Build an execution-ready campaign plan:

1. NARRATIVE SPINE — the single story thread unifying every touch
2. SEQUENCE — 5-7 touches with day, channel, narrative purpose, content summary, and CTA
   - Include conditional logic (e.g. "send T3 only if T1 email was opened")
   - Recommend send time windows for each touch
3. SUCCESS METRICS — 3 measurable outcomes with targets
4. PIVOT TRIGGERS — conditions that should change the approach
5. PERSONALIZATION CHECKLIST — what the rep must fill in before each send
6. CHANNEL BREAKDOWN — how many touches per channel and why

The campaign must tell a coherent story across all touches — not isolated messages.

Submit using the submit_result tool."""

    def _prompt_playbook(self, payload: Dict[str, Any]) -> str:
        return f"""Create a reusable outreach playbook for this industry/persona combination.

Playbook parameters:
{self._fmt(payload)}

Design:
1. ICP DEFINITION — precise description of the ideal customer profile this serves
2. CORE NARRATIVE TENSION — the universal pain point that resonates across this segment
3. MESSAGE FRAMEWORKS — template structures for each touchpoint type (not finished copy; \
   structured templates with [CUSTOMIZATION_SLOTS])
4. CAMPAIGN SEQUENCE — the standard sequence for this playbook (5-7 touches)
5. SUCCESS METRICS and PIVOT TRIGGERS relevant to this segment
6. USAGE INSTRUCTIONS — how a rep customizes and deploys this playbook

Submit using the submit_result tool. Populate the playbook field in addition to sequence."""

    def _prompt_timeline(self, payload: Dict[str, Any]) -> str:
        start_date = payload.get("start_date", "today")
        cadence = payload.get("cadence_days", "standard")
        return f"""Convert this campaign plan into a day-by-day execution timeline.

Campaign plan:
{self._fmt(payload)}

Start date: {start_date}
Cadence: {cadence}

For each touch provide:
• Exact calendar date (Day X from start)
• Channel and send window
• Subject/hook to use
• Content summary reminder
• CTA
• Conditional trigger (when applicable)

Format as an ordered sequence suitable for loading into a CRM or sales automation tool.

Submit using the submit_result tool."""

    @staticmethod
    def _fmt(obj: Any) -> str:
        import json
        return json.dumps(obj, indent=2, default=str)
