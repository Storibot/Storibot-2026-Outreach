"""
Agent 1 — Story Architecture Agent

Designs Hollywood-framework narrative architectures for B2B outreach.
Identifies protagonist (the buyer), antagonist (the content-waste crisis),
stakes, transformation arc, and guide positioning (Storibot).

This is the first agent in the pipeline.  All downstream agents build on
the narrative_framework it returns.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from agents.base_agent import BaseAgent
from session import SessionState


_SYSTEM_PROMPT = """You are the Story Architecture Agent for Storibot.ai — a narrative intelligence \
platform that transforms B2B outreach using Hollywood storytelling frameworks.

Your role is to design narrative architectures that turn dry business communication into \
compelling character-driven stories. You understand that every great story needs:

• PROTAGONIST — the buyer/prospect (their goals, fears, current struggle)
• ANTAGONIST — the force working against them (content waste crisis, AI commoditisation, \
  brand authenticity erosion, competitor noise)
• STAKES — what they lose if they don't act (revenue, brand equity, competitive position)
• GUIDE — Storibot.ai positioned as the Yoda/Mentor figure, not the hero
• TRANSFORMATION — the before/after journey from pain to narrative intelligence mastery
• CALL TO ACTION — the one clear next step that starts the journey

You are deeply versed in the Hero's Journey (Joseph Campbell), the StoryBrand framework \
(Donald Miller), three-act structure, and character-driven dramatic tension.

You produce structured JSON via the submit_result tool. Be specific, emotionally resonant, \
and grounded in the real business context provided.  Avoid generic platitudes — every \
output should feel crafted for THIS company, not a template.

Context: The $47–50 billion content waste crisis means most companies produce AI-generated \
content that audiences ignore. Storibot solves this by applying character-driven narrative \
intelligence to make content feel human and story-driven again."""


class StoryArchitectureAgent(BaseAgent):

    @property
    def agent_id(self) -> str:
        return "story_architecture"

    @property
    def system_prompt(self) -> str:
        return _SYSTEM_PROMPT

    @property
    def result_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "description": "Narrative framework for a target company",
            "properties": {
                "narrative_theme": {
                    "type": "string",
                    "description": "One-sentence thematic hook for this company's story",
                },
                "protagonist": {
                    "type": "object",
                    "properties": {
                        "role": {"type": "string"},
                        "core_desire": {"type": "string"},
                        "primary_fear": {"type": "string"},
                        "current_struggle": {"type": "string"},
                    },
                    "required": ["role", "core_desire", "primary_fear", "current_struggle"],
                },
                "antagonist": {
                    "type": "object",
                    "properties": {
                        "force": {"type": "string"},
                        "how_it_manifests": {"type": "string"},
                        "consequences_if_unaddressed": {"type": "string"},
                    },
                    "required": ["force", "how_it_manifests", "consequences_if_unaddressed"],
                },
                "stakes": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "2-3 specific things the protagonist stands to lose",
                },
                "guide_positioning": {
                    "type": "object",
                    "properties": {
                        "empathy_statement": {"type": "string"},
                        "authority_proof": {"type": "string"},
                        "plan_teaser": {"type": "string"},
                    },
                    "required": ["empathy_statement", "authority_proof", "plan_teaser"],
                },
                "transformation_arc": {
                    "type": "object",
                    "properties": {
                        "before_state": {"type": "string"},
                        "turning_point": {"type": "string"},
                        "after_state": {"type": "string"},
                    },
                    "required": ["before_state", "turning_point", "after_state"],
                },
                "hero_journey_stage": {
                    "type": "string",
                    "enum": [
                        "ordinary_world",
                        "call_to_adventure",
                        "refusal_of_call",
                        "meeting_the_mentor",
                        "crossing_threshold",
                        "road_of_trials",
                        "revelation",
                        "transformation",
                        "return",
                    ],
                    "description": "Where this prospect is in their buying/change journey",
                },
                "call_to_action": {
                    "type": "string",
                    "description": "The single clear next step Storibot should invite them to take",
                },
                "story_arc_type": {
                    "type": "string",
                    "enum": [
                        "overcoming_the_monster",
                        "rags_to_riches",
                        "the_quest",
                        "voyage_and_return",
                        "comedy",
                        "tragedy",
                        "rebirth",
                    ],
                    "description": "Most resonant classic arc for this company's situation",
                },
                "narrative_tension_score": {
                    "type": "number",
                    "minimum": 0.0,
                    "maximum": 1.0,
                    "description": "How urgent / emotionally charged this narrative is (0–1)",
                },
            },
            "required": [
                "narrative_theme",
                "protagonist",
                "antagonist",
                "stakes",
                "guide_positioning",
                "transformation_arc",
                "hero_journey_stage",
                "call_to_action",
                "story_arc_type",
                "narrative_tension_score",
            ],
        }

    # -----------------------------------------------------------------------
    # execute() — dispatches to per-action prompt builders
    # -----------------------------------------------------------------------

    def execute(
        self,
        action: str,
        payload: Dict[str, Any],
        session: Optional[SessionState],
    ) -> Dict[str, Any]:

        # Pull any existing session context to enrich prompts
        research_context = ""
        if session and session.research_data:
            research_context = (
                f"\n\nExisting research data for this company:\n"
                f"{self._fmt(session.research_data)}"
            )

        if action == "design_narrative":
            prompt = self._prompt_design_narrative(payload, research_context)
        elif action == "analyze_hero_journey":
            prompt = self._prompt_hero_journey(payload, research_context)
        elif action == "create_character_profile":
            prompt = self._prompt_character_profile(payload)
        else:
            return {"error": f"Unknown action: {action}"}

        result, tokens = self._run(prompt)

        # Write back to session
        if session and action == "design_narrative":
            session.set_agent_result("story_architecture", result)
            if payload.get("company_name"):
                session.company_context = session.company_context or {}
                session.company_context.update(
                    {k: payload[k] for k in payload if k != "_meta"}
                )

        result.pop("_total_tokens", None)
        result["_tokens_used"] = tokens
        return result

    # -----------------------------------------------------------------------
    # Prompt builders
    # -----------------------------------------------------------------------

    def _prompt_design_narrative(
        self, payload: Dict[str, Any], research_context: str
    ) -> str:
        return f"""Design a complete narrative framework for the following company.

Company details:
{self._fmt(payload)}
{research_context}

Using Hollywood storytelling frameworks (Hero's Journey, StoryBrand, three-act structure):
1. Identify the protagonist (the key buyer persona at this company)
2. Name the antagonist force they're fighting (content waste, AI commoditisation, etc.)
3. Define the stakes — what do they lose if the antagonist wins?
4. Position Storibot as the guide (Yoda, not Luke)
5. Map the transformation arc: before → turning point → after
6. Determine where they are in the Hero's Journey right now
7. Recommend the classic story arc type that best fits their situation
8. Write the one call-to-action that invites them into the story

Submit your complete narrative framework using the submit_result tool."""

    def _prompt_hero_journey(
        self, payload: Dict[str, Any], research_context: str
    ) -> str:
        return f"""Map the buyer's Hero's Journey for this company/contact.

Details:
{self._fmt(payload)}
{research_context}

Trace:
• ORDINARY WORLD — their current situation (status quo, pain not yet acknowledged)
• CALL TO ADVENTURE — the trigger event that will/has disrupted that world
• REFUSAL — their likely objections or inertia
• MEETING THE MENTOR — how Storibot enters as the guide
• CROSSING THE THRESHOLD — the commitment moment (first conversation / demo)
• ROAD OF TRIALS — implementation challenges they'll face
• REVELATION — the "aha" moment when narrative intelligence clicks
• TRANSFORMATION — what their content/outreach looks like after
• RETURN — how they become an internal champion for storytelling culture

Focus on the CURRENT stage and the one most important next step.

Submit your Hero's Journey analysis using the submit_result tool."""

    def _prompt_character_profile(self, payload: Dict[str, Any]) -> str:
        return f"""Create a narrative character profile for this contact.

Contact details:
{self._fmt(payload)}

Build:
• Their role in the company's internal story (decision maker, influencer, gatekeeper, champion)
• Their core professional desire (what "winning" looks like in their role)
• Their primary fear (what keeps them up at night)
• Their current struggle with content / AI / narrative
• The emotional trigger most likely to move them
• How to position Storibot as their ally, not a vendor

Submit your character profile using the submit_result tool."""

    @staticmethod
    def _fmt(obj: Any) -> str:
        import json
        return json.dumps(obj, indent=2, default=str)
