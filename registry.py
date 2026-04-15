"""
Capability registry — the routing table for the Storibot orchestrator.

Every agent registers itself here.  The orchestrator resolves
(agent_id, action) → (AgentConfig, ActionSpec) and hands execution
off to the right handler class.

Adding a new agent = one registry.register() call in this file.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Type


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class ActionSpec:
    """Describes a single action an agent can perform."""

    action: str
    description: str
    required_payload_fields: List[str] = field(default_factory=list)
    optional_payload_fields: List[str] = field(default_factory=list)


@dataclass
class AgentConfig:
    """Full configuration entry for one agent."""

    agent_id: str
    display_name: str
    description: str
    model: str
    actions: Dict[str, ActionSpec]
    handler_class: Type[Any]
    enabled: bool = True

    # Suggested next agent in the default pipeline
    default_next_agent: Optional[str] = None


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


class CapabilityRegistry:
    """
    Routing table used by the orchestrator.

    Usage:
        route = registry.route("story_architecture", "design_narrative")
        if route:
            config, action_spec = route
            handler = config.handler_class(client)
            result = handler.execute(action_spec.action, payload, session)
    """

    def __init__(self) -> None:
        self._registry: Dict[str, AgentConfig] = {}

    # ---------------------------------------------------------------------------
    # Registration
    # ---------------------------------------------------------------------------

    def register(self, config: AgentConfig) -> None:
        self._registry[config.agent_id] = config

    # ---------------------------------------------------------------------------
    # Lookup
    # ---------------------------------------------------------------------------

    def get_agent(self, agent_id: str) -> Optional[AgentConfig]:
        return self._registry.get(agent_id)

    def route(self, agent_id: str, action: str) -> Optional[tuple[AgentConfig, ActionSpec]]:
        """Return (AgentConfig, ActionSpec) for a valid (agent_id, action) pair,
        or None if the agent is unknown / disabled / action unsupported."""
        config = self._registry.get(agent_id)
        if config is None or not config.enabled:
            return None
        action_spec = config.actions.get(action)
        if action_spec is None:
            return None
        return config, action_spec

    def suggest_next(self, current_agent_id: str) -> Optional[str]:
        config = self._registry.get(current_agent_id)
        if config:
            return config.default_next_agent
        return None

    # ---------------------------------------------------------------------------
    # Introspection
    # ---------------------------------------------------------------------------

    def list_agents(self) -> List[Dict[str, Any]]:
        return [
            {
                "agent_id": c.agent_id,
                "display_name": c.display_name,
                "description": c.description,
                "enabled": c.enabled,
                "actions": list(c.actions.keys()),
                "default_next_agent": c.default_next_agent,
            }
            for c in self._registry.values()
        ]

    def list_actions(self, agent_id: str) -> List[Dict[str, Any]]:
        config = self._registry.get(agent_id)
        if not config:
            return []
        return [
            {
                "action": spec.action,
                "description": spec.description,
                "required_payload_fields": spec.required_payload_fields,
                "optional_payload_fields": spec.optional_payload_fields,
            }
            for spec in config.actions.values()
        ]

    @property
    def enabled_agent_ids(self) -> List[str]:
        return [aid for aid, c in self._registry.items() if c.enabled]


# ---------------------------------------------------------------------------
# Factory — build the singleton registry used at startup
# ---------------------------------------------------------------------------


def build_registry() -> CapabilityRegistry:
    """Import all agent classes and register them.  Import here (not at module
    top-level) to avoid circular imports between agents and the registry."""

    from agents.story_architecture import StoryArchitectureAgent
    from agents.lead_research import LeadResearchAgent
    from agents.narrative_personalization import NarrativePersonalizationAgent
    from agents.outreach_generator import OutreachGeneratorAgent
    from agents.qualification import QualificationAgent
    from agents.campaign_orchestration import CampaignOrchestrationAgent

    registry = CapabilityRegistry()

    # ------------------------------------------------------------------
    # Agent 1 — Story Architecture
    # ------------------------------------------------------------------
    registry.register(
        AgentConfig(
            agent_id="story_architecture",
            display_name="Story Architecture Agent",
            description=(
                "Designs Hollywood-framework narrative architectures for B2B outreach. "
                "Identifies protagonist, antagonist, stakes, and transformation arc."
            ),
            model="claude-sonnet-4-6",
            handler_class=StoryArchitectureAgent,
            default_next_agent="lead_research",
            actions={
                "design_narrative": ActionSpec(
                    action="design_narrative",
                    description="Build a full narrative framework for a target company.",
                    required_payload_fields=["company_name", "industry"],
                    optional_payload_fields=["employee_count", "website", "recent_news", "tech_stack"],
                ),
                "analyze_hero_journey": ActionSpec(
                    action="analyze_hero_journey",
                    description="Map the buyer's hero-journey: current state → transformation.",
                    required_payload_fields=["company_name", "industry"],
                    optional_payload_fields=["pain_points", "goals"],
                ),
                "create_character_profile": ActionSpec(
                    action="create_character_profile",
                    description="Build a narrative character profile for a specific contact.",
                    required_payload_fields=["contact_title", "company_name"],
                    optional_payload_fields=["contact_name", "known_priorities"],
                ),
            },
        )
    )

    # ------------------------------------------------------------------
    # Agent 2 — Lead Research  (platform functional milestone)
    # ------------------------------------------------------------------
    registry.register(
        AgentConfig(
            agent_id="lead_research",
            display_name="Lead Research Agent",
            description=(
                "Deep B2B company research: content pain points, trigger events, "
                "decision-maker mapping, and A/B/C account scoring."
            ),
            model="claude-sonnet-4-6",
            handler_class=LeadResearchAgent,
            default_next_agent="narrative_personalization",
            actions={
                "research_company": ActionSpec(
                    action="research_company",
                    description="Full research pass on a single company.",
                    required_payload_fields=["company_name", "industry"],
                    optional_payload_fields=["employee_count", "website", "recent_news", "tech_stack"],
                ),
                "score_account": ActionSpec(
                    action="score_account",
                    description="Return A/B/C score and reasoning only — fast path.",
                    required_payload_fields=["company_name", "industry"],
                    optional_payload_fields=["employee_count", "recent_news"],
                ),
                "batch_research": ActionSpec(
                    action="batch_research",
                    description="Research a list of companies in one call.",
                    required_payload_fields=["companies"],
                    optional_payload_fields=[],
                ),
            },
        )
    )

    # ------------------------------------------------------------------
    # Agent 3 — Narrative Personalization
    # ------------------------------------------------------------------
    registry.register(
        AgentConfig(
            agent_id="narrative_personalization",
            display_name="Narrative Personalization Agent",
            description=(
                "Fuses the narrative framework with research data to craft "
                "a prospect-specific story angle, hook, and emotional trigger."
            ),
            model="claude-sonnet-4-6",
            handler_class=NarrativePersonalizationAgent,
            default_next_agent="outreach_generator",
            actions={
                "personalize_narrative": ActionSpec(
                    action="personalize_narrative",
                    description="Produce a personalized story angle for one prospect.",
                    required_payload_fields=["company_name"],
                    optional_payload_fields=["contact_title", "narrative_framework", "research_data"],
                ),
                "adapt_story_angle": ActionSpec(
                    action="adapt_story_angle",
                    description="Adapt an existing narrative framework to a new industry vertical.",
                    required_payload_fields=["narrative_framework", "target_industry"],
                    optional_payload_fields=["target_company_size"],
                ),
            },
        )
    )

    # ------------------------------------------------------------------
    # Agent 4 — Outreach Generator
    # ------------------------------------------------------------------
    registry.register(
        AgentConfig(
            agent_id="outreach_generator",
            display_name="Outreach Generator Agent",
            description=(
                "Converts personalized narrative data into ready-to-send outreach: "
                "cold email, LinkedIn DM, and multi-touch sequences."
            ),
            model="claude-sonnet-4-6",
            handler_class=OutreachGeneratorAgent,
            default_next_agent="qualification",
            actions={
                "generate_email": ActionSpec(
                    action="generate_email",
                    description="Write a cold outreach email.",
                    required_payload_fields=["company_name", "contact_title"],
                    optional_payload_fields=["contact_name", "personalization_data", "tone"],
                ),
                "generate_linkedin": ActionSpec(
                    action="generate_linkedin",
                    description="Write a LinkedIn connection request + follow-up message.",
                    required_payload_fields=["company_name", "contact_title"],
                    optional_payload_fields=["contact_name", "personalization_data"],
                ),
                "generate_sequence": ActionSpec(
                    action="generate_sequence",
                    description="Write a full N-touch outreach sequence.",
                    required_payload_fields=["company_name", "contact_title"],
                    optional_payload_fields=["touchpoints", "personalization_data", "campaign_context"],
                ),
            },
        )
    )

    # ------------------------------------------------------------------
    # Agent 5 — Qualification
    # ------------------------------------------------------------------
    registry.register(
        AgentConfig(
            agent_id="qualification",
            display_name="Qualification Agent",
            description=(
                "Deep qualification scoring (MEDDIC + narrative readiness), "
                "buying-window detection, and list prioritization."
            ),
            model="claude-sonnet-4-6",
            handler_class=QualificationAgent,
            default_next_agent="campaign_orchestration",
            actions={
                "qualify_lead": ActionSpec(
                    action="qualify_lead",
                    description="Full MEDDIC + narrative-readiness qualification.",
                    required_payload_fields=["company_name", "industry"],
                    optional_payload_fields=["research_data", "employee_count", "recent_news"],
                ),
                "prioritize_list": ActionSpec(
                    action="prioritize_list",
                    description="Rank a list of pre-researched leads by qualification score.",
                    required_payload_fields=["leads"],
                    optional_payload_fields=["scoring_criteria"],
                ),
                "identify_timing_signals": ActionSpec(
                    action="identify_timing_signals",
                    description="Detect buying-window signals from recent company news.",
                    required_payload_fields=["company_name", "recent_news"],
                    optional_payload_fields=["industry"],
                ),
            },
        )
    )

    # ------------------------------------------------------------------
    # Agent 6 — Campaign Orchestration  (production-capable milestone)
    # ------------------------------------------------------------------
    registry.register(
        AgentConfig(
            agent_id="campaign_orchestration",
            display_name="Campaign Orchestration Agent",
            description=(
                "Assembles all prior agent outputs into a full multi-touch campaign: "
                "sequence timeline, channel mix, and reusable playbook."
            ),
            model="claude-sonnet-4-6",
            handler_class=CampaignOrchestrationAgent,
            default_next_agent=None,
            actions={
                "design_campaign": ActionSpec(
                    action="design_campaign",
                    description="Build an end-to-end outreach campaign for one account.",
                    required_payload_fields=["company_name"],
                    optional_payload_fields=[
                        "narrative_framework", "research_data",
                        "personalization_data", "outreach_content", "qualification_data",
                    ],
                ),
                "create_playbook": ActionSpec(
                    action="create_playbook",
                    description="Generate a reusable outreach playbook for an industry/persona pair.",
                    required_payload_fields=["industry", "target_persona"],
                    optional_payload_fields=["company_size_range", "primary_pain_point"],
                ),
                "generate_sequence_timeline": ActionSpec(
                    action="generate_sequence_timeline",
                    description="Convert a campaign plan into a day-by-day execution timeline.",
                    required_payload_fields=["campaign_plan"],
                    optional_payload_fields=["start_date", "cadence_days"],
                ),
            },
        )
    )

    return registry
