"""
Storibot Orchestrator — JSON dispatch wired to the Managed Agents API.

Dispatch format (request):
    {
        "session_id": "optional-uuid",      # omit to start a new session
        "agent_id":   "story_architecture", # which agent to invoke
        "action":     "design_narrative",   # which action on that agent
        "payload":    { ... },              # action-specific data
        "metadata":   { ... }              # pass-through; trace IDs, priorities, etc.
    }

Dispatch format (response):
    {
        "session_id":           "uuid",
        "agent_id":             "story_architecture",
        "action":               "design_narrative",
        "result":               { ... },    # agent output
        "tokens_used":          1234,
        "duration_ms":          2340.5,
        "suggested_next_agent": "lead_research",
        "session_summary":      { ... },
        "metadata":             { ... }
    }

The registry is the routing table — adding a new agent requires only a
registry.register() call.  Nothing else in this file changes.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, Optional

import anthropic
from pydantic import BaseModel, Field

from registry import CapabilityRegistry, build_registry
from session import SessionStore, SessionState


# ---------------------------------------------------------------------------
# Pydantic models for the dispatch wire format
# ---------------------------------------------------------------------------


class DispatchRequest(BaseModel):
    session_id: Optional[str] = Field(
        default=None,
        description="Existing session ID; omit to start a new session",
    )
    agent_id: str = Field(description="Target agent identifier")
    action: str = Field(description="Action to invoke on the agent")
    payload: Dict[str, Any] = Field(
        default_factory=dict,
        description="Action-specific input data",
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Pass-through metadata (trace IDs, priorities, webhook URLs, etc.)",
    )


class DispatchResponse(BaseModel):
    session_id: str
    agent_id: str
    action: str
    result: Dict[str, Any]
    tokens_used: int
    duration_ms: float
    suggested_next_agent: Optional[str] = None
    session_summary: Dict[str, Any] = Field(default_factory=dict)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class DispatchError(BaseModel):
    error: str
    error_code: str
    session_id: Optional[str] = None
    agent_id: Optional[str] = None
    action: Optional[str] = None
    available_agents: Optional[list] = None


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


class Orchestrator:
    """Routes DispatchRequests to the correct agent and manages session state.

    The orchestrator itself is stateless — all state lives in the SessionStore.
    One Orchestrator instance per process is sufficient.
    """

    def __init__(
        self,
        client: anthropic.Anthropic,
        registry: CapabilityRegistry,
        session_store: SessionStore,
    ) -> None:
        self.client = client
        self.registry = registry
        self.session_store = session_store

    # ---------------------------------------------------------------------------
    # Primary entry point
    # ---------------------------------------------------------------------------

    def dispatch(self, request: DispatchRequest) -> DispatchResponse:
        """Execute a dispatch request end-to-end.

        1. Validate (agent_id, action) against the registry
        2. Resolve or create the session
        3. Instantiate the agent handler
        4. Execute and capture result + tokens
        5. Persist agent output to session state
        6. Return a fully populated DispatchResponse
        """
        start_ts = datetime.utcnow()

        # --- route validation -----------------------------------------------
        route = self.registry.route(request.agent_id, request.action)
        if route is None:
            raise ValueError(
                f"No route: agent_id={request.agent_id!r} action={request.action!r}. "
                f"Enabled agents: {self.registry.enabled_agent_ids}"
            )
        config, _action_spec = route

        # --- session resolution ----------------------------------------------
        session: SessionState = self.session_store.get_or_create(request.session_id)

        # --- handler execution -----------------------------------------------
        handler = config.handler_class(self.client)
        raw_result: Dict[str, Any] = handler.execute(
            request.action, request.payload, session
        )

        # Extract token count injected by the agent, then strip it from result
        tokens_used: int = raw_result.pop("_tokens_used", 0)

        # --- session bookkeeping ---------------------------------------------
        duration_ms = (datetime.utcnow() - start_ts).total_seconds() * 1000
        session.log_execution(
            agent_id=request.agent_id,
            action=request.action,
            tokens_used=tokens_used,
            duration_ms=duration_ms,
        )

        # --- suggest next step -----------------------------------------------
        suggested_next = self.registry.suggest_next(request.agent_id)

        return DispatchResponse(
            session_id=session.session_id,
            agent_id=request.agent_id,
            action=request.action,
            result=raw_result,
            tokens_used=tokens_used,
            duration_ms=round(duration_ms, 1),
            suggested_next_agent=suggested_next,
            session_summary=session.summary(),
            metadata=request.metadata,
        )

    # ---------------------------------------------------------------------------
    # Convenience: run the full pipeline for one company in a single call
    # ---------------------------------------------------------------------------

    def run_full_pipeline(
        self,
        company_payload: Dict[str, Any],
        session_id: Optional[str] = None,
        stop_after: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Run the complete 6-agent pipeline for a single company.

        Executes agents in default pipeline order:
            story_architecture → lead_research → narrative_personalization →
            outreach_generator → qualification → campaign_orchestration

        Args:
            company_payload: Core company data (company_name, industry, etc.)
            session_id:      Reuse an existing session or None for a new one
            stop_after:      Stop after this agent_id (e.g. "lead_research" for
                             the 'platform functional' milestone)

        Returns a dict keyed by agent_id with each agent's DispatchResponse dict.
        """
        pipeline = [
            ("story_architecture", "design_narrative"),
            ("lead_research", "research_company"),
            ("narrative_personalization", "personalize_narrative"),
            ("outreach_generator", "generate_sequence"),
            ("qualification", "qualify_lead"),
            ("campaign_orchestration", "design_campaign"),
        ]

        results: Dict[str, Any] = {}
        current_session_id = session_id

        for agent_id, action in pipeline:
            response = self.dispatch(
                DispatchRequest(
                    session_id=current_session_id,
                    agent_id=agent_id,
                    action=action,
                    payload=company_payload,
                )
            )
            results[agent_id] = response.model_dump()
            current_session_id = response.session_id  # carry session across agents

            if stop_after and agent_id == stop_after:
                break

        return {
            "session_id": current_session_id,
            "pipeline_results": results,
            "agents_run": list(results.keys()),
        }


# ---------------------------------------------------------------------------
# Module-level singletons (initialised in main.py via init_orchestrator())
# ---------------------------------------------------------------------------

_orchestrator: Optional[Orchestrator] = None


def init_orchestrator(api_key: str) -> Orchestrator:
    """Initialise the global orchestrator singleton.  Call once at startup."""
    global _orchestrator
    client = anthropic.Anthropic(api_key=api_key)
    registry = build_registry()
    session_store = SessionStore()
    _orchestrator = Orchestrator(client, registry, session_store)
    return _orchestrator


def get_orchestrator() -> Orchestrator:
    if _orchestrator is None:
        raise RuntimeError("Orchestrator not initialised. Call init_orchestrator() first.")
    return _orchestrator
