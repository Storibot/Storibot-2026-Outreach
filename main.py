"""
Storibot 2026 Outreach Platform
================================
Multi-agent narrative intelligence system for B2B outreach.

Agent pipeline (in order):
  1. Story Architecture      — Hollywood narrative framework
  2. Lead Research           — B2B intelligence + account scoring     [platform functional]
  3. Narrative Personalization — prospect-specific story angles
  4. Outreach Generator      — cold emails, LinkedIn, sequences
  5. Qualification           — MEDDIC + narrative-readiness scoring
  6. Campaign Orchestration  — end-to-end campaign plans + playbooks  [production capable]

All agents are accessible through the /orchestrate endpoint using the
JSON dispatch format.  The /pipeline endpoint runs the full chain for
one company in a single request.

Legacy /research/* endpoints are preserved for backwards compatibility.
"""

import os
from datetime import datetime
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from orchestrator import (
    DispatchError,
    DispatchRequest,
    DispatchResponse,
    init_orchestrator,
    get_orchestrator,
)

# ---------------------------------------------------------------------------
# App bootstrap
# ---------------------------------------------------------------------------

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
if not ANTHROPIC_API_KEY:
    raise ValueError("ANTHROPIC_API_KEY environment variable not set!")

orchestrator = init_orchestrator(ANTHROPIC_API_KEY)

app = FastAPI(
    title="Storibot 2026 Outreach Platform",
    description=(
        "Narrative intelligence multi-agent system for B2B outreach. "
        "Six agents: Story Architecture, Lead Research, Narrative Personalization, "
        "Outreach Generator, Qualification, Campaign Orchestration."
    ),
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Request / response models (legacy + new)
# ---------------------------------------------------------------------------


class CompanyInput(BaseModel):
    company_name: str
    industry: str
    employee_count: int = 0
    website: str = ""
    recent_news: Optional[str] = ""
    tech_stack: Optional[List[str]] = []


class PipelineRequest(BaseModel):
    company_name: str
    industry: str
    employee_count: int = 0
    website: str = ""
    recent_news: Optional[str] = ""
    tech_stack: Optional[List[str]] = []
    session_id: Optional[str] = None
    stop_after: Optional[str] = Field(
        default=None,
        description=(
            "Stop the pipeline after this agent. "
            "e.g. 'lead_research' for the functional milestone."
        ),
    )


# ---------------------------------------------------------------------------
# Core endpoints
# ---------------------------------------------------------------------------


@app.get("/")
def root() -> Dict[str, Any]:
    reg = get_orchestrator().registry
    return {
        "service": "Storibot 2026 Outreach Platform",
        "version": "2.0.0",
        "status": "operational",
        "agents": reg.enabled_agent_ids,
        "docs": "/docs",
        "session_count": get_orchestrator().session_store.active_count,
    }


@app.get("/health")
def health_check() -> Dict[str, Any]:
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "api_key_configured": bool(ANTHROPIC_API_KEY),
        "active_sessions": get_orchestrator().session_store.active_count,
    }


@app.get("/agents")
def list_agents() -> Dict[str, Any]:
    """List all registered agents and their available actions."""
    reg = get_orchestrator().registry
    return {"agents": reg.list_agents()}


@app.get("/agents/{agent_id}/actions")
def list_actions(agent_id: str) -> Dict[str, Any]:
    """List all actions for a specific agent."""
    reg = get_orchestrator().registry
    actions = reg.list_actions(agent_id)
    if not actions:
        raise HTTPException(status_code=404, detail=f"Agent '{agent_id}' not found or disabled")
    return {"agent_id": agent_id, "actions": actions}


@app.get("/sessions/{session_id}")
def get_session(session_id: str) -> Dict[str, Any]:
    """Retrieve the current state summary for a session."""
    session = get_orchestrator().session_store.get(session_id)
    if not session:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found or expired")
    return session.summary()


# ---------------------------------------------------------------------------
# Orchestrate — primary dispatch endpoint
# ---------------------------------------------------------------------------


@app.post("/orchestrate", response_model=DispatchResponse)
def orchestrate(request: DispatchRequest) -> DispatchResponse:
    """
    Primary dispatch endpoint.  Routes any (agent_id, action) to the
    appropriate agent and returns structured results.

    **Dispatch format:**
    ```json
    {
      "session_id": "optional-uuid",
      "agent_id":   "story_architecture",
      "action":     "design_narrative",
      "payload": {
        "company_name": "Acme Corp",
        "industry":     "SaaS"
      },
      "metadata": {}
    }
    ```

    **Available agent_id values:**
    - `story_architecture` — narrative framework design
    - `lead_research` — B2B intelligence + account scoring
    - `narrative_personalization` — prospect-specific story angles
    - `outreach_generator` — cold emails, LinkedIn, sequences
    - `qualification` — MEDDIC + narrative-readiness scoring
    - `campaign_orchestration` — end-to-end campaign plans

    The `suggested_next_agent` field in the response tells you which agent
    to call next in the recommended pipeline order.
    """
    try:
        return get_orchestrator().dispatch(request)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Orchestration failed: {str(exc)}")


# ---------------------------------------------------------------------------
# Pipeline — convenience endpoint for full-chain execution
# ---------------------------------------------------------------------------


@app.post("/pipeline")
def run_pipeline(request: PipelineRequest) -> Dict[str, Any]:
    """
    Run the full 6-agent narrative intelligence pipeline for one company.

    Set `stop_after` to run a partial pipeline:
    - `"lead_research"` — agents 1 + 2 only (platform functional milestone)
    - `"outreach_generator"` — agents 1–4 (narrative + content)
    - omit — all 6 agents (production-capable)

    Sessions are persisted across agents so each agent inherits prior results.
    Returns the complete output of every agent that ran.
    """
    try:
        payload = request.model_dump(exclude={"session_id", "stop_after"})
        return get_orchestrator().run_full_pipeline(
            company_payload=payload,
            session_id=request.session_id,
            stop_after=request.stop_after,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Pipeline failed: {str(exc)}")


# ---------------------------------------------------------------------------
# Legacy research endpoints (backwards compatibility)
# ---------------------------------------------------------------------------


@app.post("/research/single")
def research_single_company(company: CompanyInput) -> Dict[str, Any]:
    """
    Legacy endpoint.  Runs Lead Research Agent only.
    Use /orchestrate with agent_id='lead_research' for full control.
    """
    try:
        response = get_orchestrator().dispatch(
            DispatchRequest(
                agent_id="lead_research",
                action="research_company",
                payload=company.model_dump(),
            )
        )
        return {
            "company_name": company.company_name,
            "industry": company.industry,
            "session_id": response.session_id,
            **response.result,
            "research_timestamp": datetime.utcnow().isoformat(),
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Research failed: {str(exc)}")


@app.post("/research/batch")
def research_batch_companies(companies: List[CompanyInput]) -> Dict[str, Any]:
    """Legacy batch endpoint.  Runs Lead Research Agent on each company."""
    try:
        response = get_orchestrator().dispatch(
            DispatchRequest(
                agent_id="lead_research",
                action="batch_research",
                payload={"companies": [c.model_dump() for c in companies]},
            )
        )
        return {
            "total": len(companies),
            "session_id": response.session_id,
            **response.result,
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Batch research failed: {str(exc)}")


@app.post("/webhook/research")
def webhook_research(company: CompanyInput) -> Dict[str, Any]:
    """Webhook endpoint for Make.com, Zapier, n8n — simplified response."""
    try:
        response = get_orchestrator().dispatch(
            DispatchRequest(
                agent_id="lead_research",
                action="research_company",
                payload=company.model_dump(),
            )
        )
        r = response.result
        return {
            "success": True,
            "session_id": response.session_id,
            "company": company.company_name,
            "score": r.get("account_score"),
            "confidence": r.get("confidence_score"),
            "pain_points": r.get("content_pain_points", []),
            "narrative_opportunity": r.get("narrative_opportunity"),
            "recommended_angle": r.get("recommended_outreach_angle"),
            "suggested_next_agent": response.suggested_next_agent,
            "timestamp": datetime.utcnow().isoformat(),
        }
    except Exception as exc:
        return {"success": False, "error": str(exc)}


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
