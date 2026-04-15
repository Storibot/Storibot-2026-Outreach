"""
Session state layer for the Storibot multi-agent orchestrator.

Each session tracks what every agent has produced so far, giving downstream
agents full context without re-fetching data from upstream.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional


SESSION_TTL_HOURS = 24


@dataclass
class SessionState:
    session_id: str

    # Populated progressively as agents run
    company_context: Optional[Dict[str, Any]] = None        # raw input
    narrative_framework: Optional[Dict[str, Any]] = None    # story_architecture
    research_data: Optional[Dict[str, Any]] = None          # lead_research
    personalization_data: Optional[Dict[str, Any]] = None   # narrative_personalization
    outreach_content: Optional[Dict[str, Any]] = None       # outreach_generator
    qualification_data: Optional[Dict[str, Any]] = None     # qualification
    campaign_plan: Optional[Dict[str, Any]] = None          # campaign_orchestration

    # Conversation history shared across agents (list of {role, content} dicts)
    conversation_history: List[Dict[str, Any]] = field(default_factory=list)

    # Audit trail — one entry per agent execution
    agent_execution_log: List[Dict[str, Any]] = field(default_factory=list)

    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    expires_at: datetime = field(
        default_factory=lambda: datetime.utcnow() + timedelta(hours=SESSION_TTL_HOURS)
    )

    # ---------------------------------------------------------------------------
    # Mutation helpers
    # ---------------------------------------------------------------------------

    def set_agent_result(self, agent_id: str, data: Dict[str, Any]) -> None:
        """Write an agent's result into the canonical slot for that agent."""
        slot_map = {
            "story_architecture": "narrative_framework",
            "lead_research": "research_data",
            "narrative_personalization": "personalization_data",
            "outreach_generator": "outreach_content",
            "qualification": "qualification_data",
            "campaign_orchestration": "campaign_plan",
        }
        slot = slot_map.get(agent_id)
        if slot:
            setattr(self, slot, data)
        self.updated_at = datetime.utcnow()

    def log_execution(
        self,
        agent_id: str,
        action: str,
        tokens_used: int,
        duration_ms: float,
    ) -> None:
        self.agent_execution_log.append(
            {
                "agent_id": agent_id,
                "action": action,
                "tokens_used": tokens_used,
                "duration_ms": round(duration_ms, 1),
                "timestamp": datetime.utcnow().isoformat(),
            }
        )
        self.updated_at = datetime.utcnow()

    def append_turn(self, role: str, content: Any) -> None:
        self.conversation_history.append({"role": role, "content": content})
        self.updated_at = datetime.utcnow()

    # ---------------------------------------------------------------------------
    # Queries
    # ---------------------------------------------------------------------------

    def is_expired(self) -> bool:
        return datetime.utcnow() > self.expires_at

    def agents_run(self) -> List[str]:
        return [e["agent_id"] for e in self.agent_execution_log]

    def summary(self) -> Dict[str, Any]:
        return {
            "session_id": self.session_id,
            "agents_run": self.agents_run(),
            "has_narrative_framework": self.narrative_framework is not None,
            "has_research_data": self.research_data is not None,
            "has_personalization_data": self.personalization_data is not None,
            "has_outreach_content": self.outreach_content is not None,
            "has_qualification_data": self.qualification_data is not None,
            "has_campaign_plan": self.campaign_plan is not None,
            "execution_count": len(self.agent_execution_log),
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }


class SessionStore:
    """In-memory session store with TTL eviction.

    For production, swap _sessions for a Redis-backed implementation with the
    same interface — nothing else in the codebase needs to change.
    """

    def __init__(self) -> None:
        self._sessions: Dict[str, SessionState] = {}

    # ---------------------------------------------------------------------------
    # CRUD
    # ---------------------------------------------------------------------------

    def create(self, session_id: Optional[str] = None) -> SessionState:
        sid = session_id or str(uuid.uuid4())
        session = SessionState(session_id=sid)
        self._sessions[sid] = session
        return session

    def get(self, session_id: str) -> Optional[SessionState]:
        session = self._sessions.get(session_id)
        if session is None:
            return None
        if session.is_expired():
            del self._sessions[session_id]
            return None
        return session

    def get_or_create(self, session_id: Optional[str] = None) -> SessionState:
        if session_id:
            existing = self.get(session_id)
            if existing:
                return existing
        return self.create(session_id)

    def delete(self, session_id: str) -> None:
        self._sessions.pop(session_id, None)

    # ---------------------------------------------------------------------------
    # Maintenance
    # ---------------------------------------------------------------------------

    def evict_expired(self) -> int:
        expired = [sid for sid, s in self._sessions.items() if s.is_expired()]
        for sid in expired:
            del self._sessions[sid]
        return len(expired)

    @property
    def active_count(self) -> int:
        self.evict_expired()
        return len(self._sessions)
