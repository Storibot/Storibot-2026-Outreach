"""
Storibot Lead Research Agent - Cloud Deployment Version
Orchestrator-Agents Architecture: specialized sub-agents coordinated by an orchestrator.
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict
from datetime import datetime
import os
import json
from anthropic import Anthropic

app = FastAPI(
    title="Storibot Lead Research Agent",
    description="AI-powered B2B lead research and qualification",
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
if not ANTHROPIC_API_KEY:
    raise ValueError("ANTHROPIC_API_KEY environment variable not set!")

client = Anthropic(api_key=ANTHROPIC_API_KEY)
MODEL = "claude-sonnet-4-6"


# ── Pydantic models ───────────────────────────────────────────────────────────

class CompanyInput(BaseModel):
    company_name: str
    industry: str
    employee_count: int
    website: str
    recent_news: Optional[str] = ""
    tech_stack: Optional[List[str]] = []

class ResearchResponse(BaseModel):
    company_name: str
    industry: str
    account_score: str
    confidence_score: float
    content_pain_points: List[str]
    trigger_events: List[str]
    recommended_angle: str
    narrative_opportunity: str
    ai_tools_detected: List[str]
    research_timestamp: str
    decision_makers: List[Dict]


# ── Sub-agent system prompts (stable prefix → cache_control applied) ──────────
# Note: Sonnet 4.6 requires ≥2048 tokens to cache; these prompts are intentionally
# structured to grow with context so caching activates as company data is appended.

_PAIN_POINTS_SYSTEM = """You are a B2B content strategist for Storibot.ai, a narrative intelligence platform \
that transforms AI-generated content into character-driven stories using Hollywood storytelling frameworks.

Your role: identify specific, actionable content pain points tied to the $47-50 billion content waste crisis.

Focus areas:
- High-volume AI content with authenticity and engagement problems
- Unused content assets and low-engagement patterns
- Misalignment between brand voice and AI-generated output
- Content scaling issues that erode trust and audience connection

Always respond with valid JSON only — no prose, no markdown fences."""

_DECISION_MAKER_SYSTEM = """You are a B2B sales intelligence specialist for Storibot.ai, a narrative intelligence \
platform that transforms AI-generated content into character-driven stories using Hollywood storytelling frameworks.

Your role: identify the decision makers who own content strategy and control budget for narrative solutions.

Priority targets: CMO, VP Marketing, Content Director, Head of Brand, VP Communications.
Key KPIs they care about: engagement rate, conversion, brand authenticity, content ROI, audience retention.

Always respond with valid JSON only — no prose, no markdown fences."""

_TRIGGER_EVENTS_SYSTEM = """You are a B2B sales trigger event specialist for Storibot.ai, a narrative intelligence \
platform that transforms AI-generated content into character-driven stories using Hollywood storytelling frameworks.

Your role: identify recent business events that create urgency and buying windows for content transformation.

High-signal triggers:
- Series A/B/C funding (expansion → content investment)
- Product or platform launches (require differentiated storytelling)
- New CMO or VP Marketing hire (new strategy mandate, fresh budget)
- Rapid hiring in content or marketing teams (scaling pain incoming)
- Rebranding initiatives (identity crisis = narrative opportunity)

Always respond with valid JSON only — no prose, no markdown fences."""

_NARRATIVE_ANALYST_SYSTEM = """You are a narrative intelligence specialist for Storibot.ai, a narrative intelligence \
platform that transforms AI-generated content into character-driven stories using Hollywood storytelling frameworks.

Your role: identify how Hollywood storytelling frameworks — character arc, conflict, transformation — can \
unlock authentic audience engagement for this company's content strategy.

Deliverables:
- Specific narrative opportunity tied to the company's audience and product
- Personalized outreach angle leading with their strongest pain point
- Account score (A/B/C): A = perfect fit with budget signals and active pain, B = good ICP match timing unclear, C = nurture

Always respond with valid JSON only — no prose, no markdown fences."""

_ORCHESTRATOR_SYSTEM = """You are the lead research orchestrator for Storibot.ai outreach intelligence.

You coordinate four specialist sub-agents to build comprehensive B2B sales intelligence on a target company:
1. analyze_content_pain_points  — content challenges and AI tool usage
2. identify_decision_makers     — who owns content strategy and budget
3. detect_trigger_events        — business events creating urgency
4. analyze_narrative_opportunity — narrative fit, outreach angle, account score

Workflow:
- Call ALL FOUR tools before producing output.
- After all tools complete, output a single JSON synthesis in exactly this structure:

{
  "content_pain_points": ["...", "..."],
  "decision_makers": [{"name": "Title", "priorities": ["..."]}],
  "trigger_events": ["...", "..."],
  "narrative_opportunity": "...",
  "recommended_angle": "...",
  "account_score": "A",
  "reasoning": "...",
  "ai_tools_detected": ["..."],
  "confidence_score": 0.85
}

Output only the JSON object — no prose before or after it."""


# ── Orchestrator tool definitions (stable → cached as part of prefix) ─────────

_ORCHESTRATOR_TOOLS = [
    {
        "name": "analyze_content_pain_points",
        "description": (
            "Runs the content pain points sub-agent. Returns specific content challenges "
            "the company faces, detected AI tools in their stack, and a confidence score."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "company_name": {"type": "string", "description": "Company to analyze"}
            },
            "required": ["company_name"]
        }
    },
    {
        "name": "identify_decision_makers",
        "description": (
            "Runs the decision maker sub-agent. Returns titles, priorities, and likely "
            "KPIs of the people who own content strategy and budget at this company."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "company_name": {"type": "string", "description": "Company to analyze"}
            },
            "required": ["company_name"]
        }
    },
    {
        "name": "detect_trigger_events",
        "description": (
            "Runs the trigger events sub-agent. Returns recent business events "
            "(funding, launches, leadership changes) that create a buying window."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "company_name": {"type": "string", "description": "Company to analyze"}
            },
            "required": ["company_name"]
        }
    },
    {
        "name": "analyze_narrative_opportunity",
        "description": (
            "Runs the narrative intelligence sub-agent. Returns the Hollywood storytelling "
            "opportunity, personalized outreach angle, and account score (A/B/C)."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "company_name": {"type": "string", "description": "Company to analyze"}
            },
            "required": ["company_name"]
        }
    }
]


# ── Helpers ───────────────────────────────────────────────────────────────────

def _company_context(company: CompanyInput) -> str:
    stack = ", ".join(company.tech_stack) if company.tech_stack else "Unknown"
    news = company.recent_news or "None available"
    return (
        f"Company: {company.company_name}\n"
        f"Industry: {company.industry}\n"
        f"Employee Count: {company.employee_count}\n"
        f"Website: {company.website}\n"
        f"Recent News: {news}\n"
        f"Technology Stack: {stack}"
    )

def _extract_json(text: str) -> str:
    if "```json" in text:
        start = text.find("```json") + 7
        return text[start:text.find("```", start)].strip()
    if "```" in text:
        start = text.find("```") + 3
        return text[start:text.find("```", start)].strip()
    start, end = text.find("{"), text.rfind("}") + 1
    if start != -1 and end > start:
        return text[start:end]
    return text.strip()

def _safe_parse(text: str) -> dict:
    try:
        return json.loads(_extract_json(text))
    except (json.JSONDecodeError, ValueError):
        return {}


# ── Sub-agent runners ─────────────────────────────────────────────────────────

def _run_pain_points_agent(company: CompanyInput) -> dict:
    resp = client.messages.create(
        model=MODEL,
        max_tokens=1024,
        system=[{
            "type": "text",
            "text": _PAIN_POINTS_SYSTEM,
            "cache_control": {"type": "ephemeral"}
        }],
        messages=[{
            "role": "user",
            "content": (
                f"{_company_context(company)}\n\n"
                "Identify 2-3 specific content pain points for this company.\n\n"
                "Return JSON:\n"
                '{"content_pain_points": ["...", "..."], "ai_tools_detected": ["..."], "confidence_score": 0.85}'
            )
        }]
    )
    return _safe_parse(resp.content[0].text)


def _run_decision_maker_agent(company: CompanyInput) -> dict:
    resp = client.messages.create(
        model=MODEL,
        max_tokens=1024,
        system=[{
            "type": "text",
            "text": _DECISION_MAKER_SYSTEM,
            "cache_control": {"type": "ephemeral"}
        }],
        messages=[{
            "role": "user",
            "content": (
                f"{_company_context(company)}\n\n"
                "Identify the decision makers who own content strategy at this company.\n\n"
                "Return JSON:\n"
                '{"decision_makers": [{"name": "Title", "priorities": ["priority1", "priority2"]}]}'
            )
        }]
    )
    return _safe_parse(resp.content[0].text)


def _run_trigger_events_agent(company: CompanyInput) -> dict:
    resp = client.messages.create(
        model=MODEL,
        max_tokens=1024,
        system=[{
            "type": "text",
            "text": _TRIGGER_EVENTS_SYSTEM,
            "cache_control": {"type": "ephemeral"}
        }],
        messages=[{
            "role": "user",
            "content": (
                f"{_company_context(company)}\n\n"
                "Identify 2-3 trigger events that create a buying window for Storibot.ai.\n\n"
                "Return JSON:\n"
                '{"trigger_events": ["event1", "event2"]}'
            )
        }]
    )
    return _safe_parse(resp.content[0].text)


def _run_narrative_analyst_agent(company: CompanyInput) -> dict:
    resp = client.messages.create(
        model=MODEL,
        max_tokens=1024,
        system=[{
            "type": "text",
            "text": _NARRATIVE_ANALYST_SYSTEM,
            "cache_control": {"type": "ephemeral"}
        }],
        messages=[{
            "role": "user",
            "content": (
                f"{_company_context(company)}\n\n"
                "Identify the narrative opportunity, outreach angle, and account score.\n\n"
                "Return JSON:\n"
                '{"narrative_opportunity": "...", "recommended_angle": "...", '
                '"account_score": "A", "reasoning": "..."}'
            )
        }]
    )
    return _safe_parse(resp.content[0].text)


def _dispatch_tool(tool_name: str, company: CompanyInput) -> str:
    dispatch = {
        "analyze_content_pain_points": _run_pain_points_agent,
        "identify_decision_makers": _run_decision_maker_agent,
        "detect_trigger_events": _run_trigger_events_agent,
        "analyze_narrative_opportunity": _run_narrative_analyst_agent,
    }
    runner = dispatch.get(tool_name)
    if runner is None:
        return json.dumps({"error": f"Unknown tool: {tool_name}"})
    try:
        return json.dumps(runner(company))
    except Exception as exc:
        return json.dumps({"error": str(exc)})


# ── Orchestrator ──────────────────────────────────────────────────────────────

def research_company(company_data: CompanyInput) -> ResearchResponse:
    messages = [{
        "role": "user",
        "content": [{
            "type": "text",
            "text": (
                f"Research this company for Storibot.ai outreach:\n\n"
                f"{_company_context(company_data)}\n\n"
                "Call all four analysis tools, then output the JSON synthesis."
            ),
            "cache_control": {"type": "ephemeral"}
        }]
    }]

    # Agentic loop: orchestrator calls sub-agents via tool use
    while True:
        response = client.messages.create(
            model=MODEL,
            max_tokens=4096,
            system=[{
                "type": "text",
                "text": _ORCHESTRATOR_SYSTEM,
                "cache_control": {"type": "ephemeral"}
            }],
            tools=_ORCHESTRATOR_TOOLS,
            messages=messages
        )

        tool_use_blocks = [b for b in response.content if b.type == "tool_use"]

        if response.stop_reason == "end_turn" or not tool_use_blocks:
            break

        messages.append({"role": "assistant", "content": response.content})

        tool_results = [
            {
                "type": "tool_result",
                "tool_use_id": block.id,
                "content": _dispatch_tool(block.name, company_data),
            }
            for block in tool_use_blocks
        ]
        messages.append({"role": "user", "content": tool_results})

    # Extract orchestrator's final synthesis
    final_text = next((b.text for b in response.content if b.type == "text"), "")
    synthesis = _safe_parse(final_text)

    # Fallback: aggregate directly from sub-agent tool results in message history
    aggregated: dict = {
        "content_pain_points": [],
        "decision_makers": [],
        "trigger_events": [],
        "ai_tools_detected": [],
        "account_score": "B",
        "confidence_score": 0.5,
        "narrative_opportunity": "",
        "recommended_angle": "",
    }
    for msg in messages:
        if msg["role"] == "user" and isinstance(msg.get("content"), list):
            for block in msg["content"]:
                if not isinstance(block, dict) or block.get("type") != "tool_result":
                    continue
                data = _safe_parse(block.get("content", "{}"))
                aggregated["content_pain_points"].extend(data.get("content_pain_points", []))
                aggregated["trigger_events"].extend(data.get("trigger_events", []))
                aggregated["decision_makers"].extend(data.get("decision_makers", []))
                aggregated["ai_tools_detected"].extend(data.get("ai_tools_detected", []))
                for key in ("account_score", "confidence_score", "narrative_opportunity", "recommended_angle"):
                    if data.get(key):
                        aggregated[key] = data[key]

    def _get(key, default=None):
        return synthesis.get(key) or aggregated.get(key) or default

    return ResearchResponse(
        company_name=company_data.company_name,
        industry=company_data.industry,
        account_score=_get("account_score", "B"),
        confidence_score=float(_get("confidence_score", 0.5)),
        content_pain_points=_get("content_pain_points", []),
        trigger_events=_get("trigger_events", []),
        recommended_angle=_get("recommended_angle", ""),
        narrative_opportunity=_get("narrative_opportunity", ""),
        ai_tools_detected=list(set(_get("ai_tools_detected", []))),
        research_timestamp=datetime.utcnow().isoformat(),
        decision_makers=_get("decision_makers", []),
    )


# ── API Endpoints ─────────────────────────────────────────────────────────────

@app.get("/")
def root():
    return {
        "service": "Storibot Lead Research Agent",
        "version": "2.0.0",
        "architecture": "orchestrator-agents",
        "status": "operational",
        "docs": "/docs"
    }

@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "api_key_configured": bool(ANTHROPIC_API_KEY)
    }

@app.post("/research/single", response_model=ResearchResponse)
def research_single_company(company: CompanyInput):
    """
    Research a single company using the orchestrator-agents pattern.

    Example:
    {
        "company_name": "HealthTech Solutions",
        "industry": "Healthcare",
        "employee_count": 250,
        "website": "healthtech.com",
        "recent_news": "Series A funding",
        "tech_stack": ["ChatGPT", "Jasper"]
    }
    """
    try:
        return research_company(company)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Research failed: {str(exc)}")

@app.post("/research/batch")
def research_batch_companies(companies: List[CompanyInput]):
    """
    Research multiple companies in batch.

    Example:
    [
        {"company_name": "Company A", "industry": "Healthcare", "employee_count": 200, "website": "a.com"},
        {"company_name": "Company B", "industry": "Education", "employee_count": 150, "website": "b.com"}
    ]
    """
    results = []
    for company in companies:
        try:
            results.append(research_company(company))
        except Exception as exc:
            print(f"Error researching {company.company_name}: {exc}")
    return {"total": len(companies), "successful": len(results), "results": results}

@app.post("/webhook/research")
def webhook_research(company: CompanyInput):
    """Webhook endpoint for Make.com, Zapier, n8n integration."""
    try:
        result = research_company(company)
        return {
            "success": True,
            "company": result.company_name,
            "score": result.account_score,
            "confidence": result.confidence_score,
            "pain_points": result.content_pain_points,
            "recommended_angle": result.recommended_angle,
            "narrative_opportunity": result.narrative_opportunity,
            "timestamp": result.research_timestamp
        }
    except Exception as exc:
        return {"success": False, "error": str(exc)}


# Run with: uvicorn main:app --host 0.0.0.0 --port 8000
if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
