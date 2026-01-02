"""
Storibot Lead Research Agent - Cloud Deployment Version
Simplified for Replit, PythonAnywhere, Heroku (No Docker Required)
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional, Dict
from datetime import datetime
import os
import json
from anthropic import Anthropic

# Initialize FastAPI
app = FastAPI(
    title="Storibot Lead Research Agent",
    description="AI-powered B2B lead research and qualification",
    version="1.0.0"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Claude client
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
if not ANTHROPIC_API_KEY:
    raise ValueError("ANTHROPIC_API_KEY environment variable not set!")

client = Anthropic(api_key=ANTHROPIC_API_KEY)

# Pydantic models
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

# Research prompt from playbook
RESEARCH_PROMPT_TEMPLATE = """You are a B2B sales research specialist for Storibot.ai, a narrative intelligence platform that transforms AI-generated content into character-driven stories using Hollywood storytelling frameworks.

Your mission: Analyze this company and identify how they struggle with the $47-50 billion content waste crisis.

Company Information:
- Name: {company_name}
- Industry: {industry}
- Employee Count: {employee_count}
- Website: {website}
- Recent News: {recent_news}
- Technology Stack: {tech_stack}

Analyze and provide intelligence in the following areas:

1. CONTENT PAIN POINTS (2-3 specific challenges):
   - Look for signs they're creating high-volume AI content
   - Identify authenticity/engagement issues
   - Detect content waste patterns (unused assets, low engagement)

2. DECISION MAKERS & PRIORITIES:
   - Who owns content strategy? (CMO, VP Marketing, Content Director)
   - What are their likely KPIs? (engagement, conversion, brand authenticity)
   - What keeps them up at night?

3. TRIGGER EVENTS:
   - Recent funding rounds (expansion = content needs)
   - Product launches (requires storytelling)
   - Leadership changes (new CMO = strategy shift)
   - Hiring patterns (content team growth)

4. NARRATIVE INTELLIGENCE OPPORTUNITY:
   - How could Hollywood storytelling transform their content?
   - What character-driven narratives would resonate with their audience?
   - Specific use cases for their industry

5. RECOMMENDED OUTREACH ANGLE:
   - Personalized hook based on their situation
   - Which pain point to lead with
   - Credibility element from our background

6. ACCOUNT SCORE (A/B/C):
   - A: Perfect fit (budget signals, active pain, buying window)
   - B: Good fit (ICP match, pain exists, timing unclear)
   - C: Acceptable fit (partial match, nurture opportunity)

Provide your analysis in JSON format with this exact structure:
{{
    "content_pain_points": ["point1", "point2", "point3"],
    "decision_makers": [
        {{"name": "Title", "priorities": ["priority1", "priority2"]}}
    ],
    "trigger_events": ["event1", "event2"],
    "narrative_opportunity": "Specific Hollywood storytelling application",
    "recommended_angle": "Personalized outreach approach",
    "account_score": "A/B/C",
    "reasoning": "Why this score",
    "ai_tools_detected": ["tool1", "tool2"],
    "confidence_score": 0.85
}}

Be specific, actionable, and focused on narrative intelligence differentiation."""

def research_company(company_data: CompanyInput) -> ResearchResponse:
    """Research a single company using Claude AI"""
    
    # Format prompt
    prompt = RESEARCH_PROMPT_TEMPLATE.format(
        company_name=company_data.company_name,
        industry=company_data.industry,
        employee_count=company_data.employee_count,
        website=company_data.website,
        recent_news=company_data.recent_news or "No recent news available",
        tech_stack=", ".join(company_data.tech_stack) if company_data.tech_stack else "Unknown"
    )
    
    # Call Claude API
    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=2000,
        temperature=0.7,
        messages=[{"role": "user", "content": prompt}]
    )
    
    # Parse response
    analysis_text = response.content[0].text
    
    # Extract JSON
    try:
        if "```json" in analysis_text:
            json_start = analysis_text.find("```json") + 7
            json_end = analysis_text.find("```", json_start)
            analysis_text = analysis_text[json_start:json_end].strip()
        elif "```" in analysis_text:
            json_start = analysis_text.find("```") + 3
            json_end = analysis_text.find("```", json_start)
            analysis_text = analysis_text[json_start:json_end].strip()
        
        analysis = json.loads(analysis_text)
    except json.JSONDecodeError as e:
        # Fallback if parsing fails
        analysis = {
            "content_pain_points": ["Analysis in progress - manual review recommended"],
            "decision_makers": [],
            "trigger_events": [],
            "narrative_opportunity": "Custom analysis required",
            "recommended_angle": "Direct outreach",
            "account_score": "B",
            "reasoning": f"Parser error: {str(e)}",
            "ai_tools_detected": company_data.tech_stack or [],
            "confidence_score": 0.5
        }
    
    # Build response
    return ResearchResponse(
        company_name=company_data.company_name,
        industry=company_data.industry,
        account_score=analysis.get("account_score", "B"),
        confidence_score=analysis.get("confidence_score", 0.5),
        content_pain_points=analysis.get("content_pain_points", []),
        trigger_events=analysis.get("trigger_events", []),
        recommended_angle=analysis.get("recommended_angle", ""),
        narrative_opportunity=analysis.get("narrative_opportunity", ""),
        ai_tools_detected=analysis.get("ai_tools_detected", []),
        research_timestamp=datetime.utcnow().isoformat(),
        decision_makers=analysis.get("decision_makers", [])
    )

# API Endpoints
@app.get("/")
def root():
    """Root endpoint"""
    return {
        "service": "Storibot Lead Research Agent",
        "version": "1.0.0",
        "status": "operational",
        "docs": "/docs"
    }

@app.get("/health")
def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "api_key_configured": bool(ANTHROPIC_API_KEY)
    }

@app.post("/research/single", response_model=ResearchResponse)
def research_single_company(company: CompanyInput):
    """
    Research a single company and return intelligence
    
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
        result = research_company(company)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Research failed: {str(e)}")

@app.post("/research/batch")
def research_batch_companies(companies: List[CompanyInput]):
    """
    Research multiple companies in batch
    
    Example:
    [
        {"company_name": "Company A", "industry": "Healthcare", "employee_count": 200, "website": "a.com"},
        {"company_name": "Company B", "industry": "Education", "employee_count": 150, "website": "b.com"}
    ]
    """
    results = []
    for company in companies:
        try:
            result = research_company(company)
            results.append(result)
        except Exception as e:
            print(f"Error researching {company.company_name}: {e}")
            continue
    
    return {
        "total": len(companies),
        "successful": len(results),
        "results": results
    }

@app.post("/webhook/research")
def webhook_research(company: CompanyInput):
    """
    Webhook endpoint for Make.com, Zapier, n8n integration
    
    Returns simplified response optimized for automation
    """
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
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }

# Run with: uvicorn main:app --host 0.0.0.0 --port 8000
if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
