"""
FastAPI Microservice for Loan Eligibility Agent.
Exposes RESTful endpoints for autonomous credit underwriting, explainability, and what-if simulation.
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Dict, Any, List, Optional
from src.agent import LoanEligibilityAgent
from src.sample_decisions import SAMPLE_PERSONAS
from src.conversational_agent import ConversationalUnderwriter
from src.model import get_trained_model

app = FastAPI(
    title="LendingClub Loan Eligibility Agent API",
    description="Institutional credit underwriting agent with explainable AI, adverse action disclosure, and what-if simulation.",
    version="1.0.0"
)

# Initialize agent singleton
model = get_trained_model()
agent = LoanEligibilityAgent(model=model)
conv_assistant = ConversationalUnderwriter(agent=agent)


class ApplicantProfileSchema(BaseModel):
    applicant_id: Optional[str] = "APP-API-USER"
    annual_inc: float = Field(..., ge=1000, description="Gross annual income in USD")
    loan_amnt: float = Field(..., ge=1000, le=40000, description="Requested loan amount in USD")
    term: str = Field(default="36 months", description="'36 months' or '60 months'")
    purpose: str = Field(default="debt_consolidation", description="Loan purpose")
    fico_score: float = Field(..., ge=300, le=850, description="Credit Bureau FICO score")
    dti: float = Field(..., ge=0.0, le=100.0, description="Debt-to-Income ratio %")
    revol_util: float = Field(default=40.0, ge=0.0, le=100.0, description="Revolving credit utilization %")
    revol_bal: float = Field(default=10000.0, ge=0.0, description="Total revolving balance in USD")
    emp_length_years: float = Field(default=3.0, ge=0.0, le=50.0, description="Years employed")
    home_ownership: str = Field(default="RENT", description="'MORTGAGE', 'RENT', or 'OWN'")
    delinq_2yrs: int = Field(default=0, ge=0, description="Past 2-year 30+ day delinquencies")
    inq_last_6mths: int = Field(default=0, ge=0, description="Credit inquiries in last 6 months")
    open_acc: int = Field(default=10, ge=1, description="Number of open credit lines")
    total_acc: int = Field(default=20, ge=1, description="Total credit lines")
    pub_rec_bankruptcies: int = Field(default=0, ge=0, description="Number of bankruptcies")
    gemini_api_key: Optional[str] = Field(default=None, description="Optional Google Gemini API key to activate LLM synthesis")
    use_llm: bool = Field(default=True, description="Whether to enable LLM underwriting synthesis")


class WhatIfRequestSchema(BaseModel):
    profile: ApplicantProfileSchema
    modified_field: str = Field(..., description="Field to modify (e.g. 'loan_amnt', 'fico_score', 'dti')")
    new_value: float = Field(..., description="New candidate value")


class ChatQuerySchema(BaseModel):
    profile: ApplicantProfileSchema
    message: str = Field(..., description="User question to the Underwriting Agent")


@app.get("/")
def read_root():
    return {
        "service": "LendingClub Loan Eligibility Agent API",
        "status": "operational",
        "documentation": "/docs",
        "endpoints": ["/health", "/personas", "/underwrite", "/what-if", "/chat"]
    }


@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "model_loaded": agent.model.is_trained,
        "metrics": agent.model.metrics,
        "llm_status": agent.llm_engine.get_status() if hasattr(agent, "llm_engine") and agent.llm_engine else None
    }


@app.get("/personas")
def get_sample_personas():
    """Retrieve preloaded approved, rejected, and borderline case studies."""
    return SAMPLE_PERSONAS


@app.post("/underwrite")
def underwrite_applicant(profile: ApplicantProfileSchema):
    """Run complete underwriting evaluation, factor attribution, and plain-language explanation."""
    try:
        data = profile.model_dump()
        gemini_key = data.pop("gemini_api_key", None)
        use_llm = data.pop("use_llm", True)
        if gemini_key and hasattr(agent, "llm_engine") and agent.llm_engine:
            agent.llm_engine.configure(api_key=gemini_key)
        result = agent.evaluate_applicant(data, use_llm=use_llm)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/what-if")
def simulate_what_if(req: WhatIfRequestSchema):
    """Simulate a what-if adjustment against the applicant's profile."""
    try:
        data = req.profile.model_dump()
        original_result = agent.evaluate_applicant(data)

        sim_data = dict(data)
        sim_data[req.modified_field] = req.new_value
        sim_result = agent.evaluate_applicant(sim_data)

        return {
            "modified_field": req.modified_field,
            "old_value": data.get(req.modified_field),
            "new_value": req.new_value,
            "before": {
                "decision": original_result["decision"],
                "approval_probability": original_result["approval_probability"],
                "grade": original_result["grade"],
                "estimated_apr": original_result["estimated_apr"]
            },
            "after": {
                "decision": sim_result["decision"],
                "approval_probability": sim_result["approval_probability"],
                "grade": sim_result["grade"],
                "estimated_apr": sim_result["estimated_apr"]
            },
            "decision_flipped": original_result["decision"] != sim_result["decision"]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/chat")
def chat_with_agent(chat_req: ChatQuerySchema):
    """Conversational endpoint to ask the Underwriting Agent questions."""
    try:
        data = chat_req.profile.model_dump()
        gemini_key = data.pop("gemini_api_key", None)
        use_llm = data.pop("use_llm", True)
        if gemini_key and hasattr(agent, "llm_engine") and agent.llm_engine:
            agent.llm_engine.configure(api_key=gemini_key)
        eval_result = agent.evaluate_applicant(data, use_llm=use_llm)
        answer = conv_assistant.respond(chat_req.message, data, eval_result, use_llm=use_llm)
        return {
            "query": chat_req.message,
            "response": answer,
            "decision": eval_result["decision"],
            "llm_enhanced": eval_result.get("llm_enhanced", False)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="127.0.0.1", port=8000, reload=True)
