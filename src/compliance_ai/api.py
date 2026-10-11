"""REST API for credit scoring, regulation Q&A and decision review.

    uvicorn compliance_ai.api:app --reload      # docs at http://127.0.0.1:8000/docs
"""
from __future__ import annotations

from functools import lru_cache
from typing import Literal

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from compliance_ai import __version__
from compliance_ai.credit.data import DataValidationError
from compliance_ai.credit.predict import load_bundle, score

app = FastAPI(
    title="Compliance AI",
    version=__version__,
    description=("Explainable credit-risk scoring, financial sentiment NLP, a grounded GDPR and "
                 "EU AI Act assistant, a review agent and drift monitoring."),
)


class Applicant(BaseModel):
    """One loan application in the German Credit schema (amounts in Deutsche Mark)."""
    status: str = Field(examples=["0 <= ... < 200 DM"])
    duration: int = Field(ge=1, le=120, examples=[48], description="months")
    credit_history: str = Field(examples=["existing credits paid back duly till now"])
    purpose: str = Field(examples=["domestic appliances"])
    amount: int = Field(ge=1, le=100_000, examples=[5951])
    savings: str = Field(examples=["... < 100 DM"])
    employment_duration: str = Field(examples=["1 <= ... < 4 years"])
    installment_rate: int = Field(ge=1, le=4, examples=[2])
    personal_status_sex: str = Field(examples=["female : divorced/separated/married"])
    other_debtors: str = Field(examples=["none"])
    present_residence: int = Field(ge=1, le=4, examples=[2])
    property: str = Field(examples=["real estate"])
    age: int = Field(ge=18, le=100, examples=[22])
    other_installment_plans: str = Field(examples=["none"])
    housing: str = Field(examples=["own"])
    number_credits: int = Field(ge=1, le=10, examples=[1])
    job: str = Field(examples=["skilled employee/official"])
    people_liable: int = Field(ge=1, le=5, examples=[1])
    telephone: str = Field(examples=["no"])
    foreign_worker: str = Field(examples=["yes"])


class Sentences(BaseModel):
    sentences: list[str] = Field(min_length=1, max_length=64, examples=[[
        "Operating profit rose to EUR 13.1 mn from EUR 8.7 mn in the corresponding period in 2007.",
        "The company said it will cut 120 jobs as demand for its products fell sharply.",
    ]])


class Question(BaseModel):
    question: str = Field(min_length=5, max_length=500,
                          examples=["Can a loan be refused by an automated system alone?"])
    regulation: Literal["GDPR", "EU AI Act"] | None = Field(
        default=None, description="Search only this law; leave empty to search both")


class ScoreResponse(BaseModel):
    p_default: float
    decision: Literal["approve", "refer_or_reject"]
    threshold: float
    model: str
    reasons: list[dict]


@lru_cache(maxsize=1)
def get_assistant():
    from compliance_ai.rag.assistant import ComplianceAssistant

    return ComplianceAssistant()


@app.get("/health")
def health() -> dict:
    status = {"api": "ok", "version": __version__}
    try:
        bundle = load_bundle()
        status["credit_model"] = {"name": bundle["model_name"], "trained_at": bundle["trained_at"]}
    except FileNotFoundError:
        status["credit_model"] = "not trained"
    try:
        status["rag_index"] = get_assistant().index_meta
    except FileNotFoundError:
        status["rag_index"] = "not built"
    try:
        from compliance_ai.nlp.predict import load_model

        status["sentiment_model"] = load_model().kind
    except FileNotFoundError:
        status["sentiment_model"] = "not trained"
    return status


@app.post("/credit/score", response_model=ScoreResponse)
def credit_score(applicant: Applicant) -> dict:
    try:
        return score([applicant.model_dump()])[0]
    except FileNotFoundError as exc:
        raise HTTPException(503, str(exc)) from exc
    except DataValidationError as exc:
        raise HTTPException(422, str(exc)) from exc


@app.post("/compliance/ask")
def compliance_ask(q: Question) -> dict:
    try:
        return get_assistant().ask(q.question, q.regulation).__dict__
    except FileNotFoundError as exc:
        raise HTTPException(503, str(exc)) from exc


@app.post("/credit/review")
def credit_review(applicant: Applicant) -> dict:
    from compliance_ai.review import review

    try:
        return review(applicant.model_dump(), get_assistant())
    except FileNotFoundError as exc:
        raise HTTPException(503, str(exc)) from exc


@app.post("/nlp/sentiment")
def nlp_sentiment(body: Sentences) -> list[dict]:
    """Financial sentiment (negative / neutral / positive) from the investor's point of view."""
    from compliance_ai.nlp.predict import classify

    try:
        return classify(body.sentences)
    except FileNotFoundError as exc:
        raise HTTPException(503, str(exc)) from exc


@app.get("/", include_in_schema=False)
def home():
    from fastapi.responses import RedirectResponse

    return RedirectResponse("/docs")


class AgentRequest(BaseModel):
    applicant: Applicant
    question: str = Field(default="Review this loan application. What does the law require?",
                          max_length=500)


@app.post("/agent/review")
def agent_review(body: AgentRequest) -> dict:
    """The agent decides which tools to call (credit score, GDPR search, fairness audit, sentiment)."""
    from compliance_ai.agent import ComplianceAgent

    try:
        return ComplianceAgent(get_assistant()).run(body.applicant.model_dump(), body.question).__dict__
    except FileNotFoundError as exc:
        raise HTTPException(503, str(exc)) from exc


class Batch(BaseModel):
    applicants: list[Applicant] = Field(min_length=30, max_length=5000)


@app.post("/monitor/drift")
def monitor_drift(body: Batch) -> dict:
    """Compare a batch of applicants (30 or more) with the training population using PSI."""
    from compliance_ai.monitoring import check_drift

    try:
        return check_drift([a.model_dump() for a in body.applicants])
    except FileNotFoundError as exc:
        raise HTTPException(503, str(exc)) from exc
