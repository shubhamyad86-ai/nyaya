"""REST API for credit scoring, regulation Q&A and decision review.

    uvicorn compliance_ai.api:app --reload      # docs at http://127.0.0.1:8000/docs
"""
from __future__ import annotations

from functools import lru_cache
from typing import Literal

from fastapi import FastAPI, HTTPException, Request
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
    credit_score: float = Field(description="100 x (1 - P(default))")
    pass_mark: float = Field(description="Lowest credit score that passes: 100 x (1 - threshold)")
    result: Literal["PASS", "FAIL"]
    requires_human_review: bool = Field(description="FAIL goes to a human (GDPR Art. 22)")
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
    except ImportError:
        status["rag_index"] = "not installed"
    try:
        from compliance_ai.nlp.predict import load_model

        status["sentiment_model"] = load_model().kind
    except FileNotFoundError:
        status["sentiment_model"] = "not trained"
    except ImportError:
        status["sentiment_model"] = "not installed"
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

    return RedirectResponse("/app")


@app.get("/app", include_in_schema=False)
def scorer_page():
    """Web form: choose an applicant's values and get a credit score with PASS or FAIL."""
    from fastapi.responses import HTMLResponse

    from compliance_ai.web import render_scorer

    return HTMLResponse(render_scorer(score_url="/credit/score"))


@app.get("/batch", include_in_schema=False)
def batch_page():
    """Web page: upload a CSV, Excel or PDF file of applicants and check it against EU and Indian rules."""
    from fastapi.responses import HTMLResponse

    from compliance_ai.web import render_batch

    return HTMLResponse(render_batch(check_url="/batch/check", sample_url="/batch/sample.csv",
                                     template_url="/batch/template.csv"))


@app.get("/explore", include_in_schema=False)
def explore_page():
    """Web page: search the GDPR and the EU AI Act, and check the sentiment of financial news."""
    from fastapi.responses import HTMLResponse

    from compliance_ai.web import render_explore

    return HTMLResponse(render_explore())


_UPLOAD_DOC = {"requestBody": {"content": {
    "multipart/form-data": {"schema": {"type": "object", "required": ["file"], "properties": {
        "file": {"type": "string", "format": "binary",
                 "description": "CSV, Excel (.xlsx) or PDF with up to 3,000 applicants"}}}},
    "application/octet-stream": {"schema": {"type": "string", "format": "binary"}},
}}}


@app.post("/batch/check", openapi_extra=_UPLOAD_DOC)
async def batch_check(request: Request, jurisdictions: str = "EU,IN", filename: str = "upload.csv",
                      currency: str = "DM") -> dict:
    """Score every applicant in a file (PASS/FAIL, reasons) and check the batch against EU and/or
    Indian rules (`jurisdictions=EU,IN`). `currency` is the currency of the amount column: EUR, INR
    or DM (default, as in the training data). Send the file as multipart form field `file`, or as the
    raw request body with `?filename=` so the format can be recognised."""
    from fastapi.concurrency import run_in_threadpool

    from compliance_ai.batch import BatchError, check_file

    if request.headers.get("content-type", "").startswith("multipart/form-data"):
        form = await request.form()
        upload = form.get("file")
        if upload is None or not hasattr(upload, "read"):
            raise HTTPException(422, "Send the file in a form field called 'file'")
        data, filename = await upload.read(), upload.filename or filename
    else:
        data = await request.body()
    try:
        return await run_in_threadpool(check_file, data, filename, jurisdictions.split(","), True, currency)
    except (BatchError, ValueError) as exc:
        raise HTTPException(422, str(exc)) from exc
    except FileNotFoundError as exc:
        raise HTTPException(503, str(exc)) from exc


@app.get("/batch/template.csv", include_in_schema=False)
def batch_template():
    from fastapi.responses import Response

    from compliance_ai.batch import template_csv

    return Response(template_csv(), media_type="text/csv",
                    headers={"Content-Disposition": 'attachment; filename="nyaya_template.csv"'})


@app.get("/batch/sample.csv", include_in_schema=False)
def batch_sample():
    from fastapi.responses import Response

    from compliance_ai.web import sample_file

    return Response(sample_file(), media_type="text/csv",
                    headers={"Content-Disposition": 'attachment; filename="sample_applicants.csv"'})


@app.get("/credit/options")
def credit_options() -> dict:
    """Allowed values for every applicant field, and which fields the model does not use."""
    from compliance_ai.web import form_options

    return form_options()


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
