"""Azure Functions (Python v2 model) wrapper for the credit-scoring model and drift check.

Build and deploy: see docs/AZURE_DEPLOY.md
    python deploy/azure_function/build.py     # -> dist/compliance_ai_function.zip
    bash deploy_azure.sh                      # in Azure Cloud Shell

Routes (function key required except /health and the public demo page)
    GET  /api/health                      -> model name, threshold, training time
    GET  /api/app                         -> web page: choose an applicant's values, see score + PASS/FAIL
    POST /api/app/score                   -> the page's scoring call (same result as /api/score)
    GET  /api/batch                       -> web page: upload a CSV, Excel or PDF of up to 3,000 applicants
    POST /api/batch/check?jurisdictions=EU,IN&filename=x.csv   (raw file as the body)
                                          -> PASS/FAIL per applicant + EU and India policy checks
    POST /api/score   one applicant       -> P(default), decision and reason codes
    POST /api/drift   {"applicants": [...]} (30+) -> PSI drift report against the training data
"""
import json
import logging

import azure.functions as func

app = func.FunctionApp(http_auth_level=func.AuthLevel.FUNCTION)


def _json(body: dict | list, status: int = 200) -> func.HttpResponse:
    return func.HttpResponse(json.dumps(body), status_code=status, mimetype="application/json")


def _error(message: str, status: int) -> func.HttpResponse:
    return _json({"error": message}, status)


@app.route(route="health", methods=["GET"], auth_level=func.AuthLevel.ANONYMOUS)
def health(req: func.HttpRequest) -> func.HttpResponse:
    from compliance_ai.credit.predict import load_bundle

    try:
        bundle = load_bundle()
    except FileNotFoundError as exc:
        return _error(str(exc), 503)
    return _json({"status": "ok", "model": bundle["model_name"],
                  "threshold": round(float(bundle["threshold"]), 3), "trained_at": bundle["trained_at"]})


@app.route(route="app", methods=["GET"], auth_level=func.AuthLevel.ANONYMOUS)
def scorer_page(req: func.HttpRequest) -> func.HttpResponse:
    from compliance_ai.web import render_scorer

    return _page(render_scorer(score_url="/api/app/score", single_url="/api/app", batch_url="/api/batch"))


def _page(html: str) -> func.HttpResponse:
    return func.HttpResponse(html, mimetype="text/html")


@app.route(route="batch", methods=["GET"], auth_level=func.AuthLevel.ANONYMOUS)
def batch_page(req: func.HttpRequest) -> func.HttpResponse:
    from compliance_ai.web import render_batch

    return _page(render_batch(check_url="/api/batch/check", sample_url="/api/batch/sample",
                              template_url="/api/batch/template", single_url="/api/app",
                              batch_url="/api/batch"))


@app.route(route="batch/check", methods=["POST"], auth_level=func.AuthLevel.ANONYMOUS)
def batch_check(req: func.HttpRequest) -> func.HttpResponse:
    """Public demo: the file is processed in memory and not stored (scale is capped in deploy_azure.sh)."""
    from compliance_ai.batch import BatchError, check_file

    jurisdictions = (req.params.get("jurisdictions") or "EU,IN").split(",")
    filename = req.params.get("filename") or "upload.csv"
    try:
        report = check_file(req.get_body(), filename, jurisdictions, True,
                            req.params.get("currency") or "DM")
    except (BatchError, ValueError) as exc:
        return _error(str(exc), 422)
    return _json(report)


@app.route(route="batch/template", methods=["GET"], auth_level=func.AuthLevel.ANONYMOUS)
def batch_template(req: func.HttpRequest) -> func.HttpResponse:
    from compliance_ai.batch import template_csv

    return func.HttpResponse(template_csv(), mimetype="text/csv",
                             headers={"Content-Disposition": 'attachment; filename="nyaya_template.csv"'})


@app.route(route="batch/sample", methods=["GET"], auth_level=func.AuthLevel.ANONYMOUS)
def batch_sample(req: func.HttpRequest) -> func.HttpResponse:
    from compliance_ai.web import sample_file

    return func.HttpResponse(sample_file(), mimetype="text/csv")


@app.route(route="app/score", methods=["POST"], auth_level=func.AuthLevel.ANONYMOUS)
def scorer_page_score(req: func.HttpRequest) -> func.HttpResponse:
    """Public twin of /api/score for the demo page. It stores nothing; scale is capped in deploy_azure.sh."""
    return _score(req)


@app.route(route="score", methods=["POST"])
def score_applicant(req: func.HttpRequest) -> func.HttpResponse:
    return _score(req)


def _score(req: func.HttpRequest) -> func.HttpResponse:
    from compliance_ai.credit.data import DataValidationError
    from compliance_ai.credit.predict import score

    try:
        applicant = req.get_json()
    except ValueError:
        return _error("Body must be JSON", 400)
    if not isinstance(applicant, dict):
        return _error("Send one applicant as a JSON object", 400)
    try:
        result = score([applicant])[0]
    except (DataValidationError, KeyError) as exc:
        return _error(f"Invalid applicant: {exc}", 422)
    logging.info("scored applicant: score=%.1f result=%s", result["credit_score"], result["result"])
    return _json(result)


@app.route(route="drift", methods=["POST"])
def drift(req: func.HttpRequest) -> func.HttpResponse:
    from compliance_ai.credit.data import DataValidationError
    from compliance_ai.monitoring import check_drift

    try:
        body = req.get_json()
    except ValueError:
        return _error("Body must be JSON", 400)
    applicants = body.get("applicants") if isinstance(body, dict) else None
    if not isinstance(applicants, list):
        return _error('Send {"applicants": [ ... ]}', 400)
    try:
        report = check_drift(applicants)
    except (DataValidationError, KeyError, ValueError) as exc:
        return _error(str(exc), 422)
    logging.info("drift check: n=%d overall=%s", report["n_current"], report["overall"])
    return _json(report)
