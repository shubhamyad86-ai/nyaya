"""Azure Functions (Python v2 model) wrapper for the credit-scoring model and drift check.

Build and deploy: see docs/AZURE_DEPLOY.md
    python deploy/azure_function/build.py     # -> dist/compliance_ai_function.zip
    bash deploy_azure.sh                      # in Azure Cloud Shell

Routes (function key required except /health)
    GET  /api/health                      -> model name, threshold, training time
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


@app.route(route="score", methods=["POST"])
def score_applicant(req: func.HttpRequest) -> func.HttpResponse:
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
    logging.info("scored applicant: p_default=%.3f decision=%s", result["p_default"], result["decision"])
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
