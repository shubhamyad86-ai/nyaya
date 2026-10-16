"""Jurisdiction policy packs (EU, India) and the automatic checks behind them.

A pack is a YAML file in this folder: the action for PASS and FAIL, notices, and a list of checks,
each with the law it comes from. The check functions below look at a finished batch (see
compliance_ai.batch.check_file) and return pass, warn, fail or info with a one-line finding.

The checks show whether the decision process meets each rule. They are not legal advice, and a
"pass" does not replace a review by the compliance team.
"""
from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from importlib import resources
from pathlib import Path

import yaml

PACK_FILES = {"EU": "eu.yaml", "IN": "india.yaml"}
ALIASES = {"EU": "EU", "EUROPE": "EU", "IN": "IN", "INDIA": "IN"}
PROTECTED_INPUTS = {"personal_status_sex": "sex", "age": "age", "foreign_worker": "nationality",
                    "sex": "sex", "age_group": "age"}

log = logging.getLogger("compliance_ai.audit")


@dataclass
class PolicyPack:
    code: str
    name: str
    applies_to: str
    actions: dict
    short_actions: dict = field(default_factory=dict)
    notices: list = field(default_factory=list)
    checks: list = field(default_factory=list)

    def action(self, result: str) -> str:
        return self.actions[result]


def load_packs(codes: list[str]) -> list[PolicyPack]:
    packs = []
    for raw in codes:
        code = ALIASES.get(str(raw).strip().upper())
        if code is None:
            raise ValueError(f"Unknown jurisdiction {raw!r}; use one of {sorted(PACK_FILES)}")
        if code not in [p.code for p in packs]:
            text = resources.files(__package__).joinpath(PACK_FILES[code]).read_text(encoding="utf-8")
            packs.append(PolicyPack(**yaml.safe_load(text)))
    if not packs:
        raise ValueError("Choose at least one jurisdiction")
    return packs


# ------------------------------------------------------------------ checks
def _result(status: str, finding: str) -> dict:
    return {"status": status, "finding": finding}


def every_fail_has_human_review(batch: dict, _: dict) -> dict:
    fails = [a for a in batch["applicants"] if a["result"] == "FAIL"]
    missing = [a for a in fails if not a["requires_human_review"]]
    if missing:
        return _result("fail", f"{len(missing)} refused applicant(s) are not flagged for review")
    return _result("pass", f"All {len(fails):,} FAIL decisions are routed to a person; none is an "
                           "automatic rejection")


def every_decision_has_reasons(batch: dict, _: dict) -> dict:
    without = [a for a in batch["applicants"] if not a["reasons"]]
    if without:
        return _result("fail", f"{len(without)} decision(s) have no reasons")
    n = len(batch["applicants"])
    return _result("pass", f"All {n:,} decisions list their top {len(batch['applicants'][0]['reasons'])} "
                           "factors in plain language" if n else "No decisions in this file")


def protected_attributes_not_inputs(batch: dict, _: dict) -> dict:
    used = sorted({PROTECTED_INPUTS[c] for c in batch["model"]["inputs"] if c in PROTECTED_INPUTS})
    if used:
        return _result("fail", f"The model uses: {', '.join(used)}")
    return _result("pass", "The model does not use sex, age or nationality; they are kept only for "
                           "this fairness audit")


def four_fifths(batch: dict, cfg: dict) -> dict:
    limit = float(cfg.get("min_ratio", 0.8))
    audits = batch["fairness"]
    if not audits:
        return _result("info", "Not enough groups in this file to compare approval rates")
    failing = [a for a in audits if a["min_disparate_impact"] < limit]
    small = [a["label"] for a in audits if a["small_group"]]
    note = f" (small groups in: {', '.join(small)}, so ratios are noisy)" if small else ""
    if failing:
        worst = ", ".join(f"{a['label']} {a['min_disparate_impact']:.2f}" for a in failing)
        return _result("warn", f"Lowest approval-rate ratio below {limit}: {worst}. Document a review of "
                               f"the lending policy for these groups{note}")
    return _result("pass", f"All groups at or above {limit} of the best group's approval rate{note}")


def invalid_share(batch: dict, cfg: dict) -> dict:
    limit = float(cfg.get("max_invalid_share", 0.05))
    s = batch["summary"]
    share = s["invalid"] / s["rows"] if s["rows"] else 0.0
    if s["invalid"] == 0:
        return _result("pass", f"All {s['rows']:,} rows passed validation")
    status = "warn" if share <= limit else "fail"
    return _result(status, f"{s['invalid']:,} of {s['rows']:,} rows ({share:.1%}) have errors and were not "
                           f"scored; correct and resubmit them" + ("" if status == "warn"
                                                                     else f" (limit {limit:.0%})"))


def drift(batch: dict, _: dict) -> dict:
    d = batch["drift"]
    if d is None:
        return _result("info", f"Needs at least {batch['drift_min_rows']} valid applicants")
    status = {"ok": "pass", "watch": "warn", "alert": "fail"}[d["overall"]]
    return _result(status, d["summary"])


def audit_logged(batch: dict, _: dict) -> dict:
    # Evaluated before the log is written; write_audit() updates the status if writing fails.
    if not batch.get("audit_enabled", True):
        return _result("info", "Audit logging was switched off for this run")
    return _result("pass", f"Batch {batch['batch_id']}: file fingerprint, model version, decisions and "
                           "check results are written to the audit log")


def file_not_stored(batch: dict, _: dict) -> dict:
    return _result("pass", f"Only the SHA-256 fingerprint {batch['file']['sha256'][:12]}... is kept")


CHECKS = {f.__name__: f for f in (every_fail_has_human_review, every_decision_has_reasons,
                                  protected_attributes_not_inputs, four_fifths, invalid_share, drift,
                                  audit_logged, file_not_stored)}


def evaluate(pack: PolicyPack, batch: dict) -> dict:
    results = []
    for cfg in pack.checks:
        outcome = CHECKS[cfg["check"]](batch, cfg)
        results.append({"id": cfg["id"], "title": cfg["title"], "law": cfg["law"], **outcome})
    out = {"code": pack.code, "name": pack.name, "applies_to": pack.applies_to, "actions": pack.actions,
           "short_actions": pack.short_actions or pack.actions,
           "notices": pack.notices, "checks": results}
    return _tally(out)


def _tally(pack_result: dict) -> dict:
    checks = pack_result["checks"]
    counts = {s: sum(c["status"] == s for c in checks) for s in ("pass", "warn", "fail", "info")}
    pack_result["counts"] = counts
    pack_result["overall"] = "fail" if counts["fail"] else "warn" if counts["warn"] else "pass"
    return pack_result


# ------------------------------------------------------------------ audit log
def audit_log_path() -> Path:
    from compliance_ai.config import resolve

    return Path(os.environ.get("NYAYA_AUDIT_LOG") or resolve("reports") / "audit_log.jsonl")


def write_audit(batch: dict) -> dict:
    """Append one line per batch: no personal data, only counts, decisions by row and the checks."""
    record = {
        "batch_id": batch["batch_id"], "checked_at": batch["checked_at"],
        "file_sha256": batch["file"]["sha256"], "rows": batch["summary"]["rows"],
        "model": batch["model"]["name"], "model_trained_at": batch["model"]["trained_at"],
        "threshold": batch["model"]["threshold"], "summary": batch["summary"],
        "decisions": [[a["row"], a["result"], a["credit_score"]] for a in batch["applicants"]],
        "policies": {p["code"]: {c["id"]: c["status"] for c in p["checks"]} for p in batch["policies"]},
    }
    log.info("batch checked %s", json.dumps({k: v for k, v in record.items() if k != "decisions"}))
    path = audit_log_path()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record) + "\n")
        return {"written": True, "where": path.name}
    except OSError as exc:  # e.g. a read-only file system: the logging line above still exists
        for p in batch["policies"]:
            for c in p["checks"]:
                if c["id"] == "logging":
                    c.update(status="warn", finding=f"Could not write the audit file ({exc.strerror}); "
                                                    "the batch was logged to the application log only")
            _tally(p)
        return {"written": False, "where": "application log"}
