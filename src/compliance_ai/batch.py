"""Check a whole file of loan applicants at once: CSV, Excel or PDF, up to 3,000 rows.

    from compliance_ai.batch import check_file
    report = check_file(open("applicants.pdf", "rb").read(), "applicants.pdf", ["EU", "IN"])

For every row: validation, a credit score with PASS or FAIL, the top reasons in plain language and
the action each jurisdiction's rules require. For the batch: a fairness audit, a drift check,
accuracy if outcomes are included, and the EU and India policy checks (see compliance_ai.policies).

A row with a bad value is reported on its own and does not stop the rest of the file. The uploaded
file is not stored: the audit log keeps only its SHA-256 fingerprint and the results.
"""
from __future__ import annotations

import hashlib
import io
import re
import uuid
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from compliance_ai.credit.data import CATEGORIES, EXPECTED_COLUMNS, NUMERIC_RANGES, add_features

MAX_ROWS = 3000
MAX_BYTES = 15 * 1024 * 1024
INPUT_COLUMNS = EXPECTED_COLUMNS[:-1]             # the 20 applicant fields
TARGET = EXPECTED_COLUMNS[-1]                     # credit_risk: optional, 1 = repaid, 0 = defaulted
REFERENCE_COLUMNS = ["applicant_id", "application_id", "customer_id", "id", "reference", "name"]
PROTECTED = {"age_group": "Age group", "sex": "Sex", "foreign_worker": "Foreign worker"}
MIN_GROUP = 20                                    # smaller groups get a caution in the fairness audit


class BatchError(ValueError):
    """The file as a whole cannot be checked (wrong format, too big, missing columns)."""


# ------------------------------------------------------------------ reading files
def _norm(name: object) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(name).strip().lower()).strip("_")


def _read_csv(data: bytes) -> pd.DataFrame:
    for encoding in ("utf-8-sig", "cp1252"):
        try:
            text = data.decode(encoding)
            break
        except UnicodeDecodeError:
            continue
    else:
        raise BatchError("Could not read the CSV text encoding; save it as UTF-8.")
    # sep=None sniffs the delimiter, so European exports with ";" work too.
    return pd.read_csv(io.StringIO(text), sep=None, engine="python", dtype=str, skipinitialspace=True)


def _read_excel(data: bytes) -> pd.DataFrame:
    try:
        return pd.read_excel(io.BytesIO(data), dtype=str)
    except ImportError as exc:
        raise BatchError("Excel support needs openpyxl: pip install openpyxl") from exc


def _table_from_pdf_tables(pages) -> pd.DataFrame | None:
    """PDFs with a table: find the header row on any page and stack the rows of every page."""
    header, rows = None, []
    for page in pages:
        for table in page.extract_tables():
            for raw in table:
                cells = ["" if c is None else str(c).replace("\n", " ").strip() for c in raw]
                normed = [_norm(c) for c in cells]
                if sum(c in INPUT_COLUMNS for c in normed) >= 10:        # a header row
                    header = normed
                    continue
                if header and len(cells) == len(header) and any(cells):
                    rows.append(cells)
    return pd.DataFrame(rows, columns=header) if header and rows else None


FIELD_LINE = re.compile(r"^\s*([A-Za-z][A-Za-z0-9 _/-]{1,40}?)\s*[:=]\s*(.+?)\s*$")


def _table_from_pdf_forms(pages) -> pd.DataFrame | None:
    """PDFs of application forms: 'field: value' lines; a repeated field starts the next applicant."""
    records, current = [], {}
    for page in pages:
        for line in (page.extract_text() or "").splitlines():
            m = FIELD_LINE.match(line)
            if not m:
                continue
            key = _norm(m.group(1))
            if key not in INPUT_COLUMNS and key not in REFERENCE_COLUMNS and key != TARGET:
                continue
            if key in current:
                records.append(current)
                current = {}
            current[key] = m.group(2)
    if current:
        records.append(current)
    return pd.DataFrame(records, dtype=str) if records else None


def _read_pdf(data: bytes) -> pd.DataFrame:
    try:
        import pdfplumber
    except ImportError as exc:
        raise BatchError("PDF support needs pdfplumber: pip install pdfplumber") from exc
    try:
        with pdfplumber.open(io.BytesIO(data)) as pdf:
            pages = list(pdf.pages)
            df = _table_from_pdf_tables(pages)
            if df is None:
                df = _table_from_pdf_forms(pages)
    except BatchError:
        raise
    except Exception as exc:  # damaged or encrypted PDFs
        raise BatchError(f"Could not open the PDF: {exc}") from exc
    if df is None:
        raise BatchError("No applicant table or 'field: value' form found in the PDF. Scanned PDFs "
                         "(images of paper) need OCR first; export the data as CSV or Excel if you can.")
    return df


def read_table(data: bytes, filename: str) -> pd.DataFrame:
    """Read CSV, Excel or PDF into a DataFrame of strings with normalised column names."""
    if not data:
        raise BatchError("The file is empty.")
    if len(data) > MAX_BYTES:
        raise BatchError(f"The file is larger than {MAX_BYTES // 1024 // 1024} MB.")
    name = filename.lower()
    if name.endswith(".pdf") or data[:5] == b"%PDF-":
        df = _read_pdf(data)
    elif name.endswith((".xlsx", ".xlsm")):
        df = _read_excel(data)
    elif name.endswith((".csv", ".txt", ".tsv")) or not name:
        df = _read_csv(data)
    else:
        raise BatchError("Upload a CSV, Excel (.xlsx) or PDF file.")
    df.columns = [_norm(c) for c in df.columns]
    df = df.dropna(how="all")
    if len(df) > MAX_ROWS:
        raise BatchError(f"The file has {len(df):,} applicants; the limit is {MAX_ROWS:,} per upload. "
                         "Split it into smaller files.")
    missing = [c for c in INPUT_COLUMNS if c not in df.columns]
    if missing:
        raise BatchError(f"Missing column(s): {', '.join(missing)}. Download the template to see the "
                         "expected columns.")
    if df.empty:
        raise BatchError("The file has a header but no applicants.")
    return df.reset_index(drop=True)


# ------------------------------------------------------------------ row validation
def validate_rows(df: pd.DataFrame, money: str = "DM") -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split into (valid applicants with typed values, invalid rows with the reasons).

    `money` is the currency of the `amount` column (EUR, INR or DM). Amounts are converted to whole
    DM, the unit the model was trained on; the limits in error messages are shown in `money`.
    """
    from compliance_ai import currency

    money = currency.check(money)
    clean = pd.DataFrame(index=df.index)
    issues = pd.Series([[] for _ in range(len(df))], index=df.index)

    def flag(mask: pd.Series, text: str) -> None:
        for i in mask[mask].index:
            issues[i].append(text)

    for col in INPUT_COLUMNS:
        raw = df[col].astype("string").str.strip()
        flag(raw.isna() | (raw == ""), f"{col} is empty")
        if col in NUMERIC_RANGES:
            lo, hi = NUMERIC_RANGES[col]
            num = pd.to_numeric(raw.str.replace(",", "", regex=False), errors="coerce")
            flag(raw.notna() & (raw != "") & num.isna(), f"{col} is not a number")
            if col == "amount" and money != "DM":
                num = (num * currency.dm_per_unit(money)).round()     # to whole DM for the model
                flag(num.notna() & ~num.between(lo, hi),
                     f"amount must be {currency.money(lo, money)}-{currency.money(hi, money)}")
            else:
                flag(num.notna() & ~num.between(lo, hi), f"{col} must be {lo}-{hi}")
                flag(num.notna() & (num % 1 != 0), f"{col} must be a whole number")
            clean[col] = num
        else:
            allowed = CATEGORIES[col]
            lookup = {a.lower(): a for a in allowed}
            value = raw.str.lower().map(lookup)       # accept any capitalisation
            flag(raw.notna() & (raw != "") & value.isna(), f"{col}: unknown value")
            clean[col] = value

    if TARGET in df.columns:
        target = pd.to_numeric(df[TARGET], errors="coerce")
        clean[TARGET] = target.where(target.isin([0, 1]))

    ref_col = next((c for c in REFERENCE_COLUMNS if c in df.columns), None)
    clean["reference"] = df[ref_col].astype(str) if ref_col else ""
    clean["row"] = df.index + 2                       # line in the file (header is line 1)

    bad = issues.map(len) > 0
    invalid = pd.DataFrame({"row": clean.loc[bad, "row"], "reference": clean.loc[bad, "reference"],
                            "issues": issues[bad].map("; ".join)})
    valid = clean.loc[~bad].copy()
    for col in NUMERIC_RANGES:
        valid[col] = valid[col].astype(int)
    return valid.reset_index(drop=True), invalid.reset_index(drop=True)


# ------------------------------------------------------------------ batch analysis
def _fairness(valid: pd.DataFrame, approved: np.ndarray) -> list[dict]:
    feats = add_features(valid[INPUT_COLUMNS])
    out = []
    for col, label in PROTECTED.items():
        rates = pd.Series(approved, index=feats.index).groupby(feats[col]).agg(["mean", "size"])
        if len(rates) < 2:
            continue
        best = rates["mean"].max()
        groups = [{"group": str(g), "applicants": int(r["size"]), "approval_rate": round(float(r["mean"]), 3),
                   "disparate_impact": round(float(r["mean"] / best), 3) if best > 0 else None}
                  for g, r in rates.iterrows()]
        ratios = [g["disparate_impact"] for g in groups if g["disparate_impact"] is not None]
        lowest = min(ratios, default=1.0)
        out.append({"attribute": col, "label": label, "groups": groups,
                    "min_disparate_impact": lowest, "passes_four_fifths": lowest >= 0.8,
                    "small_group": min(g["applicants"] for g in groups) < MIN_GROUP})
    return out


def _accuracy(valid: pd.DataFrame, approved: np.ndarray) -> dict | None:
    if TARGET not in valid.columns or valid[TARGET].isna().all():
        return None
    known = valid[TARGET].notna().to_numpy()
    y_default = (1 - valid.loc[known, TARGET]).to_numpy()
    rejected = ~approved[known]
    n_bad = int(y_default.sum())
    return {"labelled": int(known.sum()),
            "accuracy": round(float((rejected == (y_default == 1)).mean()), 3),
            "defaulters_caught": round(float(rejected[y_default == 1].mean()), 3) if n_bad else None,
            "default_rate_among_approved": round(float(y_default[~rejected].mean()), 3)
            if (~rejected).any() else None}


YES_NO = {"has_checking_account", "no_savings", "delayed_before"}


def _show(reason: dict, money: str = "DM"):
    """Readable reason values: yes/no for flags, amounts in the file's currency."""
    from compliance_ai import currency

    value = reason["value"]
    if reason["feature"] in YES_NO:
        return "yes" if value else "no"
    if reason["feature"] in currency.MONEY_FEATURES:
        return currency.money(value, money)
    if isinstance(value, str):
        return currency.relabel(value, money)
    return round(value, 1) if isinstance(value, float) else value


def check_file(data: bytes, filename: str, jurisdictions: list[str] | None = None,
               write_audit_log: bool = True, money: str = "DM") -> dict:
    """Read, validate, score and audit a file; return everything the page and the API show.

    `money` is the currency of the file's `amount` column: EUR, INR or DM (the training data).
    """
    from compliance_ai import currency
    from compliance_ai.credit.predict import load_bundle, score
    from compliance_ai.monitoring import MIN_BATCH, check_drift
    from compliance_ai.policies import evaluate, load_packs, write_audit

    packs = load_packs(jurisdictions or ["EU", "IN"])
    started = datetime.now(timezone.utc)
    money = currency.check(money)
    df = read_table(data, filename)
    valid, invalid = validate_rows(df, money)
    bundle = load_bundle()

    results = score(valid[INPUT_COLUMNS]) if len(valid) else []
    approved = np.array([r["result"] == "PASS" for r in results], dtype=bool)

    applicants = []
    for i, r in enumerate(results):
        applicants.append({
            "row": int(valid.at[i, "row"]), "reference": valid.at[i, "reference"],
            "credit_score": r["credit_score"], "result": r["result"], "p_default": r["p_default"],
            "requires_human_review": r["requires_human_review"],
            "reasons": [{"label": x["label"], "direction": x["direction"], "value": _show(x, money)}
                        for x in r["reasons"]],
            "actions": {p.code: p.action(r["result"]) for p in packs},
        })

    drift = None
    if len(valid) >= MIN_BATCH:
        d = check_drift(valid[INPUT_COLUMNS])
        drift = {"overall": d["overall"], "summary": d["summary"],
                 "features": [{k: f[k] for k in ("feature", "psi", "status")} for f in d["features"][:8]]}

    n = len(df)
    batch = {
        "batch_id": uuid.uuid4().hex[:12],
        "checked_at": started.isoformat(timespec="seconds"),
        "file": {"name": filename, "sha256": hashlib.sha256(data).hexdigest(), "rows": n},
        "currency": currency.info(money),
        "model": {"name": bundle["model_name"], "trained_at": bundle["trained_at"],
                  "threshold": round(float(bundle["threshold"]), 4),
                  "pass_mark": round(100 * (1 - float(bundle["threshold"])), 1),
                  "inputs": bundle["numeric"] + bundle["categorical"]},
        "summary": {"rows": n, "scored": len(valid), "invalid": len(invalid),
                    "passed": int(approved.sum()), "failed": int((~approved).sum()),
                    "approval_rate": round(float(approved.mean()), 3) if len(approved) else None,
                    "human_review": int((~approved).sum())},
        "fairness": _fairness(valid, approved) if len(valid) else [],
        "drift": drift,
        "drift_min_rows": MIN_BATCH,
        "accuracy": _accuracy(valid, approved) if len(valid) else None,
        "applicants": applicants,
        "invalid_rows": invalid.to_dict("records"),
    }
    batch["audit_enabled"] = write_audit_log
    batch["policies"] = [evaluate(p, batch) for p in packs]
    if write_audit_log:
        batch["audit_log"] = write_audit(batch)
    return batch


def template_csv() -> str:
    """An empty CSV with the expected header plus one example row."""
    example = {"applicant_id": "A-0001", "status": "no checking account", "duration": 18,
               "credit_history": "existing credits paid back duly till now",
               "purpose": "domestic appliances", "amount": 1473, "savings": "... < 100 DM",
               "employment_duration": "... < 1 year", "installment_rate": 3,
               "personal_status_sex": "male : married/widowed", "other_debtors": "none",
               "present_residence": 4, "property": "real estate", "age": 39,
               "other_installment_plans": "none", "housing": "own", "number_credits": 1,
               "job": "skilled employee/official", "people_liable": 1, "telephone": "yes",
               "foreign_worker": "yes"}
    return pd.DataFrame([example]).to_csv(index=False)
