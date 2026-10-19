"""The applicant scorer page: pick an applicant's values, get a credit score with PASS or FAIL.

The same HTML is served by the FastAPI app (/app) and by the Azure Function (/api/app); only the
URL the page posts to differs.
"""
from __future__ import annotations

import json
from importlib import resources

from compliance_ai import currency
from compliance_ai.credit.data import CATEGORIES, NUMERIC_RANGES
from compliance_ai.credit.predict import FEATURE_LABELS

# Collected only for the fairness audit; the model never sees them (see config.yaml).
AUDIT_ONLY = {
    "age": "Age (years)",
    "personal_status_sex": "Sex and marital status",
    "foreign_worker": "Foreign worker",
}

# Form layout: (section title, fields in order).
SECTIONS = [
    ("The loan", ["purpose", "amount", "duration", "installment_rate"]),
    ("Money and credit", ["status", "savings", "credit_history", "number_credits",
                          "other_installment_plans", "other_debtors"]),
    ("Work and home", ["employment_duration", "job", "housing", "property", "present_residence",
                       "people_liable", "telephone"]),
    ("Fairness audit only", ["age", "personal_status_sex", "foreign_worker"]),
]

# Typical ranges in the training data, shown as hints (validation uses NUMERIC_RANGES).
# Amounts are in DM here; the page converts them to the chosen currency.
TYPICAL = {"amount": (250, 18_424), "duration": (4, 72), "age": (19, 75)}

# Real applicants from the dataset, so the page can be tried with one click.
PRESETS = {
    "Strong": {
        "status": "... < 100 DM", "duration": 6, "credit_history": "existing credits paid back duly till now",
        "purpose": "radio/television", "amount": 1374, "savings": "... < 100 DM",
        "employment_duration": "1 <= ... < 4 years", "installment_rate": 1,
        "personal_status_sex": "male : single", "other_debtors": "none", "present_residence": 2,
        "property": "real estate", "age": 36, "other_installment_plans": "bank", "housing": "own",
        "number_credits": 1, "job": "unskilled - resident", "people_liable": 1, "telephone": "yes",
        "foreign_worker": "yes",
    },
    "Borderline": {
        "status": "no checking account", "duration": 18,
        "credit_history": "existing credits paid back duly till now", "purpose": "domestic appliances",
        "amount": 1473, "savings": "... < 100 DM", "employment_duration": "... < 1 year",
        "installment_rate": 3, "personal_status_sex": "male : married/widowed", "other_debtors": "none",
        "present_residence": 4, "property": "real estate", "age": 39, "other_installment_plans": "none",
        "housing": "own", "number_credits": 1, "job": "skilled employee/official", "people_liable": 1,
        "telephone": "yes", "foreign_worker": "yes",
    },
    "High risk": {
        "status": "... < 100 DM", "duration": 36,
        "credit_history": "existing credits paid back duly till now",
        "purpose": "car (used)", "amount": 8229, "savings": "... < 100 DM",
        "employment_duration": "1 <= ... < 4 years", "installment_rate": 2,
        "personal_status_sex": "male : single", "other_debtors": "none", "present_residence": 2,
        "property": "building society savings agreement/life insurance", "age": 26,
        "other_installment_plans": "none", "housing": "own", "number_credits": 1,
        "job": "skilled employee/official", "people_liable": 2, "telephone": "no", "foreign_worker": "yes",
    },
}


def form_options() -> dict:
    """Every applicant field with its label, allowed values or range, and whether the model uses it."""
    sections = []
    for title, names in SECTIONS:
        fields = []
        for name in names:
            field = {"name": name, "label": AUDIT_ONLY.get(name) or FEATURE_LABELS.get(name, name),
                     "model_input": name not in AUDIT_ONLY}
            if name in CATEGORIES:
                field.update(type="choice", options=CATEGORIES[name])
            else:
                lo, hi = NUMERIC_RANGES[name]
                field.update(type="number", min=lo, max=hi)
                if name in TYPICAL:
                    field["typical"] = list(TYPICAL[name])
            fields.append(field)
        sections.append({"title": title, "fields": fields})
    # Codebook labels that contain DM amounts, rewritten for each currency (the values stay the same).
    labels = {c["code"]: {v: currency.relabel(v, c["code"]) for opts in CATEGORIES.values() for v in opts
                          if "DM" in v} for c in currency.all_info()}
    return {"sections": sections, "presets": PRESETS, "currencies": currency.all_info(),
            "default_currency": currency.default(), "money_labels": labels,
            "money_features": sorted(currency.MONEY_FEATURES)}


def _page(name: str, **values) -> str:
    html = resources.files(__package__).joinpath(name).read_text(encoding="utf-8")
    # Values are written as JSON, and "</" is escaped so none can close the <script> tag early.
    for key, value in values.items():
        html = html.replace(f"__{key.upper()}__", json.dumps(value).replace("</", "<\\/"))
    return html


def _links(html: str, single_url: str, batch_url: str, template_url: str = "",
           explore_url: str = "") -> str:
    """Plain href attributes (navigation links), HTML-escaped.

    The law search tab is only shown where the search exists (not on the Azure Function).
    """
    from html import escape

    tab = (f'<a href="{escape(explore_url, quote=True)}">Law search &amp; sentiment</a>'
           if explore_url else "")
    html = html.replace("__EXPLORE_TAB__", tab)
    links = {"SINGLE_URL": single_url, "BATCH_URL": batch_url, "TEMPLATE_URL": template_url,
             "EXPLORE_URL": explore_url}
    for key, value in links.items():
        html = html.replace(f"__{key}__", escape(value, quote=True))
    return html


def render_scorer(score_url: str, single_url: str = "/app", batch_url: str = "/batch",
                  explore_url: str = "/explore") -> str:
    """The one-applicant page with the form options and the scoring URL filled in."""
    html = _page("scorer.html", options_json=form_options(), score_url=score_url)
    return _links(html, single_url, batch_url, explore_url=explore_url)


def render_batch(check_url: str, sample_url: str, template_url: str,
                 single_url: str = "/app", batch_url: str = "/batch", explore_url: str = "/explore") -> str:
    """The file upload page."""
    html = _page("batch.html", check_url=check_url, sample_url=sample_url)
    return _links(html, single_url, batch_url, template_url, explore_url)


def render_explore(ask_url: str = "/compliance/ask", sentiment_url: str = "/nlp/sentiment",
                   single_url: str = "/app", batch_url: str = "/batch", explore_url: str = "/explore") -> str:
    """The law search and news sentiment page."""
    html = _page("explore.html", ask_url=ask_url, sentiment_url=sentiment_url)
    return _links(html, single_url, batch_url, explore_url=explore_url)


def sample_file() -> bytes:
    """203 applicants (200 real rows from the data, 3 with deliberate errors) for trying the upload."""
    return resources.files(__package__).joinpath("sample_applicants.csv").read_bytes()
