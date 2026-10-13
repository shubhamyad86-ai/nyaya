"""Rewrite the result sections of README.md and docs/MODEL_CARD.md from your own reports.

    python scripts/update_docs.py

Model selection, thresholds and fairness numbers depend on the library versions installed, so a run
on another computer can pick a different model (for example logistic regression instead of XGBoost).
This keeps the documentation consistent with reports/metrics.json, rag_eval_tfidf.json and
drift_report.json. Only text between <!-- auto:NAME --> and <!-- /auto:NAME --> is replaced.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"

NAMES = {
    "xgboost": "XGBoost",
    "gradient_boosting": "Gradient boosting",
    "logistic_regression": "Logistic regression",
    "random_forest": "Random forest",
}
DESCRIPTIONS = {
    "xgboost": "XGBoost gradient-boosted trees",
    "gradient_boosting": "scikit-learn gradient-boosted trees",
    "logistic_regression": "Logistic regression (class-weighted)",
    "random_forest": "Random forest",
}
FEATURE_LABELS = {
    "has_checking_account": "checking-account status", "status": "checking-account balance",
    "amount": "loan amount", "duration": "duration", "amount_per_month": "monthly repayment",
    "credit_history": "credit history", "purpose": "loan purpose", "savings": "savings",
    "installment_rate": "instalment rate", "employment_duration": "employment length",
    "property": "property", "housing": "housing", "other_installment_plans": "other instalment plans",
    "no_savings": "no savings", "delayed_before": "past payment delays", "age": "age",
    "log_amount": "loan amount (log)",
}


def pct(x: float) -> str:
    return f"{100 * x:.0f}%"


def f2(x: float) -> str:
    return f"{x:.2f}"


def load(name: str) -> dict | None:
    path = REPORTS / name
    return json.loads(path.read_text()) if path.exists() else None


def group(fair: dict, attribute: str, name: str) -> dict:
    return next(g for g in fair[attribute]["groups"] if g["group"] == name)


def joined(items: list[str]) -> str:
    items = [i.replace("_", " ") for i in items]
    return items[0] if len(items) == 1 else ", ".join(items[:-1]) + " and " + items[-1]


# ---------------------------------------------------------------- credit

def credit_blocks(m: dict) -> dict[str, str]:
    t, cv = m["test"], {r["model"]: r for r in m["cv_comparison"]}
    sel = m["selected_model"]
    name = NAMES.get(sel, sel)
    c = t["confusion"]
    oof = m["fairness_out_of_fold"]
    age_di = oof["age_group"]["min_disparate_impact_ratio"]
    sex_di = oof["sex"]["min_disparate_impact_ratio"]
    fw_di = oof["foreign_worker"]["min_disparate_impact_ratio"]
    young, older = group(oof, "age_group", "under_25"), group(oof, "age_group", "25_plus")
    mit = m.get("fairness_mitigation", {})
    with_age = mit.get("with_age", {}).get("age_min_disparate_impact")
    without_age = mit.get("without_age", {}).get("age_min_disparate_impact")
    age_is_input = "age" in m["data"]["features_numeric"] + m["data"]["features_categorical"]
    n_features = len(m["data"]["features_numeric"]) + len(m["data"]["features_categorical"])

    def verdict(di: float) -> str:
        return "passes" if di >= 0.8 else "fails"

    readme = f"""**Credit model: {name}** (held-out test set, 200 applicants)

| Metric | Value |
|---|---|
| ROC AUC | {t['roc_auc']:.3f} (CV {cv[sel]['cv_roc_auc_mean']:.3f} ± {cv[sel]['cv_roc_auc_std']:.3f}) |
| Defaulters caught | {pct(t['recall_defaulters'])} ({c['tp']} of {c['tp'] + c['fn']}) |
| Expected cost per applicant | {t['expected_cost']:.3f} vs {t['cost_reject_everyone']:.2f} reject-all and \
{t['cost_approve_everyone']:.2f} approve-all |

**Fairness:** {"age is a model input in this run; " if age_is_input else "age, "}sex and foreign-worker \
status are {"" if age_is_input else "not "}{"audit-only" if age_is_input else "model inputs"}. \
Sex {verdict(sex_di)} the four-fifths rule ({f2(sex_di)}). Age {verdict(age_di)} ({f2(age_di)}, \
out-of-fold): under-25s are approved {pct(young['approval_rate'])} of the time versus \
{pct(older['approval_rate'])} for older applicants."""
    if with_age is not None and without_age is not None:
        effect = ("barely changes this" if abs(without_age - with_age) < 0.05
                  else "helps only partly")
        readme += (f" A controlled experiment shows that removing age as an input {effect} (disparate "
                   f"impact {f2(with_age)} with it, {f2(without_age)} without) and does not reach 0.80, "
                   "because proxies such as housing and employment length carry the signal, so the model "
                   "card recommends a dedicated mitigation step.")
    readme += " See [docs/MODEL_CARD.md](docs/MODEL_CARD.md)."

    header = f"""| | |
|---|---|
| Model | {DESCRIPTIONS.get(sel, sel)} inside a scikit-learn pipeline (scaling + one-hot encoding) |
| Task | Estimate the probability that a loan applicant defaults; refer high-risk cases to a human |
| Data | Statlog German Credit (UCI), 1,000 real applicants, 30% defaulted |
| Selection | Lowest expected business cost across 4 models, 5-fold stratified cross-validation |
| Decision rule | Refer or reject when P(default) >= {t['threshold']:.2f} |
| Inputs | {n_features} features. {"Sex/marital status and foreign-worker status are excluded" if age_is_input
                                  else "Age, sex/marital status and foreign-worker status are excluded; "
                                  "`log_amount` (a duplicate of `amount`) was dropped"} |
| Tracking | MLflow (parameters, CV metrics, fairness metrics, artefacts, model) |
| Monitoring | PSI drift check per feature and on the scores, against a profile saved at training |"""

    rows = sorted(m["cv_comparison"], key=lambda r: r["oof_expected_cost"])
    cv_rows = "\n".join(
        f"| {NAMES.get(r['model'], r['model'])} | {r['cv_roc_auc_mean']:.3f} ± {r['cv_roc_auc_std']:.3f} | "
        f"{r['cv_pr_auc_mean']:.3f} | {r['cv_brier_mean']:.3f} | {r['oof_best_threshold']:.2f} | "
        + (f"**{r['oof_expected_cost']:.3f}**" if r["model"] == sel else f"{r['oof_expected_cost']:.3f}") + " |"
        for r in rows)
    best_auc = max(m["cv_comparison"], key=lambda r: r["cv_roc_auc_mean"])["model"]
    spread = rows[-1]["oof_expected_cost"] - rows[0]["oof_expected_cost"]

    performance = f"""## Performance (20% held-out test set, 200 applicants)

| Metric | Value |
|---|---|
| ROC AUC | {t['roc_auc']:.3f} |
| PR AUC (defaulters) | {t['pr_auc']:.3f} |
| Brier score | {t['brier']:.3f} |
| Defaulters caught (recall) | {pct(t['recall_defaulters'])} ({c['tp']} of {c['tp'] + c['fn']}) |
| Precision on refer/reject | {pct(t['precision_defaulters'])} |
| Approval rate | {pct(t['approval_rate'])} |
| Expected cost per applicant | {t['expected_cost']:.3f} |

Cost baselines on the same test set: approve everyone {t['cost_approve_everyone']:.2f}, reject everyone \
{t['cost_reject_everyone']:.2f}, this model at the default 0.5 threshold {t['cost_default_threshold_0_5']:.3f}. \
The 5:1 cost matrix makes missed defaulters expensive, so the cost-optimal threshold is low and the \
approval rate is low. With 200 test applicants, expect a few points of noise on every number.

## Cross-validated comparison (800 training applicants)

| Model | ROC AUC (mean ± sd) | PR AUC | Brier | Best threshold | Expected cost |
|---|---|---|---|---|---|
{cv_rows}

The models are close: expected costs span only {spread:.3f} per applicant, within cross-validation \
noise, so a different library version can change which one wins. {NAMES.get(best_auc, best_auc)} has \
the best ranking quality (ROC AUC). {"The selected logistic regression is also the easiest to explain to a "
"credit committee." if sel == "logistic_regression" else "The interpretable logistic regression is a "
"credible alternative where simplicity matters more than a few percent of cost."}"""

    def frow(label: str, attr: str, a: str, b: str, la: str, lb: str, di: float, note: str) -> str:
        ga, gb = group(oof, attr, a), group(oof, attr, b)
        return (f"| {label} | {la} ({ga['n']}) / {lb} ({gb['n']}) | {pct(ga['approval_rate'])} / "
                f"{pct(gb['approval_rate'])} | {pct(ga['approval_rate_if_repaid'])} / "
                f"{pct(gb['approval_rate_if_repaid'])} | {pct(ga['actual_default_rate'])} / "
                f"{pct(gb['actual_default_rate'])} | {note} |")

    test_age = m["fairness_test"]["age_group"]["min_disparate_impact_ratio"]
    fairness = f"""## Fairness audit

Sex{"" if age_is_input else ", age"} and foreign-worker status are never model inputs; they are kept only \
for auditing. Figures below use out-of-fold predictions on 800 applicants for steadier group estimates.

| Attribute | Groups (n) | Approval rate | Approval rate if repaid | Actual default rate | Disparate impact |
|---|---|---|---|---|---|
{frow("Sex", "sex", "female", "male", "female", "male", sex_di, f"{f2(sex_di)} - {verdict(sex_di)} 4/5 rule")}
{frow("Age", "age_group", "under_25", "25_plus", "under 25", "25+", age_di,
      f"**{f2(age_di)} - {verdict(age_di)}**")}
{frow("Foreign worker", "foreign_worker", "yes", "no", "yes", "no", fw_di,
      f"{f2(fw_di)} - {verdict(fw_di)}, tiny group")}

On the 200-applicant test set the age ratio is {f2(test_age)}, but only about 40 test applicants are \
under 25, so the out-of-fold figure is the more reliable one.

Findings
- **Age is the material issue.** Younger applicants do default more often in this data \
({pct(young['actual_default_rate'])} vs {pct(older['actual_default_rate'])}), but the model approves them \
at {pct(young['approval_rate'])} versus {pct(older['approval_rate'])}, and creditworthy under-25s are \
approved far less often ({pct(young['approval_rate_if_repaid'])} vs {pct(older['approval_rate_if_repaid'])})."""
    if with_age is not None and without_age is not None:
        fairness += f"""
- **Removing age does not fix it.** It is still far below 0.80. In a controlled experiment the age disparate impact is \
{f2(with_age)} with age as an input and {f2(without_age)} without it, with expected cost \
{mit['with_age']['expected_cost']:.3f} vs {mit['without_age']['expected_cost']:.3f}. Age information \
reaches the model through proxies; the drift check on under-25 applicants points at housing and \
employment duration. Next step: a dedicated mitigation (group-aware thresholds or reweighing), with \
legal review of which approach is permissible."""
    fairness += """
- The foreign-worker result rests on 31 people and should not be read as reliable."""

    top = [FEATURE_LABELS.get(f, f.replace("_", " ")) for f in list(m["top_features"])[:5]]
    explain = (f"SHAP values give global importance (`reports/shap_importance.png`) and per-applicant "
               f"reason codes. Top drivers: {', '.join(top)}.")
    explain += (" Age is an input in this run and can appear in reason codes, which is one more reason to "
                "remove it before any real use." if age_is_input else
                " Age is not an input, so it cannot appear in reason codes; proxies for it can.")

    return {"credit": readme, "card-header": header, "card-performance": performance,
            "card-fairness": fairness, "card-explain": explain}


# ---------------------------------------------------------------- RAG and drift

def rag_block(r: dict) -> str:
    o, f = r["overall"], r["with_regulation_filter"]
    by = r.get("by_regulation", {})
    counts = ", ".join(f"{v['search_all']['n_questions']} {k}" for k, v in sorted(by.items(), key=lambda x: x[0] != "GDPR"))
    text = f"""**RAG retrieval** ({o['n_questions']} hand-labelled questions: {counts}; \
{'offline TF-IDF' if r['embedding_backend'] == 'tfidf' else 'semantic ' + r['embedding_model']} embeddings)

| Search | hit@1 | hit@4 | primary article in top 4 | MRR@10 |
|---|---|---|---|---|
| Both laws together | {f2(o['hit@1'])} | {f2(o['hit@4'])} | {f2(o['primary_article@4'])} | {f2(o['mrr@10'])} |
| Filtered to the right law | {f2(f['hit@1'])} | {f2(f['hit@4'])} | {f2(f['primary_article@4'])} | \
{f2(f['mrr@10'])} |
"""
    if "GDPR" in by:
        g = by["GDPR"]
        text += f"""
The two laws use similar language, so searching both makes GDPR questions harder (hit@4 \
{f2(g['search_all']['hit@4'])}); filtering by regulation recovers it ({f2(g['filtered']['hit@4'])}). These \
are baseline numbers from the offline index; `python scripts/compare_retrieval.py` measures the semantic \
`all-MiniLM-L6-v2` index on the same questions."""
    return text


def drift_rows(d: dict, short: bool) -> str:
    rows = []
    for name, rep in d.items():
        label = name.split(" (")[0] if short else name
        shifted = [f["feature"] for f in rep["features"] if f["status"] == "significant_shift"]
        watch = [f["feature"] for f in rep["features"] if f["status"] == "monitor"]
        p = rep.get("prediction", {})
        mean = (f"; mean P(default) {p['reference_mean']:.2f} → {p['current_mean']:.2f}"
                if p and abs(p["current_mean"] - p["reference_mean"]) >= 0.05 else "")
        if rep["overall"] == "alert":
            result = f"ALERT: {joined(shifted[:4]) if shifted else 'model scores'}{mean}"
        elif rep["overall"] == "watch":
            result = f"WATCH: {joined(watch[:4]) if watch else 'model scores'}{mean}"
        else:
            result = "OK"
        rows.append(f"| {label} | {result} |")
    return "| Batch | Result |\n|---|---|\n" + "\n".join(rows)


def sentiment_rows(path: Path) -> str | None:
    import csv

    if not path.exists():
        return None
    with open(path, newline="", encoding="utf-8") as fh:
        rows = {r["method"]: r for r in csv.DictReader(fh)}   # last run of each method wins
    labels = [("tfidf_baseline", "TF-IDF + logistic regression"),
              ("frozen_embeddings", "Frozen MiniLM embeddings + logistic regression"),
              ("fine_tuned_transformer", "DistilBERT fine-tuned (PyTorch)")]
    lines = ["| Method | Accuracy | Macro-F1 | F1 negative | F1 neutral | F1 positive |",
             "|---|---|---|---|---|---|"]
    for key, label in labels:
        r = rows.get(key)
        if r:
            lines.append(f"| {label} | {float(r['accuracy']):.3f} | **{float(r['macro_f1']):.3f}** | "
                         f"{float(r['f1_negative']):.3f} | {float(r['f1_neutral']):.3f} | "
                         f"{float(r['f1_positive']):.3f} |")
        else:
            lines.append(f"| {label} | not run yet - see notebooks/financial_sentiment_colab.ipynb | | | | |")
    return "\n".join(lines)


def highlights(metrics: dict | None, rag: dict | None, nlp_path: Path) -> str | None:
    import csv

    cells = []
    if metrics:
        t = metrics["test"]
        name = NAMES.get(metrics["selected_model"], metrics["selected_model"])
        cells.append((f"{t['roc_auc']:.2f}", f"credit ROC AUC ({name})"))
        cells.append((pct(t["recall_defaulters"]), "of defaulters caught"))
    if nlp_path.exists():
        with open(nlp_path, newline="", encoding="utf-8") as fh:
            rows = list(csv.DictReader(fh))
        if rows:
            best = max(rows, key=lambda r: float(r["macro_f1"]))
            label = "DistilBERT macro-F1" if best["method"] == "fine_tuned_transformer" else "sentiment macro-F1"
            cells.append((f"{float(best['macro_f1']):.3f}", label))
    if rag and "with_regulation_filter" in rag:
        cells.append((pct(rag["with_regulation_filter"]["hit@4"]), "right law article in top 4"))
    if not cells:
        return None
    return ("| " + " | ".join(f"**{v}**" for v, _ in cells) + " |\n"
            + "|" + "---|" * len(cells) + "\n"
            + "| " + " | ".join(k for _, k in cells) + " |")


# ---------------------------------------------------------------- write

def add_highlights_marker(text: str) -> str:
    """Add an empty Key results block above "Why this project exists" if the README has none."""
    anchor = "## Why this project exists"
    if "<!-- auto:highlights -->" in text or anchor not in text:
        return text
    return text.replace(anchor, "## Key results\n\n<!-- auto:highlights -->\n(filled by update_docs.py)\n<!-- /auto:highlights -->\n\n"
                        "---\n\n" + anchor, 1)


def add_sentiment_markers(text: str) -> str:
    """Older READMEs have the sentiment table without markers; wrap it once."""
    header = "| Method | Accuracy | Macro-F1"
    if "<!-- auto:sentiment -->" in text or header not in text:
        return text
    start = text.index(header)
    end = text.find("\n\n", start)
    end = len(text) if end == -1 else end
    return (text[:start] + "<!-- auto:sentiment -->\n" + text[start:end] + "\n<!-- /auto:sentiment -->"
            + text[end:])


def replace_block(text: str, name: str, content: str) -> tuple[str, bool]:
    pattern = re.compile(rf"(<!-- auto:{re.escape(name)} -->\n).*?(\n<!-- /auto:{re.escape(name)} -->)", re.S)
    if not pattern.search(text):
        return text, False
    return pattern.sub(lambda mt: mt.group(1) + content.strip() + mt.group(2), text), True


def main() -> int:
    metrics, rag, drift = load("metrics.json"), load("rag_eval_tfidf.json"), load("drift_report.json")
    blocks: dict[str, str] = {}
    if metrics:
        blocks.update(credit_blocks(metrics))
    if rag and "with_regulation_filter" in rag:
        blocks["rag"] = rag_block(rag)
    if drift:
        blocks["drift"] = drift_rows(drift, short=False)
        blocks["card-drift"] = drift_rows(drift, short=True)
    top = highlights(metrics, rag, REPORTS / "nlp_comparison.csv")
    if top:
        blocks["highlights"] = top
    sentiment = sentiment_rows(REPORTS / "nlp_comparison.csv")
    if sentiment:
        blocks["sentiment"] = sentiment
    if not blocks:
        print("No reports found - run setup.bat (or the training commands) first.")
        return 1

    for doc in (ROOT / "README.md", ROOT / "docs/MODEL_CARD.md"):
        text = add_highlights_marker(add_sentiment_markers(doc.read_text(encoding="utf-8")))
        updated = []
        for name, content in blocks.items():
            text, ok = replace_block(text, name, content)
            if ok:
                updated.append(name)
        doc.write_text(text, encoding="utf-8")
        print(f"{doc.relative_to(ROOT)}: updated {', '.join(updated) or 'nothing'}")
    if metrics:
        print(f"Selected model: {NAMES.get(metrics['selected_model'], metrics['selected_model'])}, "
              f"test ROC AUC {metrics['test']['roc_auc']:.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
