"""Decision review workflow: links the credit model to the regulation assistant.

For one applicant it
1. scores the application and extracts SHAP reason codes (credit model tool),
2. retrieves the regulation passages that govern an automated credit decision
   (GDPR Articles 13-15 and 22 on transparency and automated decisions),
3. writes a short reviewer memo with the LLM, grounded in both, or a
   template memo when no LLM is available.

Under GDPR Article 22 a refusal should not rest solely on the model, so every
"refer_or_reject" decision is routed to a human reviewer.
"""
from __future__ import annotations

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from compliance_ai.credit.predict import score
from compliance_ai.rag.assistant import ComplianceAssistant, check_citations, format_context

REG_QUERY = ("rights of a person subject to automated decision-making and profiling, "
             "human intervention, and meaningful information about the logic involved")

MEMO_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You write short credit-review memos for a human reviewer. Use only the model output and the "
     "regulation passages given. Cite passages as [label]. Do not decide the case; recommend next steps."),
    ("human",
     "Model output:\n{model_output}\n\nRegulation passages:\n{context}\n\n"
     "Write a memo of at most 8 sentences: the risk estimate, the main reasons in plain language, "
     "and the regulatory obligations that apply before a final decision."),
])


def _model_summary(result: dict) -> str:
    reasons = "; ".join(f"{r['feature']}={r['value']} ({r['direction']})" for r in result["reasons"])
    return (f"P(default)={result['p_default']:.2f} vs threshold {result['threshold']:.2f} "
            f"-> {result['decision']}. Top factors: {reasons}.")


def review(applicant: dict, assistant: ComplianceAssistant | None = None) -> dict:
    result = score([applicant])[0]
    assistant = assistant or ComplianceAssistant()
    docs = [d for d, _ in assistant.retrieve(REG_QUERY, k=4)]
    citations = [d.metadata["citation"] for d in docs]
    human_review = result["decision"] != "approve"

    memo, mode, unsupported = None, "template", []
    if assistant.llm is not None:
        try:
            chain = MEMO_PROMPT | assistant.llm | StrOutputParser()
            memo = chain.invoke({"model_output": _model_summary(result), "context": format_context(docs)})
            mode, unsupported = "llm", check_citations(memo, docs)
        except Exception:
            memo = None
    if memo is None:
        memo = (_model_summary(result) + " "
                + ("Route to a human reviewer before any refusal; the applicant may request human "
                   "intervention, express their view and contest the decision, and must receive "
                   "meaningful information about the logic involved. " if human_review else
                   "Approval may proceed; keep the reason codes on file for transparency requests. ")
                + "Relevant provisions: " + ", ".join(dict.fromkeys(citations)) + ".")

    return {"model": result, "requires_human_review": human_review, "memo": memo,
            "memo_mode": mode, "regulation_sources": list(dict.fromkeys(citations)),
            "unsupported_citations": unsupported}
