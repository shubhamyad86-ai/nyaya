"""Compliance review agent: an LLM that decides which tools to call to review a loan case.

Tools (the agent chooses which to call, in what order, with what arguments)
- score_applicant      : credit model -> P(default), decision and SHAP reason codes
- search_regulation    : RAG search over the GDPR and the EU AI Act -> cited passages
- check_fairness       : the model's audited approval-rate gaps for sex, age or foreign workers
- classify_sentiment   : financial sentiment of news about the applicant's employer or sector

Guardrails (enforced in code, not left to the model)
- At most `max_steps` tool rounds; unknown tools or bad arguments are reported back, not executed.
- Human review is decided by policy: any `refer_or_reject` score requires a human, whatever the LLM writes.
- Every citation in the final answer is checked against passages the tools actually returned.
- If the model cannot call tools (no LLM, a model without tool support, or a failure), a fixed
  plan runs instead: score -> fairness -> regulation search -> template memo.
The full tool trace is returned so a reviewer can audit how the answer was produced.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool

from compliance_ai.config import load_config, resolve
from compliance_ai.rag.assistant import check_citations

SYSTEM = """You are a credit-risk compliance agent helping a human reviewer.
Use the tools to gather facts before answering:
- call score_applicant to get the model's risk estimate and reasons,
- call search_regulation for the legal rules that apply: the GDPR on automated decisions (search
  "automated decision-making human intervention", regulation "GDPR") and the EU AI Act, where credit
  scoring is a high-risk use (search "creditworthiness credit score high-risk", regulation "EU AI Act"),
- call check_fairness for "age_group", "sex" or "foreign_worker" when a protected characteristic
  matters, e.g. a young applicant.
Then answer in at most 8 sentences: the risk, the main reasons, the legal obligations with citations in
square brackets such as [GDPR Article 22], and a recommended next step. Do not make the final decision;
a human reviewer does. Only cite passages returned by search_regulation."""


@dataclass
class AgentResult:
    answer: str
    requires_human_review: bool
    mode: str                                   # "agent" or "fixed_plan"
    steps: list[dict] = field(default_factory=list)
    regulation_sources: list[str] = field(default_factory=list)
    unsupported_citations: list[str] = field(default_factory=list)


class ComplianceAgent:
    def __init__(self, assistant=None, llm: BaseChatModel | None = None, max_steps: int = 5):
        if assistant is None:
            from compliance_ai.rag.assistant import ComplianceAssistant

            assistant = ComplianceAssistant()
        self.assistant = assistant
        self.llm = llm if llm is not None else assistant.llm
        self.max_steps = max_steps

    # ---- tools -------------------------------------------------------------------------
    def _make_tools(self, applicant: dict, state: dict):
        assistant = self.assistant

        @tool
        def score_applicant() -> str:
            """Score the applicant under review with the credit-risk model. Returns P(default),
            the decision (approve or refer_or_reject) and the top reasons."""
            from compliance_ai.credit.predict import score

            result = score([applicant])[0]
            state["score"] = result
            return json.dumps(result)

        @tool
        def search_regulation(query: str, regulation: str = "") -> str:
            """Search the regulations for passages relevant to the query. Optionally restrict the search
            with regulation = "GDPR" or "EU AI Act". Returns labelled passages."""
            reg = regulation if regulation in ("GDPR", "EU AI Act") else None
            docs = [d for d, _ in assistant.retrieve(query, k=3, regulation=reg)]
            state["docs"].extend(docs)
            return "\n\n".join(f"[{d.metadata['citation']}] {d.page_content[:700]}" for d in docs)

        @tool
        def check_fairness(attribute: str) -> str:
            """Audited approval-rate gaps of the credit model for one protected attribute:
            'age_group', 'sex' or 'foreign_worker'."""
            path = resolve(load_config()["credit"]["report_dir"]) / "metrics.json"
            audit = json.loads(path.read_text()).get("fairness_out_of_fold", {})
            if attribute not in audit:
                return f"Unknown attribute {attribute!r}; use one of {sorted(audit)}."
            res = audit[attribute]
            state["fairness"][attribute] = res
            groups = "; ".join(
                f"{g['group']}: approval {g['approval_rate']:.0%}, "
                f"approval if repaid {g['approval_rate_if_repaid']:.0%}"
                for g in res["groups"]
            )
            verdict = "passes" if res["passes_four_fifths_rule"] else "FAILS"
            return (f"{attribute}: {groups}. Disparate impact {res['min_disparate_impact_ratio']:.2f} "
                    f"({verdict} the four-fifths rule).")

        @tool
        def classify_sentiment(text: str) -> str:
            """Financial sentiment (positive, negative, neutral) of a news sentence, for example about
            the applicant's employer or sector."""
            from compliance_ai.nlp.predict import classify

            r = classify([text])[0]
            return f"{r['label']} (confidence {r['confidence']:.2f})"

        return [score_applicant, search_regulation, check_fairness, classify_sentiment]

    # ---- runs ----------------------------------------------------------------------------
    def run(self, applicant: dict, question: str = "Review this loan application.") -> AgentResult:
        state = {"score": None, "docs": [], "fairness": {}}
        tools = self._make_tools(applicant, state)
        if self.llm is not None:
            try:
                result = self._agent_loop(tools, state, applicant, question)
                if result is not None:
                    return result
            except Exception as exc:   # model without tool support, Ollama down, ...
                state["error"] = f"{type(exc).__name__}: {exc}"[:200]
        return self._fixed_plan(tools, state, applicant)

    def _agent_loop(self, tools, state, applicant, question) -> AgentResult | None:
        by_name = {t.name: t for t in tools}
        llm = self.llm.bind_tools(tools)
        facts = {k: v for k, v in applicant.items() if k in ("age", "amount", "duration", "purpose")}
        messages = [SystemMessage(SYSTEM),
                    HumanMessage(f"{question}\nKey facts about the applicant: {json.dumps(facts)}")]
        steps: list[dict] = []
        for _ in range(self.max_steps):
            reply: AIMessage = llm.invoke(messages)
            messages.append(reply)
            if not reply.tool_calls:
                break
            for call in reply.tool_calls:
                name, args = call["name"], call.get("args") or {}
                if name not in by_name:
                    output = f"Error: unknown tool {name!r}. Available: {sorted(by_name)}."
                else:
                    try:
                        output = by_name[name].invoke(args)
                    except Exception as exc:
                        output = f"Error running {name}: {exc}"
                steps.append({"tool": name, "args": args, "output": str(output)[:400]})
                messages.append(ToolMessage(content=str(output), tool_call_id=call["id"]))
        else:
            reply = None   # step budget exhausted without a final answer

        answer = ""
        if isinstance(reply, AIMessage) and isinstance(reply.content, str):
            answer = reply.content.strip()
        if not steps or not answer:
            return None    # the model did not use tools or did not finish: use the fixed plan
        if state["score"] is None:   # policy needs the score even if the model skipped it
            by_name["score_applicant"].invoke({})
        return self._result(answer, "agent", steps, state)

    def _fixed_plan(self, tools, state, applicant) -> AgentResult:
        by_name = {t.name: t for t in tools}
        plan = [("score_applicant", {}), ("check_fairness", {"attribute": "age_group"}),
                ("search_regulation", {"query": "automated decision-making human intervention "
                                                "right to contest the decision", "regulation": "GDPR"}),
                ("search_regulation", {"query": "AI systems used to evaluate the creditworthiness of natural "
                                                "persons or establish their credit score",
                                       "regulation": "EU AI Act"})]
        steps = [{"tool": n, "args": a, "output": str(by_name[n].invoke(a))[:400]} for n, a in plan]
        s = state["score"]
        reasons = ", ".join(f"{r['feature']}={r['value']} ({r['direction']})" for r in s["reasons"][:3])
        cites = list(dict.fromkeys(d.metadata["citation"] for d in state["docs"]))
        age = state["fairness"].get("age_group", {})
        lines = [
            f"Estimated probability of default {s['p_default']:.0%} (threshold {s['threshold']:.0%}): "
            f"{s['decision']}. Main factors: {reasons}.",
        ]
        if s["decision"] != "approve":
            if "GDPR Article 22" in cites:
                text = ("A refusal must not rest on the model alone: route the case to a human reviewer "
                        "and let the applicant express their view and contest the decision "
                        "[GDPR Article 22]")
                if "GDPR Recital 71" in cites:
                    text += ", including obtaining an explanation of the decision [GDPR Recital 71]"
                lines.append(text + ".")
            else:
                provisions = ", ".join(f"[{c}]" for c in cites)
                lines.append("Route the case to a human reviewer before any refusal"
                             + (f"; relevant provisions: {provisions}." if cites else "."))
        if "EU AI Act Annex III" in cites:
            lines.append("Credit scoring is a high-risk AI use under the EU AI Act [EU AI Act Annex III], so "
                         "human oversight, logging and explanations to affected persons are required.")
        if age and not age.get("passes_four_fifths_rule", True) and applicant.get("age", 99) < 25:
            lines.append("Caution: the model approves under-25 applicants far less often (disparate "
                         f"impact {age['min_disparate_impact_ratio']:.2f}); check the decision is not "
                         "driven by age.")
        note = ""
        if self.llm is not None:
            note = " (Fixed plan used: the language model could not run the tools.)"
        return self._result(" ".join(lines) + note, "fixed_plan", steps, state)

    def _result(self, answer, mode, steps, state) -> AgentResult:
        cites = list(dict.fromkeys(d.metadata["citation"] for d in state["docs"]))
        return AgentResult(
            answer=answer,
            requires_human_review=state["score"]["decision"] != "approve",   # policy, not the LLM
            mode=mode,
            steps=steps,
            regulation_sources=cites,
            unsupported_citations=check_citations(answer, state["docs"]),
        )
