"""Agent tests with a scripted chat model, so the tool loop is tested without a real LLM."""
from __future__ import annotations

from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from compliance_ai.agent import ComplianceAgent
from compliance_ai.rag.assistant import ComplianceAssistant
from tests.conftest import MODEL, require


class ScriptedToolModel(BaseChatModel):
    """Returns pre-written AI messages in order; supports bind_tools like a real chat model."""
    script: list[AIMessage]
    calls: int = 0

    @property
    def _llm_type(self) -> str:
        return "scripted"

    def bind_tools(self, tools, **kwargs: Any):
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs) -> ChatResult:
        msg = self.script[min(self.calls, len(self.script) - 1)]
        self.calls += 1
        return ChatResult(generations=[ChatGeneration(message=msg)])


def call(name, args, i):
    return {"name": name, "args": args, "id": f"call_{i}", "type": "tool_call"}


def test_agent_uses_tools_and_cites_retrieved_law(applicant, tfidf_index):
    require(MODEL, "run python -m compliance_ai.credit.train")
    assistant = ComplianceAssistant(llm_provider="none", index_dir=tfidf_index)
    llm = ScriptedToolModel(script=[
        AIMessage(content="", tool_calls=[call("score_applicant", {}, 1)]),
        AIMessage(content="", tool_calls=[
            call("search_regulation", {"query": "decision based solely on automated processing"}, 2),
            call("check_fairness", {"attribute": "age_group"}, 3)]),
        AIMessage(content="High risk. The applicant may not be refused by the model alone "
                          "[GDPR Article 22]. Fines may follow [GDPR Article 999]."),
    ])
    result = ComplianceAgent(assistant, llm=llm).run(applicant, "Should we approve this loan?")
    assert result.mode == "agent"
    assert [s["tool"] for s in result.steps] == ["score_applicant", "search_regulation", "check_fairness"]
    assert result.requires_human_review is True            # set by policy from the score
    assert "GDPR Article 999" in result.unsupported_citations
    assert "GDPR Article 22" in result.regulation_sources


def test_agent_reports_unknown_tool_and_still_finishes(applicant, tfidf_index):
    require(MODEL, "run python -m compliance_ai.credit.train")
    assistant = ComplianceAssistant(llm_provider="none", index_dir=tfidf_index)
    llm = ScriptedToolModel(script=[
        AIMessage(content="", tool_calls=[call("delete_database", {}, 1)]),
        AIMessage(content="I could not find that tool. Refer to a human reviewer."),
    ])
    result = ComplianceAgent(assistant, llm=llm).run(applicant)
    assert "unknown tool" in result.steps[0]["output"]
    assert result.mode == "agent" and result.requires_human_review is True  # score fetched by policy


def test_fixed_plan_when_model_never_calls_tools(applicant, tfidf_index):
    require(MODEL, "run python -m compliance_ai.credit.train")
    assistant = ComplianceAssistant(llm_provider="none", index_dir=tfidf_index)
    llm = ScriptedToolModel(script=[AIMessage(content="Looks fine to me, approve.")])
    result = ComplianceAgent(assistant, llm=llm).run(applicant)
    assert result.mode == "fixed_plan"
    assert [s["tool"] for s in result.steps] == ["score_applicant", "check_fairness",
                                                 "search_regulation", "search_regulation"]
    assert "human reviewer" in result.answer and result.unsupported_citations == []
    assert "EU AI Act Annex III" in result.regulation_sources      # credit scoring is high-risk AI


def test_fixed_plan_without_llm(applicant, tfidf_index):
    require(MODEL, "run python -m compliance_ai.credit.train")
    assistant = ComplianceAssistant(llm_provider="none", index_dir=tfidf_index)
    result = ComplianceAgent(assistant).run(applicant)
    assert result.mode == "fixed_plan" and "(Fixed plan used" not in result.answer
    assert "under-25" in result.answer                    # applicant is 22 and the age audit fails
