"""The law search and sentiment page, and the light setup the free Render server uses."""
import re

from fastapi.testclient import TestClient

from compliance_ai.api import app
from compliance_ai.rag.assistant import ComplianceAssistant
from compliance_ai.web import render_batch, render_explore, render_scorer

client = TestClient(app)


def test_explore_page_has_no_unfilled_placeholders():
    page = client.get("/explore").text
    assert re.findall(r"__[A-Z_]+__", page) == []
    assert 'const ASK_URL = "/compliance/ask"' in page
    assert 'const SENTIMENT_URL = "/nlp/sentiment"' in page


def test_every_page_links_to_the_others():
    for html in (render_scorer("/credit/score"), render_batch("/c", "/s", "/t"), render_explore()):
        assert 'href="/app"' in html and 'href="/batch"' in html and 'href="/explore"' in html


def test_azure_pages_hide_the_search_tab():
    # The Azure Function has no search index, so its pages must not link to one.
    html = render_scorer("/api/app/score", "/api/app", "/api/batch", explore_url="")
    assert "Law search" not in html and "__EXPLORE" not in html


def test_retrieval_only_mode_returns_cited_passages(tfidf_index):
    # LLM_PROVIDER=none on Render: no answer is written, the passages come back with citations.
    assistant = ComplianceAssistant(llm_provider="none", index_dir=tfidf_index)
    assistant.llm = None
    answer = assistant.ask("decision based solely on automated processing, including profiling",
                           regulation="GDPR")
    assert answer.mode == "retrieval_only"
    assert "GDPR Article 22" in [s["citation"] for s in answer.sources]
    assert all(s["citation"].startswith("GDPR") for s in answer.sources)
