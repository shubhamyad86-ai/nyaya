from langchain_core.language_models.fake_chat_models import FakeListChatModel

from compliance_ai.rag.assistant import NOT_FOUND, ComplianceAssistant, check_citations
from compliance_ai.rag.ingest import chunk, load_sections
from tests.conftest import GDPR_HTML, require


def test_gdpr_structure_is_parsed():
    require(GDPR_HTML, "run scripts/download_data.py")
    sections = load_sections(GDPR_HTML)
    articles = [s for s in sections if s.metadata["kind"] == "article"]
    recitals = [s for s in sections if s.metadata["kind"] == "recital"]
    assert len(articles) == 99 and len(recitals) == 173
    art22 = next(s for s in articles if s.metadata["section"] == "Article 22")
    assert art22.metadata["title"].startswith("Automated individual decision-making")
    assert "solely on automated processing" in art22.page_content
    # Section headings of the next chapter must not leak into the previous article.
    assert "Restrictions" not in art22.page_content.splitlines()[-1]


def test_chunks_carry_citations():
    require(GDPR_HTML, "run scripts/download_data.py")
    chunks = chunk(load_sections(GDPR_HTML), chunk_size=800, chunk_overlap=100)
    assert all(c.metadata["citation"].startswith("GDPR ") for c in chunks)
    assert all(c.page_content.startswith(c.metadata["citation"]) for c in chunks)
    assert max(len(c.page_content) for c in chunks) < 1000


def test_markdown_loader(tmp_path):
    md = tmp_path / "policy.md"
    md.write_text("# Scope\nApplies to all staff.\n\n# Retention\nKeep records 5 years.")
    secs = load_sections(md)
    assert [s.metadata["section"] for s in secs] == ["Scope", "Retention"]


def test_retrieval_finds_automated_decision_article(tfidf_index):
    a = ComplianceAssistant(llm_provider="none", index_dir=tfidf_index)
    hits = a.retrieve("refusal of credit decided solely by an algorithm")
    cites = [d.metadata["citation"] for d, _ in hits]
    assert {"GDPR Article 22", "GDPR Recital 71"} & set(cites)


def test_retrieval_only_mode(tfidf_index):
    a = ComplianceAssistant(llm_provider="none", index_dir=tfidf_index)
    ans = a.ask("How fast must a personal data breach be notified?")
    assert ans.mode == "retrieval_only" and ans.sources


def test_llm_answer_and_citation_check(tfidf_index):
    fake = FakeListChatModel(responses=[
        "A decision may not rest solely on automated processing [GDPR Article 22]. "
        "Fines apply [GDPR Article 999]."
    ])
    a = ComplianceAssistant(index_dir=tfidf_index, llm=fake)
    ans = a.ask("Can a loan be refused by an automated system alone?")
    assert ans.mode == "llm"
    assert "GDPR Article 999" in ans.unsupported_citations      # invented citation is flagged
    retrieved = {s["citation"] for s in ans.sources}
    assert ("GDPR Article 22" in ans.unsupported_citations) == ("GDPR Article 22" not in retrieved)


def test_check_citations_handles_multiple_labels():
    from langchain_core.documents import Document

    docs = [Document(page_content="x", metadata={"citation": "GDPR Article 17"})]
    assert check_citations("Erase it [GDPR Article 17; GDPR Article 5].", docs) == ["GDPR Article 5"]
    assert check_citations(NOT_FOUND, docs) == []


def test_eu_ai_act_parsed_with_annexes():
    from compliance_ai.config import ROOT

    path = ROOT / "data/regulations/eu_ai_act_2024_1689.html"
    require(path, "run scripts/download_data.py")
    sections = {s.metadata["section"]: s for s in load_sections(path)}
    kinds = [s.metadata["kind"] for s in sections.values()]
    assert kinds.count("article") == 113 and kinds.count("annex") == 13 and kinds.count("recital") == 180
    assert sections["Article 86"].metadata["title"] == "Right to explanation of individual decision-making"
    assert "creditworthiness of natural persons" in sections["Annex III"].page_content
    # The last article must stop before the signatures and annexes.
    assert "ANNEX" not in sections["Article 113"].page_content


def test_regulation_filter_restricts_search(tfidf_index):
    a = ComplianceAssistant(llm_provider="none", index_dir=tfidf_index)
    hits = a.retrieve("right to an explanation of an automated decision", k=4, regulation="EU AI Act")
    assert hits and all(d.metadata["regulation"] == "EU AI Act" for d, _ in hits)
