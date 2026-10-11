"""Turn regulation files into section-aware LangChain Documents.

Legal text is cited by article, so chunks follow the law's own structure:
one Document per article (or recital), split further only when an article is
long. Every chunk carries its citation (e.g. "GDPR Article 22") in metadata
and in a short header, which helps both retrieval and the answer's citations.

Supported inputs
- EUR-Lex HTML (articles marked with class="ti-art"/"sti-art")
- PDF (e.g. the EU AI Act from EUR-Lex): articles found by "Article N" lines
- Plain text / Markdown: split on headings
"""
from __future__ import annotations

import re
import warnings
from pathlib import Path

from bs4 import BeautifulSoup, XMLParsedAsHTMLWarning
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)

# File-name hints -> short regulation name used in citations.
KNOWN_SOURCES = {
    "gdpr": "GDPR",
    "ai_act": "EU AI Act",
}


def regulation_name(path: Path) -> str:
    stem = path.stem.lower()
    for key, name in KNOWN_SOURCES.items():
        if key in stem:
            return name
    return path.stem.replace("_", " ")


def _clean(text: str) -> str:
    text = text.replace("\xa0", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n\s*\n+", "\n\n", text)
    return text.strip()


def _cls(el, *names: str) -> bool:
    """True if the element has one of the EUR-Lex classes, in the old ("ti-art") or the
    2024 Official Journal format ("oj-ti-art")."""
    classes = el.get("class", []) if el is not None and el.name else []
    return any(c in classes or f"oj-{c}" in classes for c in names)


def _cls_prefix(el, *prefixes: str) -> bool:
    classes = el.get("class", []) if el is not None and el.name else []
    return any(c.removeprefix("oj-").startswith(prefixes) for c in classes)


def _is_annex_heading(el) -> bool:
    return el.name == "p" and _cls(el, "doc-ti") and _norm(el.get_text()).startswith("ANNEX")


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", text.replace("\xa0", " ")).strip()


def _sections_from_eurlex_html(path: Path, reg: str) -> list[Document]:
    soup = BeautifulSoup(path.read_text(encoding="utf-8"), "lxml")
    docs: list[Document] = []
    articles = [p for p in soup.find_all("p") if _cls(p, "ti-art")]

    # Recitals: two-cell rows "(n)" | text, before Article 1. (Numbered definitions in the
    # articles use the same layout, so only rows preceding Article 1 count.)
    rows = reversed(articles[0].find_all_previous("tr")) if articles else soup.find_all("tr")
    for row in rows:
        cells = row.find_all("td", recursive=False)
        if len(cells) == 2:
            m = re.fullmatch(r"\((\d+)\)", cells[0].get_text(strip=True))
            if m:
                docs.append(Document(
                    page_content=_clean(cells[1].get_text(" ")),
                    metadata={"regulation": reg, "section": f"Recital {m.group(1)}",
                              "title": "", "kind": "recital", "source": path.name},
                ))

    # Articles: everything from one article heading to the next (or to the signatures/annexes).
    for head in articles:
        number = _norm(head.get_text())
        sub = head.find_next("p")
        title = _norm(sub.get_text()) if _cls(sub, "sti-art") else ""
        parts = []
        for el in head.find_all_next("p"):
            if _cls(el, "ti-art", "final", "signatory") or _is_annex_heading(el):
                break
            if el is sub and title:
                continue
            # Chapter / section headings belong to the next article, not this one.
            if _cls_prefix(el, "ti-section", "ti-grseq") or any(
                    _cls_prefix(parent, "ti-section", "ti-grseq") for parent in el.find_parents("p")):
                continue
            if not el.find("p"):
                parts.append(el.get_text(" "))
        docs.append(Document(
            page_content=_clean("\n".join(parts)),
            metadata={"regulation": reg, "section": number, "title": title,
                      "kind": "article", "source": path.name},
        ))

    # Annexes (e.g. EU AI Act Annex III, the list of high-risk uses such as credit scoring).
    annex_heads = [p for p in soup.find_all("p") if _is_annex_heading(p)]
    for head in annex_heads:
        number = "Annex " + _norm(head.get_text()).split(" ", 1)[1]
        sub = head.find_next("p")
        title = _norm(sub.get_text()) if _cls(sub, "doc-ti") else ""
        parts = []
        for el in head.find_all_next("p"):
            if _is_annex_heading(el):
                break
            if el is sub and title:
                continue
            if not el.find("p"):
                parts.append(el.get_text(" "))
        docs.append(Document(
            page_content=_clean("\n".join(parts)),
            metadata={"regulation": reg, "section": number, "title": title,
                      "kind": "annex", "source": path.name},
        ))
    return docs


def _sections_from_pdf(path: Path, reg: str) -> list[Document]:
    from pypdf import PdfReader

    text = "\n".join(page.extract_text() or "" for page in PdfReader(str(path)).pages)
    pieces = re.split(r"\n(Article \d+)\s*\n", text)
    docs = []
    for i in range(1, len(pieces) - 1, 2):
        body = pieces[i + 1].strip()
        title, _, rest = body.partition("\n")
        docs.append(Document(
            page_content=_clean(rest),
            metadata={"regulation": reg, "section": pieces[i], "title": title.strip(),
                      "kind": "article", "source": path.name},
        ))
    return docs


def _sections_from_text(path: Path, reg: str) -> list[Document]:
    text = path.read_text(encoding="utf-8")
    blocks = re.split(r"\n(?=#{1,3} )", "\n" + text)
    docs = []
    for block in blocks:
        block = block.strip()
        if not block:
            continue
        heading, _, body = block.partition("\n")
        docs.append(Document(
            page_content=_clean(body or heading),
            metadata={"regulation": reg, "section": heading.lstrip("# ").strip(),
                      "title": "", "kind": "section", "source": path.name},
        ))
    return docs


def load_sections(path: Path) -> list[Document]:
    reg = regulation_name(path)
    suffix = path.suffix.lower()
    if suffix in {".html", ".htm", ".xhtml"}:
        return _sections_from_eurlex_html(path, reg)
    if suffix == ".pdf":
        return _sections_from_pdf(path, reg)
    if suffix in {".txt", ".md"}:
        return _sections_from_text(path, reg)
    raise ValueError(f"unsupported file type: {path.name}")


def citation(meta: dict) -> str:
    return f"{meta['regulation']} {meta['section']}"


def chunk(sections: list[Document], chunk_size: int, chunk_overlap: int) -> list[Document]:
    """Split long sections; prefix each chunk with its citation and title."""
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap,
                                              separators=["\n\n", "\n", ". ", " "])
    chunks = []
    for sec in sections:
        if not sec.page_content:
            continue
        header = citation(sec.metadata) + (f" - {sec.metadata['title']}" if sec.metadata["title"] else "")
        for j, piece in enumerate(splitter.split_text(sec.page_content)):
            chunks.append(Document(
                page_content=f"{header}\n{piece}",
                metadata={**sec.metadata, "chunk": j, "citation": citation(sec.metadata)},
            ))
    return chunks


def load_corpus(corpus_dir: Path, chunk_size: int, chunk_overlap: int) -> list[Document]:
    files = sorted(p for p in corpus_dir.iterdir()
                   if p.suffix.lower() in {".html", ".htm", ".pdf", ".txt", ".md"}
                   and not p.name.upper().startswith("README"))
    if not files:
        raise FileNotFoundError(f"no regulation files in {corpus_dir} - run scripts/download_data.py")
    sections = [doc for f in files for doc in load_sections(f)]
    return chunk(sections, chunk_size, chunk_overlap)
