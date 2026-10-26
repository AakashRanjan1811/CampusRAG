"""Fact coverage, provenance, bounded batches, and real-model summary regressions."""

from dataclasses import replace
import json
import os

from langchain_core.documents import Document
import pytest
from pydantic import ValidationError

from conftest import pdf_bytes
from config import Settings
from rag.document_processor import DocumentProcessor
from rag.rag_pipeline import normalized, valid_quote
from rag.vector_store import VectorStore
from services.llm import OllamaLLM
from services.summarizer import (
    SectionAssignment, SummarySections, Summarizer, SummarizationError,
    classification_schema, source_passages,
)

LIBRARY_FACTS = [
    "The university library opens at 8:00 AM and closes at 10:00 PM.",
    "Students can borrow a maximum of 5 books at a time.",
    "The borrowing period is 14 days.",
    "The fine for late returns is Rs. 10 per day.",
]
LIBRARY_TEXT = "Library Guidelines - Fictional Test Document\n\n" + "\n\n".join(LIBRARY_FACTS)
MULTIPAGE_FACTS = [
    ["Applications open on 3 August 2027.", "An identity card must be presented at the desk."],
    ["Students may reserve a maximum of 2 instruments.", "Equipment must be returned within 7 days.",
     "The security deposit is Rs. 250 per instrument."],
    ["Late equipment returns incur a charge of Rs. 15 per day.", "Appeals must be filed by 12 September 2027."],
]


class Classifier:
    def __init__(self, section="main_points"):
        self.section, self.calls = section, []

    def generate(self, system, payload, schema):
        self.calls.append((payload, schema))
        return SummarySections(assignments=[SectionAssignment(passage_id=pid, section=self.section)
                                           for pid in payload["passages"]])


def index(vectors, pages, filename):
    vectors.ingest(pdf_bytes(pages), filename, DocumentProcessor())
    return next(d["document_id"] for d in vectors.list_documents() if d["source"] == filename)


def assert_citations(summary, vectors, doc_id, filename):
    chunks = {chunk.metadata["chunk_id"]: chunk for chunk in vectors.document_chunks(doc_id)}
    for source in summary.citations:
        assert source["filename"] == filename and source["document_id"] == doc_id
        chunk = chunks[source["chunk_id"]]
        assert source["page"] == chunk.metadata["page"]
        assert valid_quote(source["snippet"], chunk.page_content)


@pytest.mark.parametrize("text", [LIBRARY_TEXT, " ".join(LIBRARY_FACTS)])
def test_library_summary_keeps_all_facts_in_rendered_text(settings, vectors, text):
    doc_id = index(vectors, [text], "Library_Guidelines.pdf")
    summary = Summarizer(settings, vectors, Classifier()).summarize(doc_id)
    for fact in LIBRARY_FACTS:
        assert fact in normalized(summary.text)
    assert "No explicit information" not in summary.text
    assert "Main points" in summary.text
    assert "Important rules" not in summary.text  # Unused sections aren't forced.
    assert_citations(summary, vectors, doc_id, "Library_Guidelines.pdf")


def test_short_fragments_abbreviations_and_units_are_not_lost():
    text = "Policy\n\n8 AM\n\n10 PM\n\n5 books\n\n14 days\n\nRs. 10/day"
    passages = source_passages(text)
    assert normalized(" ".join(passages)) == normalized(text)
    assert all(valid_quote(passage, text) for passage in passages)


@pytest.mark.parametrize("invalid_ids", [["P1"], ["P1", "P1"], ["P1", "UNKNOWN"]])
def test_incomplete_duplicate_or_unknown_assignments_cannot_return_partial_summary(settings, invalid_ids):
    class Incomplete:
        def generate(self, system, payload, schema):
            return SummarySections(assignments=[SectionAssignment(passage_id=pid, section="overview")
                                               for pid in invalid_ids])

    summarizer = Summarizer(settings, None, Incomplete())
    with pytest.raises(SummarizationError, match="No partial summary"):
        summarizer._classify({"P1": "First important requirement.", "P2": "Second important requirement."})


def test_overview_only_output_and_invented_fact_text_fail_schema_validation():
    schema = classification_schema(2)
    with pytest.raises(ValidationError):
        schema.model_validate({"assignments": [{"passage_id": "P1", "section": "overview"}]})
    with pytest.raises(ValidationError):
        SectionAssignment.model_validate({"passage_id": "P1", "section": "rules", "text": "Invented fee of Rs. 500."})


def test_every_assignment_is_required_in_bounded_batches(settings, vectors):
    facts = [f"Requirement {i}: approval is required." for i in range(36)]
    doc_id = index(vectors, ["\n\n".join(facts[:18]), "\n\n".join(facts[18:])], "Many Rules.pdf")
    llm = Classifier("rules")
    summary = Summarizer(settings, vectors, llm).summarize(doc_id)
    for fact in facts:
        assert fact in normalized(summary.text)
    for payload, schema in llm.calls:
        assert len(payload["passages"]) <= 12
        assert len(json.dumps(payload, ensure_ascii=False).encode()) < settings.llm_context_tokens
        constraint = schema.model_json_schema()["properties"]["assignments"]
        assert constraint["minItems"] == constraint["maxItems"] == len(payload["passages"])


def test_duplicate_passages_merge_without_losing_page_citations(settings, vectors):
    fact = "Students must present an identity card at the desk."
    doc_id = index(vectors, [fact, fact], "Access.pdf")
    summary = Summarizer(settings, vectors, Classifier("rules")).summarize(doc_id)
    assert summary.text.count(fact) == 1
    assert {source["page"] for source in summary.citations} == {1, 2}
    assert_citations(summary, vectors, doc_id, "Access.pdf")


def test_multi_page_summary_preserves_dates_penalties_and_nonnumeric_requirements(settings, vectors):
    pages = ["\n\n".join(facts) for facts in MULTIPAGE_FACTS]
    doc_id = index(vectors, pages, "Research Access.pdf")
    summary = Summarizer(settings, vectors, Classifier()).summarize(doc_id)
    for facts in MULTIPAGE_FACTS:
        for fact in facts:
            assert fact in normalized(summary.text)
    assert {source["page"] for source in summary.citations} == {1, 2, 3}
    assert_citations(summary, vectors, doc_id, "Research Access.pdf")


@pytest.mark.integration
@pytest.mark.skipif(os.getenv("RUN_OLLAMA_TESTS") != "1", reason="Set RUN_OLLAMA_TESTS=1 with Ollama and the configured model available")
@pytest.mark.parametrize("filename,pages,facts", [
    ("Library_Guidelines.pdf", [LIBRARY_TEXT], LIBRARY_FACTS),
    ("Research Access.pdf", ["\n\n".join(page) for page in MULTIPAGE_FACTS],
     [fact for page in MULTIPAGE_FACTS for fact in page]),
])
def test_complete_summary_with_real_models(settings, filename, pages, facts):
    configured = replace(Settings.from_env(), chroma_dir=settings.chroma_dir, sqlite_path=settings.sqlite_path)
    llm = OllamaLLM(configured)
    ready, message = llm.status()
    if not ready:
        pytest.skip(message)
    vectors = VectorStore(configured)
    doc_id = index(vectors, pages, filename)
    summary = Summarizer(configured, vectors, llm).summarize(doc_id)
    for fact in facts:
        assert fact in normalized(summary.text)
    assert {source["page"] for source in summary.citations} == set(range(1, len(pages) + 1))
    assert "No explicit information" not in summary.text
    if filename == "Library_Guidelines.pdf":
        assert "Important rules and guidelines" in summary.text
        assert "Dates and requirements" in summary.text
    assert_citations(summary, vectors, doc_id, filename)
