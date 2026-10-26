"""Regression coverage for amounts lost during post-retrieval passage splitting."""

from dataclasses import replace
import os
import re

from langchain_core.documents import Document
import pytest

from conftest import pdf_bytes
from config import ROOT, Settings
from database.session_store import SessionStore
from rag.document_processor import DocumentProcessor
from rag.rag_pipeline import (
    RAGPipeline, RewrittenQuestion, SelectedAnswer, SelectedClaim,
    answer_passages, evidence_context, normalized, valid_quote,
)
from rag.vector_store import VectorStore
from services.chat_service import ChatService
from services.llm import OllamaLLM

FILENAME = "Library_Guidelines.pdf"
FINE_RULE = "The fine for late returns is Rs. 10 per day."
LIBRARY_TEXT = (
    "Library Guidelines - Fictional Test Document\n\n"
    "The university library opens at 8:00 AM and closes at 10:00 PM.\n\n"
    "Students can borrow a maximum of 5 books at a time.\n\n"
    "The borrowing period is 14 days.\n\n" + FINE_RULE
)
HISTORY = [
    {"role": "user", "content": "How many books can students borrow from the library?"},
    {"role": "assistant", "content": "Students can borrow a maximum of 5 books at a time."},
    {"role": "user", "content": "And how long can they keep them?"},
    {"role": "assistant", "content": "The borrowing period is 14 days."},
]
FAILED_QUESTIONS = [
    "What happens if they return them late?",
    "What is the fine for returning library books late?",
]


@pytest.mark.parametrize("paragraph", [
    FINE_RULE,
    "Overdue charges are Rs. 42 per week.",
    "The charge is USD. 7 daily.",
    "The fee is Rs. 12.50 per day.",
    "Contact Dr. Rao at 5 p.m. for details.",
    "Return it soon. Pay Rs. 10.",
])
def test_passages_preserve_abbreviations_amounts_and_short_tails(paragraph):
    chunk = Document(page_content=paragraph, metadata={"source": FILENAME})
    context, sources = evidence_context([chunk], 1500)
    evidence, passages = answer_passages(context, sources, 1500)
    assert [item["text"] for record in evidence for item in record["passages"]] == [paragraph]
    assert passages["E1"] == ("S1", paragraph)
    assert valid_quote(paragraph, chunk.page_content)


class EvidenceOnlyLLM:
    """Abstain if the amount was lost, just as the real model did before the fix."""

    def __init__(self):
        self.calls = []

    def generate(self, system, payload, schema):
        self.calls.append((payload, schema))
        if schema is RewrittenQuestion:
            return RewrittenQuestion(question="What happens if students return library books late?")
        assert list(payload)[-1] == "question"
        passages = [item for record in payload["evidence"] for item in record["passages"]]
        match = next((item for item in passages if FINE_RULE in item["text"]), None)
        if match is None:
            return SelectedAnswer(answerable=False)
        return SelectedAnswer(answerable=True, claims=[SelectedClaim(
            evidence_ids=[match["evidence_id"]], text="Late returns incur a fine of Rs. 10 per day.",
        )])


@pytest.mark.parametrize("question", FAILED_QUESTIONS)
def test_late_return_questions_keep_the_amount_and_citation(settings, vectors, question):
    vectors.ingest(pdf_bytes([LIBRARY_TEXT]), FILENAME, DocumentProcessor())
    llm = EvidenceOnlyLLM()
    result = RAGPipeline(settings, vectors, llm).answer(question, HISTORY)
    assert "Rs. 10 per day" in result.text
    assert len(result.citations) == 1
    source = result.citations[0]
    assert source["filename"] == FILENAME and source["page"] == 1
    assert source["snippet"] == FINE_RULE
    stored = vectors.document_chunks(source["document_id"])
    assert any(chunk.metadata["chunk_id"] == source["chunk_id"] and valid_quote(source["snippet"], chunk.page_content)
               for chunk in stored)
    if question == FAILED_QUESTIONS[0]:
        assert "library books" in result.retrieval_query
        assert llm.calls[0][1] is RewrittenQuestion
    else:
        assert result.retrieval_query == question
        assert all(schema is not RewrittenQuestion for _, schema in llm.calls)


def assert_fine_answer(answer):
    assert re.search(r"\b10\b", answer.text)
    assert re.search(r"\b(per|each|a) day\b|\bdaily\b", answer.text, re.IGNORECASE)
    assert any(source["filename"] == FILENAME and source["page"] == 1
               and FINE_RULE in normalized(source["snippet"]) for source in answer.citations)


@pytest.mark.integration
@pytest.mark.skipif(os.getenv("RUN_OLLAMA_TESTS") != "1", reason="Set RUN_OLLAMA_TESTS=1 with Ollama and the configured model available")
def test_complete_library_conversation_with_real_models(settings):
    configured = replace(Settings.from_env(), chroma_dir=settings.chroma_dir, sqlite_path=settings.sqlite_path)
    llm = OllamaLLM(configured)
    ready, message = llm.status()
    if not ready:
        pytest.skip(message)
    vectors = VectorStore(configured)
    processor = DocumentProcessor(configured.chunk_size, configured.chunk_overlap)
    for path in sorted((ROOT / "data/sample_documents").glob("*.pdf")):
        vectors.ingest(path.read_bytes(), path.name, processor)
    vectors.ingest(pdf_bytes([LIBRARY_TEXT]), FILENAME, processor)
    assert vectors.retrieve(FAILED_QUESTIONS[1])[0].metadata["source"] == FILENAME
    sessions = SessionStore(configured.sqlite_path)
    session, independent = sessions.create_session(), sessions.create_session()
    pipeline = RAGPipeline(configured, vectors, llm)
    chat = ChatService(sessions, pipeline, configured.memory_messages)

    books = chat.ask(session, HISTORY[0]["content"])
    assert re.search(r"\b5\b", books.text) and "book" in books.text.lower()
    period = chat.ask(session, HISTORY[2]["content"])
    assert re.search(r"\b14\b", period.text) and "day" in period.text.lower()
    for question in FAILED_QUESTIONS:
        answer = chat.ask(session, question)
        assert_fine_answer(answer)
        if question == FAILED_QUESTIONS[0]:
            assert "library" in answer.retrieval_query.lower() and "late" in answer.retrieval_query.lower()

    # The explicit question must also work without any prior conversation.
    assert_fine_answer(pipeline.answer(FAILED_QUESTIONS[1]))
    reopened = SessionStore(configured.sqlite_path)
    messages = reopened.get_messages(session)
    assert len(messages) == 8
    assert FINE_RULE in messages[5]["citations"][0]["snippet"]
    assert FINE_RULE in messages[7]["citations"][0]["snippet"]
    assert reopened.get_messages(independent) == []
