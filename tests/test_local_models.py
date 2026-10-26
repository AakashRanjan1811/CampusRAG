"""Opt-in real-model checks. Ordinary tests need neither downloads nor Ollama."""

import os

import pytest

from config import ROOT, Settings
from database.session_store import SessionStore
from rag.document_processor import DocumentProcessor
from rag.rag_pipeline import RAGPipeline, INSUFFICIENT, normalized
from rag.vector_store import VectorStore
from services.chat_service import ChatService
from services.llm import OllamaLLM
from services.summarizer import Summarizer


def load_samples(vectors, settings):
    processor = DocumentProcessor(settings.chunk_size, settings.chunk_overlap)
    for path in sorted((ROOT / "data" / "sample_documents").glob("*.pdf")):
        vectors.ingest(path.read_bytes(), path.name, processor)


@pytest.mark.integration
@pytest.mark.skipif(os.getenv("RUN_EMBEDDING_TESTS") != "1", reason="Set RUN_EMBEDDING_TESTS=1 to download/use real MiniLM embeddings")
def test_real_minilm_index_retrieve_and_reopen(settings):
    load_samples(vectors := VectorStore(settings), settings)
    results = vectors.retrieve("What is the minimum attendance requirement?", k=3)
    assert any(chunk.metadata["source"] == "Attendance Guidelines.pdf" and "75%" in chunk.page_content for chunk in results)
    reopened = VectorStore(settings)
    assert len(reopened.list_documents()) == 3
    results = reopened.retrieve("What is the deadline for requesting re-evaluation?", k=3)
    assert any(chunk.metadata["source"] == "Examination Guidelines.pdf" and "seven calendar days" in normalized(chunk.page_content) for chunk in results)


@pytest.mark.integration
@pytest.mark.skipif(os.getenv("RUN_OLLAMA_TESTS") != "1", reason="Set RUN_OLLAMA_TESTS=1 with Ollama and the configured model available")
def test_real_ollama_conversation_and_citations(settings):
    configured = Settings.from_env()
    # Use configured model settings, but temporary storage supplied by pytest.
    from dataclasses import replace
    configured = replace(configured, chroma_dir=settings.chroma_dir, sqlite_path=settings.sqlite_path)
    llm = OllamaLLM(configured)
    ready, message = llm.status()
    if not ready:
        pytest.skip(message)
    vectors = VectorStore(configured)
    load_samples(vectors, configured)
    sessions = SessionStore(configured.sqlite_path)
    chat = ChatService(sessions, RAGPipeline(configured, vectors, llm))
    session = sessions.create_session()
    first = chat.ask(session, "What is the minimum attendance requirement?")
    assert "75" in first.text and first.citations
    assert any(c["filename"] == "Attendance Guidelines.pdf" and c["page"] == 1 and "75%" in c["snippet"] for c in first.citations)
    followup = chat.ask(session, "Does this also apply to M.Tech students?")
    assert followup.citations and "75" in followup.text
    assert "attendance" in followup.retrieval_query.lower() and "M.Tech" in followup.retrieval_query
    assert any(c["filename"] == "Attendance Guidelines.pdf" and c["page"] == 1 for c in followup.citations)
    second_followup = chat.ask(session, "What about undergraduate students?")
    assert "75" in second_followup.text and second_followup.citations
    absent = chat.ask(session, "How much does a hostel room cost?")
    assert absent.text == INSUFFICIENT and not absent.citations
    # Follow-ups and source metadata survive an actual SQLite reopen.
    saved = SessionStore(configured.sqlite_path).get_messages(session)
    assert saved[3]["citations"] == followup.citations


@pytest.mark.integration
@pytest.mark.skipif(os.getenv("RUN_OLLAMA_TESTS") != "1", reason="Set RUN_OLLAMA_TESTS=1 with Ollama and the configured model available")
def test_real_ollama_summary(settings):
    from dataclasses import replace
    configured = replace(Settings.from_env(), chroma_dir=settings.chroma_dir, sqlite_path=settings.sqlite_path)
    llm = OllamaLLM(configured)
    ready, message = llm.status()
    if not ready:
        pytest.skip(message)
    vectors = VectorStore(configured)
    load_samples(vectors, configured)
    exam = next(d for d in vectors.list_documents() if d["source"] == "Examination Guidelines.pdf")
    summary = Summarizer(configured, vectors, llm).summarize(exam["document_id"])
    assert summary.chunk_count == exam["chunk_count"] and summary.citations
