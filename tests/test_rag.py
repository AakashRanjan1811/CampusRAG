from dataclasses import replace
import json
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from pydantic import ValidationError

from conftest import pdf_bytes
from database.session_store import SessionStore
from rag.document_processor import DocumentProcessor
from rag.rag_pipeline import (
    INSUFFICIENT, Answer, Claim, Evidence, GroundedAnswer, RAGPipeline,
    RewrittenQuestion, SelectedAnswer, SelectedClaim, answer_passages, context_budget, evidence_context, valid_quote,
)
from services.chat_service import ChatService
from services.llm import LLMError, OllamaLLM
from services.summarizer import (
    SummarizationError, Summarizer, SectionAssignment, SummarySections,
)


class QueueLLM:
    def __init__(self, *responses):
        self.responses = list(responses)
        self.calls = []

    def generate(self, system, payload, schema):
        self.calls.append((system, payload, schema))
        return self.responses.pop(0)


def grounded(evidence_id="E1"):
    return SelectedAnswer(answerable=True, claims=[SelectedClaim(text="The minimum attendance is 75%.", evidence_ids=[evidence_id])])


def attendance_index(vectors):
    vectors.ingest(pdf_bytes(["Students need 75% attendance in every course."]), "Attendance.pdf", DocumentProcessor())


def test_real_chroma_retrieval_order_and_top_k(vectors):
    attendance_index(vectors)
    vectors.ingest(pdf_bytes(["Registration requires 18 credits."]), "Academic.pdf", DocumentProcessor())
    results = vectors.retrieve("attendance 75", k=1)
    assert len(results) == 1
    assert results[0].metadata["source"] == "Attendance.pdf"
    assert results[0].metadata["page"] == 1


def test_answers_have_actual_source_quotes(settings, vectors):
    attendance_index(vectors)
    answer = RAGPipeline(settings, vectors, QueueLLM(grounded())).answer("What is the attendance requirement?")
    assert "75%" in answer.text and "[1]" in answer.text
    assert answer.citations[0]["filename"] == "Attendance.pdf"
    assert answer.citations[0]["page"] == 1
    assert answer.citations[0]["snippet"] == "Students need 75% attendance in every course."


@pytest.mark.parametrize("response", [
    SelectedAnswer(answerable=False),
    SelectedAnswer(answerable=False, claims=grounded().claims),
    grounded(evidence_id="INVENTED"),
])
def test_unsupported_or_fabricated_evidence_has_no_citations(settings, vectors, response):
    attendance_index(vectors)
    answer = RAGPipeline(settings, vectors, QueueLLM(response, response)).answer("How much does a hostel room cost?")
    assert answer.text == INSUFFICIENT
    assert answer.citations == []


def test_invalid_reference_is_corrected_once_and_revalidated(settings, vectors):
    attendance_index(vectors)
    llm = QueueLLM(grounded(evidence_id="WRONG"), grounded())
    answer = RAGPipeline(settings, vectors, llm).answer("What is the attendance requirement?")
    assert "75%" in answer.text and answer.citations[0]["filename"] == "Attendance.pdf"
    assert len(llm.calls) == 2
    feedback = llm.calls[1][1]["validation_feedback"]
    assert feedback == {"evidence_id": "WRONG", "reason": "unknown_evidence_id"}
    assert llm.calls[0][1]["evidence"] == llm.calls[1][1]["evidence"]


def test_repeated_invalid_passage_ids_are_never_accepted(settings, vectors):
    attendance_index(vectors)
    invalid = grounded(evidence_id="INVENTED")
    llm = QueueLLM(invalid, invalid)
    answer = RAGPipeline(settings, vectors, llm).answer("What is the attendance requirement?")
    assert answer.text == INSUFFICIENT and not answer.citations
    assert len(llm.calls) == 2


def test_exact_quote_safeguard_and_model_quote_injection():
    text = "Students need 75% attendance in every course."
    assert valid_quote(text, text)
    assert not valid_quote(text.replace("75%", "20%"), text)
    assert not valid_quote(text.lower(), text)
    with pytest.raises(ValidationError):
        SelectedAnswer.model_validate({"answerable": True, "claims": [
            {"text": "Invented", "evidence_ids": ["E1"], "quote": "Students need 20%."},
        ]})


def test_passage_numbering_preserves_source_and_context_budget():
    from langchain_core.documents import Document
    chunks = [Document(page_content="First policy sentence.\n\nSecond policy sentence.", metadata={"source": "First.pdf"}),
              Document(page_content="Another policy sentence.", metadata={"source": "Second.pdf"})]
    context, sources = evidence_context(chunks, 1000)
    evidence, passages = answer_passages(context, sources, budget=230)
    assert len(json.dumps(evidence, ensure_ascii=False).encode("utf-8")) <= 230
    assert passages
    for source_id, quote in passages.values():
        assert valid_quote(quote, sources[source_id].page_content)
    assert len(passages) == len({item["evidence_id"] for source in evidence for item in source["passages"]})


def test_empty_retrieval_does_not_call_llm(settings):
    vectors = Mock()
    vectors.list_documents.return_value = [{"source": "x"}]
    vectors.retrieve.return_value = []
    llm = QueueLLM()
    answer = RAGPipeline(settings, vectors, llm).answer("Attendance?")
    assert answer.text == INSUFFICIENT
    assert not llm.calls
    vectors.list_documents.return_value = []
    assert "Upload" in RAGPipeline(settings, vectors, llm).answer("Attendance?").text


def test_followup_uses_bounded_history_only_for_retrieval(settings, vectors):
    attendance_index(vectors)
    llm = QueueLLM(RewrittenQuestion(question="Does the 75% attendance requirement apply to M.Tech students?"), grounded())
    history = [{"role": "user" if i % 2 == 0 else "assistant", "content": "older policy claim " * 200} for i in range(30)]
    answer = RAGPipeline(settings, vectors, llm).answer("Does this apply to M.Tech students?", history)
    rewrite_payload = llm.calls[0][1]
    answer_payload = llm.calls[1][1]
    assert len(rewrite_payload["history"]) <= settings.memory_messages
    assert sum(len(m["content"].encode("utf-8")) for m in rewrite_payload["history"]) <= 3500
    assert "history" not in answer_payload
    assert "older policy claim" not in json.dumps(answer_payload)
    assert "M.Tech" in answer.retrieval_query


def test_new_topic_is_not_overwritten_by_conversation_history(settings):
    llm = QueueLLM()
    history = [{"role": "user", "content": "What is the minimum attendance requirement?"},
               {"role": "assistant", "content": "The attendance requirement is 75%."}]
    pipeline = RAGPipeline(settings, Mock(), llm)
    question = "How much does a hostel room cost?"
    assert pipeline.reformulate(question, history) == question
    assert not llm.calls


def test_chat_service_persists_citations_and_does_not_mix_sessions(settings):
    sessions = SessionStore(settings.sqlite_path)
    a, b = sessions.create_session(), sessions.create_session()
    pipeline = Mock()
    pipeline.answer.return_value = Answer("75% [1]", [{"filename": "policy.pdf", "page": 1}])
    chat = ChatService(sessions, pipeline, memory_messages=2)
    chat.ask(a, "Attendance?")
    chat.ask(b, "Different question")
    assert pipeline.answer.call_args.args[1] == []
    chat.ask(a, "Does this apply?")
    assert [m["content"] for m in pipeline.answer.call_args.args[1]] == ["Attendance?", "75% [1]"]
    assert sessions.get_messages(a)[1]["citations"][0]["filename"] == "policy.pdf"


def test_failed_inference_does_not_save_half_turn(settings):
    sessions = SessionStore(settings.sqlite_path)
    session = sessions.create_session()
    pipeline = Mock()
    pipeline.answer.side_effect = LLMError("Ollama offline")
    with pytest.raises(LLMError):
        ChatService(sessions, pipeline).ask(session, "Question")
    assert sessions.get_messages(session) == []


def test_ollama_adapter_validates_json_and_handles_failure(settings, monkeypatch):
    import services.llm as module
    model = Mock()
    factory = Mock(return_value=model)
    monkeypatch.setattr(module, "ChatOllama", factory)
    model.invoke.return_value = SimpleNamespace(content=grounded().model_dump_json())
    assert OllamaLLM(settings).generate("System", {}, SelectedAnswer).answerable
    assert factory.call_args.kwargs["num_ctx"] == settings.llm_context_tokens
    assert factory.call_args.kwargs["format"]["type"] == "object"
    sent_messages = model.invoke.call_args.args[0]
    assert sent_messages[0].content.startswith("System\n")
    # Same schema is visible to the model AND enforced by Ollama's decoder.
    assert json.loads(sent_messages[0].content.split("response schema:\n", 1)[1]) == factory.call_args.kwargs["format"]
    assert list(factory.call_args.kwargs["format"]["properties"]) == ["claims", "answerable"]
    assert list(factory.call_args.kwargs["format"]["$defs"]["SelectedClaim"]["properties"]) == ["evidence_ids", "text"]
    model.invoke.return_value = SimpleNamespace(content="not json")
    with pytest.raises(LLMError, match="incomplete response"):
        OllamaLLM(settings).generate("System", {}, SelectedAnswer)
    model.invoke.side_effect = TimeoutError("timed out")
    with pytest.raises(LLMError, match="Ollama is running"):
        OllamaLLM(settings).generate("System", {}, SelectedAnswer)


class SummaryLLM:
    def __init__(self):
        self.payloads = []

    def generate(self, system, payload, schema):
        self.payloads.append(payload)
        return SummarySections(assignments=[SectionAssignment(passage_id=pid, section="rules")
                                           for pid in payload["passages"]])


def test_summary_covers_every_page_with_verified_quotes(settings, vectors):
    vectors.ingest(pdf_bytes(["First page: attendance is at least 75%.", "Last page: examination requests close in seven days."]), "Policy.pdf", DocumentProcessor())
    doc_id = vectors.list_documents()[0]["document_id"]
    llm, seen = SummaryLLM(), []
    summary = Summarizer(settings, vectors, llm).summarize(doc_id, lambda done, total: seen.append((done, total)))
    assert summary.chunk_count == 2
    assert {c["page"] for c in summary.citations} == {1, 2}
    assert "Last page" in summary.text
    assert seen[-1] == (2, 2)
    assert "Document overview" in summary.text and "Important rules" in summary.text
    assert "Dates and requirements" not in summary.text
    assert "No explicit information" not in summary.text


def test_summary_limit_is_explicit_and_not_a_partial_summary(settings, vectors):
    vectors.ingest(pdf_bytes(["First page rule needs attendance.", "Second page rule needs registration."]), "Policy.pdf", DocumentProcessor())
    llm = QueueLLM()
    with pytest.raises(SummarizationError, match="No partial summary"):
        Summarizer(replace(settings, max_summary_chunks=1), vectors, llm).summarize(vectors.list_documents()[0]["document_id"])
    assert not llm.calls


def test_summary_retains_distinct_facts_beyond_sixteen_notes(settings, vectors):
    vectors.ingest(pdf_bytes([f"Page {i}: attendance requirement for course {i} is 75%." for i in range(22)]), "Long.pdf", DocumentProcessor())
    llm = SummaryLLM()
    summary = Summarizer(settings, vectors, llm).summarize(vectors.list_documents()[0]["document_id"])
    assert len(llm.payloads) == 22
    assert all(len(json.dumps(payload).encode("utf-8")) < context_budget(settings) for payload in llm.payloads)
    assert {source["page"] for source in summary.citations} == set(range(1, 23))
    assert all(f"course {i} is 75%" in summary.text for i in range(22))
    assert summary.chunk_count == 22


def test_unknown_summary_reference_fails(settings, vectors):
    attendance_index(vectors)
    invalid = SummarySections(assignments=[SectionAssignment(passage_id="INVENTED", section="rules")])
    llm = QueueLLM(invalid, invalid)
    with pytest.raises(SummarizationError, match="did not cover every source passage"):
        Summarizer(settings, vectors, llm).summarize(vectors.list_documents()[0]["document_id"])


def test_context_budget_skips_oversized_passages_and_bounds_rewrite(settings):
    from langchain_core.documents import Document
    large = Document(page_content="x" * 10000, metadata={"source": "Large.pdf"})
    small = Document(page_content="A useful short passage.", metadata={"source": "Small.pdf"})
    context, sources = evidence_context([large, small], 200)
    assert len(context) == 1 and sources["S1"] == small
    llm = QueueLLM(RewrittenQuestion(question="漢" * 1000))
    with pytest.raises(ValueError, match="standalone query is too long"):
        RAGPipeline(settings, Mock(), llm).reformulate("Does this apply?", [{"role": "user", "content": "Attendance?"}])


def test_pdf_line_wraps_are_normalized_without_relaxing_quote_validation(settings, vectors):
    from config import ROOT
    path = ROOT / "data/sample_documents/Attendance Guidelines.pdf"
    vectors.ingest(path.read_bytes(), path.name, DocumentProcessor())
    quote = "Students must maintain at least 75% attendance in each course to be eligible for the end-semester examination."

    class AttendanceLLM:
        def generate(self, system, payload, schema):
            passages = [passage for source in payload["evidence"] for passage in source["passages"]]
            assert all("\n" not in passage["text"] for passage in passages)
            supporting = next(passage for passage in passages if quote in passage["text"])
            return SelectedAnswer(answerable=True, claims=[SelectedClaim(
                text="The minimum attendance requirement is 75% in each course.",
                evidence_ids=[supporting["evidence_id"]],
            )])

    answer = RAGPipeline(settings, vectors, AttendanceLLM()).answer("What is the minimum attendance requirement?")
    assert "75%" in answer.text
    assert answer.citations[0]["filename"] == path.name
    assert answer.citations[0]["page"] == 1
    assert quote in answer.citations[0]["snippet"]
