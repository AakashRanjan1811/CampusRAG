"""Exercise real Streamlit widgets with local storage and mocked model inference."""

from pathlib import Path

from streamlit.testing.v1 import AppTest

from conftest import TestEmbeddings, pdf_bytes
from rag.rag_pipeline import SelectedAnswer, SelectedClaim, Evidence, RewrittenQuestion
from services.summarizer import SummarySections, SectionAssignment


def button(app, label):
    return next(item for item in app.button if item.label == label)


def test_upload_sample_chat_switch_delete_and_summary(tmp_path, monkeypatch):
    import rag.vector_store as vector_module
    from services.llm import OllamaLLM

    monkeypatch.setenv("CHROMA_DIR", str(tmp_path / "chroma"))
    monkeypatch.setenv("SQLITE_PATH", str(tmp_path / "chat.db"))
    monkeypatch.setattr(vector_module, "SentenceEmbeddings", lambda *args: TestEmbeddings())
    monkeypatch.setattr(OllamaLLM, "status", lambda self: (False, "Ollama unavailable in UI test"))

    def generate(self, system, payload, schema):
        if schema is RewrittenQuestion:
            return RewrittenQuestion(question=payload["current_question"])
        if issubclass(schema, SummarySections):
            return SummarySections(assignments=[SectionAssignment(passage_id=pid, section="main_points")
                                               for pid in payload["passages"]])
        passage = payload["evidence"][0]["passages"][0]
        return SelectedAnswer(answerable=True, claims=[SelectedClaim(text="This is a fictional demonstration policy.", evidence_ids=[passage["evidence_id"]])])

    monkeypatch.setattr(OllamaLLM, "generate", generate)
    app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "app.py"), default_timeout=30).run()
    assert not app.exception
    assert app.chat_input[0].disabled
    assert len(app.file_uploader) == 1
    assert button(app, "Process documents").disabled
    button(app, "Load fictional sample PDFs").click().run()
    assert not app.exception
    assert app.metric[0].value == "3"
    assert not app.chat_input[0].disabled
    app.file_uploader[0].set_value([
        ("Extra A.pdf", pdf_bytes(["Additional attendance policy for demonstration." ]), "application/pdf"),
        ("Extra B.pdf", pdf_bytes(["Additional examination policy for demonstration." ]), "application/pdf"),
    ]).run()
    button(app, "Process documents").click().run()
    assert not app.exception
    assert app.metric[0].value == "5"
    button(app, "Process documents").click().run()
    assert app.metric[0].value == "5"
    assert any("Already indexed" in item.value for item in app.info)
    app.file_uploader[0].set_value(("Broken.pdf", b"not a PDF", "application/pdf")).run()
    button(app, "Process documents").click().run()
    assert not app.exception
    assert any("valid PDF" in error.value for error in app.error)
    app.file_uploader[0].set_value(None).run()
    app.chat_input[0].set_value("What is the attendance requirement?").run()
    assert not app.exception
    assert len(app.chat_message) == 2
    assert any("Supporting sources" in expander.label for expander in app.expander)
    first_session = app.selectbox[0].value
    button(app, "＋ New chat").click().run()
    assert not app.exception
    assert len(app.chat_message) == 0
    assert len(app.selectbox[0].options) == 2
    app.selectbox[0].set_value(first_session).run()
    assert len(app.chat_message) == 2
    button(app, "Generate summary").click().run()
    assert not app.exception
    assert app.radio[0].value == "Document summary"
    assert any("Document overview" in markdown.value for markdown in app.markdown)
    button(app, "Delete selected chat").click().run()
    assert not app.exception
    assert len(app.selectbox[0].options) == 1
    app.radio[0].set_value("Conversation").run()
    assert len(app.chat_message) == 0
