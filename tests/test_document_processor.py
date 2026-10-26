import pymupdf
import pytest

from conftest import pdf_bytes
from rag.document_processor import DocumentError, DocumentProcessor, document_hash


def test_pages_chunks_and_metadata():
    data = pdf_bytes(["Attendance is 75%. " * 70, "Examination re-evaluation deadline is seven days."])
    processed = DocumentProcessor(200, 30).process(data, "C:\\uploads\\policy.pdf")
    assert processed.filename == "policy.pdf"
    assert processed.page_count == 2
    assert len(processed.chunks) > 2
    assert {chunk.metadata["page"] for chunk in processed.chunks} == {1, 2}
    assert all(len(chunk.page_content) <= 200 for chunk in processed.chunks)
    assert all(chunk.metadata["document_id"] == document_hash(data) for chunk in processed.chunks)
    assert len({chunk.metadata["chunk_id"] for chunk in processed.chunks}) == len(processed.chunks)
    assert processed.chunks[0].page_content[-20:] in processed.chunks[1].page_content


@pytest.mark.parametrize("data,name,message", [
    (b"", "empty.pdf", "empty"), (b"not a pdf", "bad.pdf", "valid PDF"),
    (b"%PDF-corrupted", "bad.pdf", "corrupted"), (b"x", "notes.txt", "upload a PDF"),
])
def test_invalid_pdf(data, name, message):
    with pytest.raises(DocumentError, match=message):
        DocumentProcessor().process(data, name)


def test_scanned_blank_and_partial_pages():
    with pytest.raises(DocumentError, match="scanned or blank"):
        DocumentProcessor().process(pdf_bytes([""]), "scan.pdf")
    doc = DocumentProcessor().process(pdf_bytes(["", "Text on page two."]), "partial.pdf")
    assert doc.skipped_pages == [1]
    assert doc.chunks[0].metadata["page"] == 2
    assert doc.chunks[0].metadata["skipped_pages"] == "1"


def test_encrypted_pdf():
    with pymupdf.open(stream=pdf_bytes(["Secret policy"]), filetype="pdf") as pdf:
        data = pdf.tobytes(encryption=pymupdf.PDF_ENCRYPT_AES_256, owner_pw="owner", user_pw="reader")
    with pytest.raises(DocumentError, match="encrypted"):
        DocumentProcessor().process(data, "encrypted.pdf")


def test_duplicate_detection_persistence_and_no_reembedding(vectors, settings, embeddings):
    from rag.vector_store import VectorStore
    data = pdf_bytes(["Attendance is at least 75% in every course.", "Postgraduate attendance is also 75%."])
    first = vectors.ingest(data, "Attendance.pdf", DocumentProcessor())
    calls = embeddings.calls
    restarted = VectorStore(settings, embeddings)
    second = restarted.ingest(data, "Renamed.pdf", DocumentProcessor())
    assert not first.duplicate and second.duplicate
    assert embeddings.calls == calls
    assert len(restarted.list_documents()) == 1
    chunks = restarted.document_chunks(document_hash(data))
    assert len(chunks) == first.chunk_count
    assert chunks[0].metadata["source"] == "Attendance.pdf"
    assert {chunk.metadata["page"] for chunk in chunks} == {1, 2}


def test_index_is_readable_from_a_fresh_python_process(vectors, settings):
    import subprocess
    import sys
    from config import ROOT
    vectors.ingest(pdf_bytes(["Attendance is 75% in each course."]), "Policy.pdf", DocumentProcessor())
    script = """
import sys
from pathlib import Path
from dataclasses import replace
sys.path.insert(0, 'tests')
from conftest import TestEmbeddings
from config import Settings
from rag.vector_store import VectorStore
vectors = VectorStore(replace(Settings(), chroma_dir=Path(sys.argv[1])), TestEmbeddings())
assert len(vectors.list_documents()) == 1
assert vectors.retrieve('attendance', 1)[0].metadata['source'] == 'Policy.pdf'
"""
    result = subprocess.run([sys.executable, "-c", script, str(settings.chroma_dir)],
                            cwd=ROOT, capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr


def test_failed_index_is_hidden_and_retryable(vectors, monkeypatch):
    from rag.vector_store import VectorStoreError
    data = pdf_bytes(["Attendance policy requires 75%."])
    original = vectors.store.add_documents

    def fail_after_write(*args, **kwargs):
        original(*args, **kwargs)
        raise RuntimeError("Simulated disk failure")

    monkeypatch.setattr(vectors.store, "add_documents", fail_after_write)
    with pytest.raises(VectorStoreError, match="Indexing failed"):
        vectors.ingest(data, "Policy.pdf", DocumentProcessor())
    assert vectors.list_documents() == []
    assert vectors.retrieve("attendance") == []
    monkeypatch.setattr(vectors.store, "add_documents", original)
    vectors.ingest(data, "Policy.pdf", DocumentProcessor())
    assert len(vectors.list_documents()) == 1


def test_model_change_requires_new_index(vectors, settings, embeddings):
    from dataclasses import replace
    from rag.vector_store import VectorStoreError, VectorStore
    with pytest.raises(VectorStoreError, match="different embedding"):
        VectorStore(replace(settings, embedding_model="different-model"), embeddings)
