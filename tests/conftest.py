from dataclasses import replace
import math
import re
from uuid import uuid4

import pymupdf
import pytest
from langchain_core.embeddings import Embeddings

from config import Settings
from rag.vector_store import VectorStore


def pdf_bytes(pages):
    with pymupdf.open() as pdf:
        for text in pages:
            page = pdf.new_page()
            page.insert_textbox(pymupdf.Rect(40, 40, 550, 800), text, fontsize=10)
        return pdf.tobytes()


class TestEmbeddings(Embeddings):
    __test__ = False
    words = ["attendance", "75", "examination", "evaluation", "deadline", "credits", "registration", "hostel"]

    def __init__(self):
        self.calls = 0

    def embed_documents(self, texts):
        self.calls += len(texts)
        vectors = []
        for text in texts:
            words = re.findall(r"\w+", text.lower())
            vector = [words.count(word) for word in self.words] + [0.01]
            norm = math.sqrt(sum(value * value for value in vector))
            vectors.append([value / norm for value in vector])
        return vectors

    def embed_query(self, text):
        return self.embed_documents([text])[0]


@pytest.fixture
def settings(tmp_path):
    # Native Chroma caches must not reuse a previous run's removed directory.
    return replace(Settings(), chroma_dir=tmp_path / f"chroma-{uuid4().hex}", sqlite_path=tmp_path / "chat.db")


@pytest.fixture
def embeddings():
    return TestEmbeddings()


@pytest.fixture
def vectors(settings, embeddings):
    store = VectorStore(settings, embeddings)
    try:
        yield store
    finally:
        # Release native Chroma workers/file handles before the next temporary index.
        store.client.close()
