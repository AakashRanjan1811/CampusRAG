"""Chroma persistence with a small commit catalog and a shared embedding model.

The catalog is written last. Interrupted/failed imports stay invisible to retrieval
and can be retried using deterministic chunk IDs without duplicating documents.
"""

from dataclasses import dataclass
import logging
from threading import RLock

import chromadb
from chromadb.config import Settings as ChromaSettings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from config import ROOT, Settings
from rag.document_processor import DocumentProcessor, ProcessedDocument, document_hash

log = logging.getLogger(__name__)


class VectorStoreError(RuntimeError):
    """An actionable storage or embedding error."""


class SentenceEmbeddings(Embeddings):
    """Minimal LangChain adapter; no model download until actual indexing/search."""

    def __init__(self, model_name: str):
        self.model_name = model_name
        self._model = None
        self._lock = RLock()

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        with self._lock:
            if self._model is None:
                from sentence_transformers import SentenceTransformer
                self._model = SentenceTransformer(
                    self.model_name, device="cpu", cache_folder=str(ROOT / ".cache" / "models"),
                )
            return self._model.encode(
                texts, normalize_embeddings=True, show_progress_bar=False,
                batch_size=32,
            ).tolist()

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]


@dataclass
class IngestionResult:
    filename: str
    duplicate: bool
    chunk_count: int
    skipped_pages: list[int]


class VectorStore:
    def __init__(self, settings: Settings, embeddings: Embeddings | None = None):
        self.settings = settings
        self._lock = RLock()
        settings.chroma_dir.mkdir(parents=True, exist_ok=True)
        self.client = chromadb.PersistentClient(
            path=str(settings.chroma_dir), settings=ChromaSettings(anonymized_telemetry=False),
        )
        self.collection = self.client.get_or_create_collection(
            "campus_chunks", metadata={
                "hnsw:space": "cosine", "embedding_model": settings.embedding_model,
            },
        )
        if (self.collection.metadata or {}).get("embedding_model") != settings.embedding_model:
            raise VectorStoreError("The index uses a different embedding model. Restore EMBEDDING_MODEL or choose a new CHROMA_DIR.")
        self.catalog = self.client.get_or_create_collection("campus_documents", embedding_function=None)
        self.store = Chroma(
            client=self.client, collection_name="campus_chunks",
            embedding_function=embeddings or SentenceEmbeddings(settings.embedding_model),
        )

    def list_documents(self) -> list[dict]:
        with self._lock:
            result = self.catalog.get(include=["metadatas"])
            documents = [dict(metadata, document_id=doc_id)
                         for doc_id, metadata in zip(result["ids"], result["metadatas"])]
            return sorted(documents, key=lambda doc: (doc["source"].lower(), doc["document_id"]))

    def contains(self, doc_id: str) -> bool:
        return bool(self.catalog.get(ids=[doc_id], include=[])["ids"])

    def ingest(self, data: bytes, filename: str, processor: DocumentProcessor) -> IngestionResult:
        with self._lock:
            doc_id = document_hash(data)
            if self.contains(doc_id):
                record = self.catalog.get(ids=[doc_id], include=["metadatas"])["metadatas"][0]
                return IngestionResult(record["source"], True, record["chunk_count"], [])
            processed = processor.process(data, filename)
            self._index(processed)
            return IngestionResult(processed.filename, False, len(processed.chunks), processed.skipped_pages)

    def _index(self, processed: ProcessedDocument):
        try:
            # Clear any orphan chunks left by an interrupted earlier attempt.
            self.collection.delete(where={"document_id": processed.document_id})
            for start in range(0, len(processed.chunks), 128):
                batch = processed.chunks[start:start + 128]
                self.store.add_documents(batch, ids=[item.metadata["chunk_id"] for item in batch])
            self.catalog.upsert(
                ids=[processed.document_id], embeddings=[[0.0]],
                metadatas=[{
                    "source": processed.filename, "page_count": processed.page_count,
                    "chunk_count": len(processed.chunks),
                    "skipped_pages": ",".join(map(str, processed.skipped_pages)),
                }],
            )
        except Exception as exc:
            log.exception("Document indexing failed: %s", processed.filename)
            try:
                self.collection.delete(where={"document_id": processed.document_id})
            except Exception:
                log.exception("Could not clean up uncommitted chunks")
            raise VectorStoreError(
                "Indexing failed. Check disk space and the embedding model download/network connection, then retry. See data/campusrag.log."
            ) from exc

    def retrieve(self, query: str, k: int | None = None) -> list[Document]:
        with self._lock:
            docs = self.list_documents()
            if not docs:
                return []
            try:
                return self.store.similarity_search(
                    query, k=min(k or self.settings.retrieval_top_k, sum(doc["chunk_count"] for doc in docs)),
                    filter={"document_id": {"$in": [doc["document_id"] for doc in docs]}},
                )
            except Exception as exc:
                log.exception("Retrieval failed")
                raise VectorStoreError("Document retrieval failed. Check the embedding model and local index; see data/campusrag.log.") from exc

    def document_chunks(self, doc_id: str) -> list[Document]:
        with self._lock:
            if not self.contains(doc_id):
                raise VectorStoreError("The selected document is no longer indexed.")
            record = self.catalog.get(ids=[doc_id], include=["metadatas"])["metadatas"][0]
            result = self.collection.get(where={"document_id": doc_id}, include=["documents", "metadatas"])
            chunks = [Document(page_content=text, metadata=metadata)
                      for text, metadata in zip(result["documents"], result["metadatas"])]
            if len(chunks) != record["chunk_count"]:
                raise VectorStoreError("This document's index is incomplete. Use a fresh CHROMA_DIR and reprocess the PDF.")
            return sorted(chunks, key=lambda chunk: chunk.metadata["chunk_index"])
