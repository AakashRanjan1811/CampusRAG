"""Extract one PDF page at a time and keep page boundaries in every chunk."""

from dataclasses import dataclass
from hashlib import sha256
from pathlib import PureWindowsPath

import pymupdf
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


class DocumentError(ValueError):
    """A PDF could not be safely processed."""


@dataclass
class ProcessedDocument:
    document_id: str
    filename: str
    page_count: int
    chunks: list[Document]
    skipped_pages: list[int]


def document_hash(data: bytes) -> str:
    return sha256(data).hexdigest()


class DocumentProcessor:
    def __init__(self, chunk_size=800, chunk_overlap=120):
        if chunk_size <= 0 or not 0 <= chunk_overlap < chunk_size:
            raise ValueError("Chunk overlap must be smaller than the positive chunk size.")
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
        )

    def process(self, data: bytes, filename: str) -> ProcessedDocument:
        filename = PureWindowsPath(filename).name
        if not filename.lower().endswith(".pdf"):
            raise DocumentError("Please upload a PDF file.")
        if not data:
            raise DocumentError("The PDF file is empty.")
        if len(data) > 20 * 1024 * 1024:
            raise DocumentError("This PDF exceeds the 20 MB upload limit.")
        if b"%PDF-" not in data[:1024]:
            raise DocumentError("The file is not a valid PDF.")
        doc_id = document_hash(data)
        chunks, skipped = [], []
        try:
            with pymupdf.open(stream=data, filetype="pdf") as pdf:
                if pdf.needs_pass:
                    raise DocumentError("This PDF is encrypted. Upload an unencrypted copy.")
                if len(pdf) > 500:
                    raise DocumentError("This PDF exceeds the 500-page processing limit.")
                page_count = len(pdf)
                for number, page in enumerate(pdf, start=1):
                    text = page.get_text("text", sort=True).replace("\x00", "").strip()
                    if not text:
                        skipped.append(number)
                        continue
                    for part in self.splitter.split_text(text):
                        index = len(chunks)
                        chunks.append(Document(page_content=part, metadata={
                            "source": filename, "page": number, "document_id": doc_id,
                            "chunk_id": f"{doc_id}:{index:06d}", "chunk_index": index,
                            "page_count": page_count,
                        }))
                        if len(chunks) > 10000:
                            raise DocumentError("This PDF produces too many chunks; split it into smaller PDFs.")
        except DocumentError:
            raise
        except Exception as exc:
            raise DocumentError("The PDF is corrupted or could not be read. Try exporting it again.") from exc
        if not chunks:
            raise DocumentError("No extractable text was found. This PDF may be scanned or blank; OCR is not included.")
        for chunk in chunks:
            chunk.metadata["total_chunks"] = len(chunks)
            chunk.metadata["skipped_pages"] = ",".join(map(str, skipped))
        return ProcessedDocument(doc_id, filename, page_count, chunks, skipped)
