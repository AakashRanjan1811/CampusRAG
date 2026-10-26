"""Validated configuration; relative paths always resolve against the project."""

from dataclasses import dataclass
from pathlib import Path
import os

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent
# Keep third-party model caches inside this portable local project.
os.environ.setdefault("HF_HOME", str(ROOT / ".cache" / "huggingface"))
os.environ.setdefault("TORCH_HOME", str(ROOT / ".cache" / "torch"))


@dataclass(frozen=True)
class Settings:
    ollama_model: str = "llama3.2:3b"
    ollama_base_url: str = "http://localhost:11434"
    chroma_dir: Path = ROOT / "data" / "chroma"
    sqlite_path: Path = ROOT / "data" / "campusrag.db"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    chunk_size: int = 800
    chunk_overlap: int = 120
    retrieval_top_k: int = 5
    llm_temperature: float = 0.0
    llm_context_tokens: int = 8192
    llm_timeout_seconds: int = 180
    memory_messages: int = 8
    max_summary_chunks: int = 100

    def __post_init__(self):
        if not 200 <= self.chunk_size <= 1200:
            raise ValueError("CHUNK_SIZE must be between 200 and 1200 characters.")
        if not 0 <= self.chunk_overlap < self.chunk_size:
            raise ValueError("CHUNK_OVERLAP must be nonnegative and smaller than CHUNK_SIZE.")
        if not 1 <= self.retrieval_top_k <= 12:
            raise ValueError("RETRIEVAL_TOP_K must be between 1 and 12.")
        if not 0 <= self.llm_temperature <= 1:
            raise ValueError("LLM_TEMPERATURE must be between 0 and 1.")
        if self.llm_context_tokens < 8192:
            raise ValueError("LLM_CONTEXT_TOKENS must be at least 8192 to fit evidence, prompts, and output.")
        if not 0 <= self.memory_messages <= 20:
            raise ValueError("MEMORY_MESSAGES must be between 0 and 20.")
        if self.llm_timeout_seconds < 1 or self.max_summary_chunks < 1:
            raise ValueError("Timeout and summary limits must be positive.")
        if not self.ollama_model.strip() or not self.embedding_model.strip():
            raise ValueError("Model names cannot be empty.")
        if not self.ollama_base_url.startswith(("http://", "https://")):
            raise ValueError("OLLAMA_BASE_URL must start with http:// or https://.")

    @classmethod
    def from_env(cls):
        load_dotenv(ROOT / ".env")

        def path(name, default):
            value = Path(os.getenv(name, default)).expanduser()
            return value if value.is_absolute() else ROOT / value

        return cls(
            ollama_model=os.getenv("OLLAMA_MODEL", "llama3.2:3b"),
            ollama_base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/"),
            chroma_dir=path("CHROMA_DIR", "data/chroma"),
            sqlite_path=path("SQLITE_PATH", "data/campusrag.db"),
            embedding_model=os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"),
            chunk_size=int(os.getenv("CHUNK_SIZE", "800")),
            chunk_overlap=int(os.getenv("CHUNK_OVERLAP", "120")),
            retrieval_top_k=int(os.getenv("RETRIEVAL_TOP_K", "5")),
            llm_temperature=float(os.getenv("LLM_TEMPERATURE", "0")),
            llm_context_tokens=int(os.getenv("LLM_CONTEXT_TOKENS", "8192")),
            llm_timeout_seconds=int(os.getenv("LLM_TIMEOUT_SECONDS", "180")),
            memory_messages=int(os.getenv("MEMORY_MESSAGES", "8")),
            max_summary_chunks=int(os.getenv("MAX_SUMMARY_CHUNKS", "100")),
        )
