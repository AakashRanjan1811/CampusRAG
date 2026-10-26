# CampusRAG
### Session-Aware University AI Assistant

CampusRAG is a locally running Retrieval-Augmented Generation (RAG) application that allows students to ask questions about university documents, receive answers with PDF citations, maintain conversations across sessions, and generate document summaries.

Built using **Python, LangChain, ChromaDB, Ollama, and Streamlit**, without requiring paid APIs.

## Features

- **PDF Document Processing:** Upload and index multiple PDFs using text extraction, chunking, and semantic embeddings.
- **Grounded Question Answering:** Retrieve relevant passages and generate contextual answers using a local LLM.
- **Source Citations:** Display supporting PDF filenames, page numbers, and source excerpts.
- **Conversational Memory:** Understand follow-up questions and preserve chat history using SQLite.
- **Document Summarization:** Generate structured, source-backed summaries of uploaded PDFs.
- **Persistent Vector Storage:** Store and retrieve document embeddings with ChromaDB.

## Tech Stack

| Component | Technology |
|---|---|
| Language | Python 3.11 |
| Interface | Streamlit |
| RAG Framework | LangChain |
| Vector Database | ChromaDB |
| Embedding Model | all-MiniLM-L6-v2 |
| Local LLM | Llama 3.2 (3B) via Ollama |
| PDF Processing | PyMuPDF |
| Conversation Storage | SQLite |
| Testing | pytest |

## Architecture

```text
PDF Upload
    ↓
Text Extraction (PyMuPDF)
    ↓
Text Chunking (LangChain)
    ↓
Embeddings (MiniLM)
    ↓
