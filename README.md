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
ChromaDB
    ↓
User Question → Query Reformulation
    ↓
Relevant Document Retrieval
    ↓
Ollama (Llama 3.2)
    ↓
Grounded Answer + Source Citations
```

SQLite maintains persistent conversation history and provides context for follow-up questions.

## Installation

**Prerequisites:** Python 3.11 and Ollama.

1. Clone the repository:

```bash
git clone https://github.com/AakashRanjan1811/CampusRAG.git
cd CampusRAG
```

2. Create and activate the virtual environment (Windows):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

3. Configure and download the local LLM:

```powershell
Copy-Item .env.example .env
ollama pull llama3.2:3b
```

4. Start the application:

```powershell
python -m streamlit run app.py
```

Open **http://localhost:8501** in your browser. Ensure Ollama is running.

## Usage

1. Upload university PDF documents or load the included fictional samples.
2. Ask questions and inspect supporting citations.
3. Continue with context-dependent follow-up questions.
4. Switch between previously saved conversations.
5. Select a document to generate a structured summary.

## Project Structure

```text
app.py                   Streamlit interface
config.py                Application configuration
rag/                     PDF processing and RAG pipeline
database/                SQLite conversation storage
services/                LLM, chat, and summary services
data/sample_documents/   Fictional sample PDFs
tests/                   Automated tests
scripts/                 Sample PDF generation
```

## Testing

Run:

```powershell
python -m pytest -q
```

The project passed **60 automated tests** during local validation, including integration checks using MiniLM and Ollama.

## Screenshots

Screenshots demonstrating the chat interface, source citations, and document summaries can be added here.

## Limitations

- Supports text-based PDFs; scanned documents require additional OCR support.
- Answer quality depends on retrieved evidence and the local language model.
- Designed for local, single-user operation rather than production deployment.

**Note:** Included sample university policies are fictional and do not represent official university regulations.
