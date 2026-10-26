"""CampusRAG Streamlit UI. Run: python -m streamlit run app.py"""

import logging

import streamlit as st

from config import ROOT, Settings
from database.session_store import SessionStore
from rag.document_processor import DocumentProcessor
from rag.rag_pipeline import RAGPipeline
from rag.vector_store import VectorStore
from services.chat_service import ChatService
from services.llm import OllamaLLM
from services.summarizer import Summarizer

st.set_page_config(page_title="CampusRAG | University AI Assistant", page_icon="🎓", layout="wide")


@st.cache_resource
def resources():
    settings = Settings.from_env()
    (ROOT / "data").mkdir(exist_ok=True)
    logger = logging.getLogger()
    if not any(getattr(handler, "campusrag_handler", False) for handler in logger.handlers):
        handler = logging.FileHandler(ROOT / "data" / "campusrag.log", encoding="utf-8")
        handler.campusrag_handler = True
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    sessions = SessionStore(settings.sqlite_path)
    vectors = VectorStore(settings)
    llm = OllamaLLM(settings)
    pipeline = RAGPipeline(settings, vectors, llm)
    return settings, sessions, vectors, llm, ChatService(sessions, pipeline, settings.memory_messages)


def show_sources(citations):
    if citations:
        with st.expander(f"Supporting sources · {len(citations)}"):
            for index, source in enumerate(citations, 1):
                st.markdown(f"**[{index}] {source['filename']} · Page {source['page']}**")
                st.text(source["snippet"])


def show_error(exc):
    logging.getLogger(__name__).exception("UI operation failed")
    if isinstance(exc, (ValueError, RuntimeError)):
        st.error(str(exc))
    else:
        st.error("This operation could not be completed. Check data/campusrag.log for details, then try again.")


try:
    settings, sessions, vectors, llm, chat = resources()
except Exception as exc:
    st.title("CampusRAG")
    st.error("CampusRAG could not initialize. Check .env, storage permissions, and installed dependencies.")
    if isinstance(exc, (ValueError, RuntimeError)):
        st.info(str(exc))
    logging.getLogger(__name__).exception("Initialization failed")
    st.stop()

st.markdown("""<style>
    [data-testid="stAppViewContainer"] {
        --campus-content-width: 1160px;
        --campus-gutter: clamp(1rem, 3vw, 3rem);
    }
    /* Match the page and sticky chat composer without affecting the sidebar. */
    [data-testid="stMainBlockContainer"],
    [data-testid="stBottomBlockContainer"] {
        box-sizing: border-box;
        width: 100%;
        max-width: var(--campus-content-width);
        margin-inline: auto;
        padding-inline: var(--campus-gutter);
    }
    [data-testid="stMainBlockContainer"] {
        /* Clear Streamlit's fixed header before rendering the first caption. */
        padding-top: 5rem;
        padding-bottom: 1.5rem;
    }
    [data-testid="stBottomBlockContainer"] {
        padding-top: 0.5rem;
        padding-bottom: 1rem;
    }
    .st-key-campus_header[data-testid="stVerticalBlock"],
    .st-key-campus_header [data-testid="stVerticalBlock"] {gap: 0.375rem;}
    .st-key-campus_header [data-testid="stCaptionContainer"],
    .st-key-campus_header [data-testid="stMarkdownContainer"] {margin-bottom: 0;}
    .st-key-campus_header [data-testid="stCaptionContainer"] p {
        margin: 0;
        padding-block: 0.125rem;
        line-height: 1.6;
        overflow-wrap: anywhere;
    }
    .st-key-campus_header h1 {
        margin: 0;
        padding-block: 0;
        line-height: 1.2;
    }
    .st-key-campus_header [data-testid="stMarkdownContainer"] p {margin: 0;}
    .st-key-campus_stats [data-testid="stHorizontalBlock"] {
        display: grid;
        grid-template-columns: repeat(3, minmax(0, 1fr));
        gap: clamp(0.75rem, 2vw, 1.5rem);
        align-items: start;
    }
    .st-key-campus_stats [data-testid="stColumn"] {width: 100%; min-width: 0;}
    .st-key-campus_stats [data-testid="stMarkdownContainer"],
    .st-key-campus_stats [data-testid="stMarkdownContainer"] p {
        white-space: normal;
        overflow: visible;
        text-overflow: clip;
    }
    .st-key-campus_stats [data-testid="stMetricLabel"] {height: auto;}
    [data-testid="stSidebar"] {border-right: 1px solid #dfe7ee;}
    [data-testid="stChatMessage"] {
        border: 1px solid #e2e9ef;
        border-radius: 12px;
        min-width: 0;
    }
    [data-testid="stChatMessageContent"] {min-width: 0; overflow-wrap: anywhere;}
    @media (max-width: 640px) {
        [data-testid="stMainBlockContainer"] {padding-top: 4.75rem;}
        .st-key-campus_header h1 {font-size: clamp(2rem, 7vw, 2.75rem);}
        .st-key-campus_stats [data-testid="stMetricLabel"] {min-height: 2.8em;}
        .st-key-campus_stats [data-testid="stMetricValue"] {
            font-size: clamp(1.25rem, 4vw, 1.75rem);
            line-height: 1.25;
        }
    }
</style>""", unsafe_allow_html=True)

if "model_status" not in st.session_state:
    st.session_state.model_status = llm.status()
if "workspace_view" not in st.session_state:
    st.session_state.workspace_view = "Conversation"


def process_files(files):
    processor = DocumentProcessor(settings.chunk_size, settings.chunk_overlap)
    with st.status("Processing documents…", expanded=True) as status:
        failures = 0
        for filename, data in files:
            st.write(f"Processing {filename}")
            try:
                result = vectors.ingest(data, filename, processor)
                if result.duplicate:
                    st.info(f"Already indexed as {result.filename}; skipped duplicate.")
                else:
                    st.success(f"{result.filename}: indexed {result.chunk_count} chunks.")
                if result.skipped_pages:
                    st.warning(f"No text on page(s) {', '.join(map(str, result.skipped_pages))}. These pages were excluded; OCR is not included.")
            except Exception as error:
                failures += 1
                show_error(error)
        status.update(label="Processing finished" if not failures else "Processing finished with errors",
                      state="complete" if not failures else "error", expanded=True)


with st.sidebar:
    st.title("🎓 CampusRAG")
    st.caption("Your university documents. Answers with evidence. Conversations that stay with you.")
    st.divider()
    st.subheader("Document library")
    uploads = st.file_uploader("Upload university PDFs", type=["pdf"], accept_multiple_files=True,
                               help="Text-based PDFs, up to 20 MB and 500 pages each. Scans need OCR first.")
    if st.button("Process documents", type="primary", disabled=not uploads, use_container_width=True):
        process_files([(file.name, file.getvalue()) for file in uploads])
    if st.button("Load fictional sample PDFs", use_container_width=True,
                 help="Three small demonstration policies from fictional Meridian Example University."):
        process_files([(path.name, path.read_bytes()) for path in sorted((ROOT / "data" / "sample_documents").glob("*.pdf"))])
    try:
        documents = vectors.list_documents()
    except Exception as exc:
        show_error(exc)
        documents = []
    if documents:
        with st.expander(f"Indexed documents · {len(documents)}", expanded=True):
            for doc in documents:
                st.text(doc["source"])
                st.caption(f"{doc['page_count']} pages · {doc['chunk_count']} chunks · {doc['document_id'][:8]}")
                if doc.get("skipped_pages"):
                    st.caption(f"Pages without text: {doc['skipped_pages']}")
    else:
        st.caption("Your library is empty. Upload PDFs or load the fictional examples.")

    st.divider()
    st.subheader("Conversations")
    if st.button("＋ New chat", use_container_width=True):
        st.session_state.chat_selector = sessions.create_session()
        st.session_state.workspace_view = "Conversation"
    session_list = sessions.list_sessions()
    if not session_list:
        st.session_state.chat_selector = sessions.create_session()
        session_list = sessions.list_sessions()
    titles = {item["id"]: item["title"] for item in session_list}
    if st.session_state.get("chat_selector") not in titles:
        st.session_state.chat_selector = session_list[0]["id"]
    selected_session = st.selectbox("Chat session", list(titles), key="chat_selector",
                                    format_func=lambda sid: f"{titles[sid]} · {sid[:6]}")
    if st.button("Delete selected chat", use_container_width=True):
        try:
            sessions.delete_session(selected_session)
            # The next run reconciles the removed widget value against live IDs.
            st.rerun()
        except Exception as exc:
            show_error(exc)

    st.divider()
    st.subheader("Document summary")
    document_map = {doc["document_id"]: doc for doc in documents}
    summary_doc = st.selectbox("Document to summarize", list(document_map),
                               format_func=lambda did: f"{document_map[did]['source']} · {did[:6]}",
                               disabled=not documents, placeholder="Index a PDF first")
    if st.button("Generate summary", disabled=not documents, use_container_width=True):
        try:
            progress = st.progress(0, text="Reading every document chunk…")
            summary = Summarizer(settings, vectors, llm).summarize(
                summary_doc, lambda done, total: progress.progress(done / total, text=f"Summarizing chunk {done} of {total}…"),
            )
            progress.empty()
            st.session_state.latest_summary = (document_map[summary_doc]["source"], summary)
            st.session_state.workspace_view = "Document summary"
        except Exception as exc:
            show_error(exc)

    st.divider()
    ready, status_text = st.session_state.model_status
    if ready:
        st.caption(f"🟢 {status_text}")
    else:
        st.caption(f"⚪ {status_text}")
    if st.button("Refresh model status", use_container_width=True):
        st.session_state.model_status = llm.status()
        st.rerun()
    st.caption("Local storage · No paid API · Shared document library")

with st.container(key="campus_header"):
    st.caption("SESSION-AWARE UNIVERSITY AI ASSISTANT")
    st.title("Clarity for your campus questions.")
    st.write("Ask your documents, keep the conversation, and follow every answer back to its source.")
with st.container(key="campus_stats"):
    left, middle, right = st.columns(3)
    left.metric("Indexed PDFs", len(documents))
    middle.metric("Conversations", len(session_list))
    right.metric("Recent memory", f"{settings.memory_messages} messages")
st.radio("Workspace", ["Conversation", "Document summary"], horizontal=True, key="workspace_view", label_visibility="collapsed")

if st.session_state.workspace_view == "Document summary":
    if "latest_summary" in st.session_state:
        filename, summary = st.session_state.latest_summary
        st.subheader(filename)
        st.caption(f"Processed all {summary.chunk_count} text chunks across a {summary.page_count}-page document. Summaries may omit minor details.")
        st.markdown(summary.text)
        show_sources(summary.citations)
    else:
        st.info("Select an indexed PDF in the sidebar, then choose Generate summary. Longer documents take more time.")
else:
    try:
        messages = sessions.get_messages(selected_session)
    except Exception as exc:
        show_error(exc)
        st.stop()
    if not messages:
        st.subheader("Start with a question")
        st.write("Upload your university's policies, or explore the clearly labeled fictional sample PDFs.")
        with st.container(border=True):
            st.markdown("**Try with the fictional examples**")
            st.write("What is the minimum attendance requirement?")
            st.write("Does this also apply to M.Tech students?")
            st.write("What is the deadline to request examination re-evaluation?")
        st.caption("Answers use indexed documents. Conversation history helps interpret follow-ups; it is never treated as policy evidence.")
    for message in messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            show_sources(message["citations"])
    question = st.chat_input("Ask about your university documents…", disabled=not documents, max_chars=1000)
    if not documents:
        st.info("Add at least one PDF using the sidebar to enable chat.")
    if question:
        with st.chat_message("user"):
            st.markdown(question)
        try:
            with st.spinner("Finding evidence and preparing your answer…"):
                chat.ask(selected_session, question)
            st.rerun()
        except Exception as exc:
            show_error(exc)
