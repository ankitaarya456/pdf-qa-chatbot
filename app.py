import os
import re
import tempfile
from typing import List, Tuple

import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document  # ✅ important for type hints
from transformers import pipeline

# ======================================================
# 1. STREAMLIT PAGE CONFIG
# ======================================================
st.set_page_config(
    page_title="PDF Insight Assistant using RAG",
    page_icon="📄",
    layout="wide",
)

# ======================================================
# 2. THEME TOGGLE (LIGHT / DARK)
# ======================================================
theme_choice = st.sidebar.selectbox(
    "Theme",
    ["Dark (Green)", "Light (Green)"],
    index=0,
)

if "Dark" in theme_choice:
    primary = "#22c55e"
    primary_soft = "#bbf7d0"
    bg = "#020617"
    card_bg = "#020617"
    text = "#e5e7eb"
    sidebar_bg = "#020617"
else:
    primary = "#16a34a"
    primary_soft = "#bbf7d0"
    bg = "#f8fafc"
    card_bg = "#ffffff"
    text = "#111827"
    sidebar_bg = "#e5e7eb"

CUSTOM_CSS = f"""
<style>
:root {{
  --primary: {primary};
  --primary-soft: {primary_soft};
  --bg: {bg};
  --card-bg: {card_bg};
  --text-color: {text};
  --sidebar-bg: {sidebar_bg};
}}

.stApp {{
  background-color: var(--bg);
}}

section[data-testid="stSidebar"] {{
  background-color: var(--sidebar-bg) !important;
  color: var(--text-color) !important;
}}

.block-container {{
  color: var(--text-color);
  background-color: var(--card-bg);
  border-radius: 18px;
  padding: 1.5rem 2rem 2.5rem 2rem;
  box-shadow: 0 18px 45px rgba(15,23,42,0.55);
  margin-top: 1.5rem;
  margin-bottom: 2.5rem;
}}

h1 {{
  margin-bottom: 0.4rem;
}}

p {{
  font-size: 0.95rem;
}}

input[type="text"] {{
  border-radius: 0.75rem !important;
  border: 1px solid #4b5563 !important;
  padding: 0.65rem 0.8rem !important;
  outline: none !important;
  box-shadow: none !important;
  transition: all 0.15s ease-out !important;
}}

input[type="text"]:focus {{
  border: 1px solid var(--primary) !important;
  box-shadow: 0 0 0 1px rgba(34,197,94,0.35) !important;
}}

.stButton > button {{
  background: linear-gradient(135deg, var(--primary), #15803d);
  color: white;
  border-radius: 999px !important;
  padding: 0.45rem 1.4rem !important;
  border: none;
  font-weight: 600;
  font-size: 0.95rem;
  transition: transform 0.15s ease-out, box-shadow 0.15s ease-out, background 0.2s ease-out;
}}

.stButton > button:hover {{
  transform: translateY(-1px);
  box-shadow: 0 8px 20px rgba(22,163,74,0.45);
  background: linear-gradient(135deg, #22c55e, #15803d);
}}

.stSlider > div > div > div > div {{
  background-color: var(--primary) !important;
  height: 6px !important;
  border-radius: 999px !important;
}}

.stSlider > div > div > div > div > div {{
  background-color: var(--primary) !important;
  border: 3px solid #ffffff !important;
  width: 22px !important;
  height: 22px !important;
  border-radius: 50% !important;
  box-shadow: 0 0 10px rgba(34,197,94,0.7) !important;
  transition: transform 0.15s ease-out, box-shadow 0.15s ease-out;
}}

.stSlider > div > div > div > div > div:hover {{
  transform: scale(1.05);
  box-shadow: 0 0 14px rgba(34,197,94,0.9) !important;
}}

.stSlider > div > div > div[data-testid="stTickBar"] {{
  background: transparent !important;
}}

.streamlit-expanderHeader {{
  font-weight: 600;
}}

div[data-testid="stExpander"] > details {{
  border-radius: 0.75rem;
  border: 1px solid #4b5563;
  background-color: rgba(15,23,42,0.65);
}}

.stAlert {{
  border-radius: 0.75rem;
}}

.css-1iyw2u1, .css-10trblm {{
  transition: color 0.2s ease-out;
}}
</style>
"""

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# ======================================================
# 3. TITLE & HIGH-LEVEL DESCRIPTION
# ======================================================
st.title("📄 PDF Insight Assistant using RAG")
st.write(
    """
This app is designed to implement a full **Retrieval-Augmented Generation (RAG)** pipeline on top of your PDFs.

1. Upload one or more PDF documents.
2. Ask *any* question, generate summaries, or extract keywords.
3. The app shows retrieved chunks, similarity scores, and page numbers for transparency.
"""
)

# ======================================================
# 4. HUGGINGFACE MODELS (EMBEDDINGS + GENERATION)
# ======================================================
@st.cache_resource(show_spinner=True)
def get_embedding_model():
    """
    Sentence-level embedding model used for dense retrieval.
    Small and fast → good for Streamlit Cloud.
    """
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    return HuggingFaceEmbeddings(model_name=model_name)


@st.cache_resource(show_spinner=True)
def get_text_generator():
    """
    A single generative model (FLAN-T5-small) used for:
    - Question answering
    - Abstractive summarization
    - Keyword / concept extraction
    """
    model_name = "google/flan-t5-small"
    return pipeline("text2text-generation", model=model_name, tokenizer=model_name)


def generate_text(prompt: str, max_new_tokens: int = 256) -> str:
    """Helper around the HuggingFace pipeline."""
    generator = get_text_generator()
    result = generator(
        prompt,
        max_new_tokens=max_new_tokens,
        clean_up_tokenization_spaces=True,
    )[0]["generated_text"]
    return result.strip()


# ======================================================
# 5. HELPER: PERSONAL INFO REDACTION
# ======================================================
def redact_personal_info(text: str) -> str:
    """
    Very simple regex-based redaction to hide obvious personal identifiers
    (signatures, roll numbers, explicit names lines).
    """
    patterns = [
        r"signature of the student.*",
        r"signature of student.*",
        r"roll no\.\s*\S+",
        r"roll number\s*\S+",
        r"enrollment no\.\s*\S+",
        r"enrolment no\.\s*\S+",
        r"student name\s*:.*",
        r"name\s*:\s*.*",
    ]
    for p in patterns:
        text = re.sub(p, "[REDACTED]", text, flags=re.IGNORECASE)
    return text


def truncate_text(text: str, max_chars: int = 4000) -> str:
    """
    HuggingFace small models have limited context window.
    To avoid crashing, truncate very long contexts.
    """
    if len(text) <= max_chars:
        return text
    return text[:max_chars]


# ======================================================
# 6. SIDEBAR – UPLOAD + SETTINGS + MODE
# ======================================================
st.sidebar.header("📁 Upload PDFs & Global Settings")

uploaded_files = st.sidebar.file_uploader(
    "Upload one or more PDF files",
    type=["pdf"],
    accept_multiple_files=True,
)

chunk_size = st.sidebar.slider(
    "Chunk size (characters)",
    min_value=500,
    max_value=2000,
    value=1000,
    step=100,
    help="Larger chunks ⇒ more context per chunk but fewer chunks overall.",
)

chunk_overlap = st.sidebar.slider(
    "Chunk overlap (characters)",
    min_value=0,
    max_value=500,
    value=200,
    step=50,
    help="Overlap helps preserve context continuity across chunks.",
)

top_k = st.sidebar.slider(
    "Top-k relevant chunks to retrieve",
    min_value=1,
    max_value=10,
    value=4,
    step=1,
    help="How many chunks to use for answering / summarizing.",
)

redact_flag = st.sidebar.checkbox(
    "Redact possible personal info from context",
    value=True,
)

mode = st.sidebar.radio(
    "Assistant Mode",
    ["Question Answering", "Global Summary", "Local Summary (based on question)", "Keyword Extraction"],
)

# ======================================================
# 7. DOCUMENT LOADING & VECTOR STORE CREATION
# ======================================================
def load_and_split_pdfs(files) -> List[Document]:
    """
    1. Save each uploaded PDF to a temporary file
    2. Load text using PyPDFLoader
    3. Split into overlapping chunks with RecursiveCharacterTextSplitter
    """
    all_docs: List[Document] = []

    for file in files:
        # Save to a temporary file because PyPDFLoader expects a path
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(file.read())
            tmp_path = tmp_file.name

        loader = PyPDFLoader(tmp_path)
        pdf_docs = loader.load()
        all_docs.extend(pdf_docs)

        # Clean up temp file
        os.remove(tmp_path)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
    )
    chunks = splitter.split_documents(all_docs)
    return chunks


def build_vector_store(docs: List[Document]) -> Chroma:
    """
    Build a fresh Chroma vector store from documents.

    NOTE:
    - persist_directory=None → no on-disk persistence.
      Iska matlab har run me naya in-memory store banega,
      old chunks accidentally reuse nahi honge.
    """
    embeddings = get_embedding_model()
    vectordb = Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        persist_directory=None,  # ✅ important change to avoid old PDFs mixing
    )
    return vectordb


vectordb = None
docs: List[Document] = []

if uploaded_files:
    with st.spinner("📚 Reading PDFs and building vector store..."):
        docs = load_and_split_pdfs(uploaded_files)
        st.success(f"Loaded and split into **{len(docs)}** text chunks.")

        if len(docs) == 0:
            st.error(
                "Could not extract any text from the uploaded PDFs.\n"
                "They may be scanned/image-only or empty.\n"
                "Please try another PDF with selectable text."
            )
            vectordb = None
        else:
            vectordb = build_vector_store(docs)
            st.sidebar.success("✅ Vector store ready! You can now query or summarize.")
else:
    st.info("👈 Upload at least one PDF from the sidebar to get started.")

# ======================================================
# 8. CORE RAG OPERATIONS
# ======================================================
def retrieve_with_scores(query: str, k: int = 4) -> List[Tuple[Document, float]]:
    """
    Wrapper around Chroma.similarity_search_with_score.
    Returns (Document, similarity_score) pairs.
    """
    assert vectordb is not None, "Vector store is not initialized."
    results = vectordb.similarity_search_with_score(query, k=k)
    return results


def build_context_from_results(
    results: List[Tuple[Document, float]],
    redact: bool = True,
    max_chars: int = 4000,
) -> str:
    """
    Concatenate retrieved chunks into a single context string.
    Optionally apply redaction and truncation.
    """
    texts = []
    for doc, _score in results:
        chunk_text = doc.page_content
        if redact:
            chunk_text = redact_personal_info(chunk_text)
        texts.append(chunk_text)

    context = "\n\n".join(texts)
    context = truncate_text(context, max_chars=max_chars)
    return context


# ======================================================
# 9. MAIN UI – MODES
# ======================================================
st.subheader("🧠 Assistant Interface")

if mode in ["Question Answering", "Local Summary (based on question)", "Keyword Extraction"]:
    user_question = st.text_input(
        "Type your query here:",
        placeholder=(
            "Examples: "
            "• What are the main contributions of this paper? "
            "• Explain the methodology. "
            "• What are future work directions?"
        ),
    )
else:
    user_question = ""  # not needed for global summary

# ------------------ HANDLER ---------------------------
if st.button("Run"):
    if not uploaded_files:
        st.warning("Please upload at least one PDF first.")
    elif vectordb is None:
        st.warning("Vector store is not ready (no readable text).")
    elif mode != "Global Summary" and not user_question.strip():
        st.warning("Please enter a query for this mode.")
    else:
        try:
            # ========== GLOBAL SUMMARY ==========
            if mode == "Global Summary":
                query = "overall summary and main contributions of the documents"
                with st.spinner("🔎 Retrieving representative chunks and generating global summary..."):
                    results = retrieve_with_scores(query, k=max(top_k * 2, 6))
                    context = build_context_from_results(results, redact=redact_flag)

                    prompt = (
                        "You are summarizing a collection of PDF documents (e.g., project report / thesis / research papers).\n"
                        "Using ONLY the context below, generate a high-level summary describing:\n"
                        " - overall topic and objectives\n"
                        " - key sections or chapters\n"
                        " - main findings / contributions\n\n"
                        f"Context:\n{context}\n\n"
                        "Write the summary in 5–8 clear sentences, suitable for an abstract-level description."
                    )
                    answer = generate_text(prompt, max_new_tokens=256)

            # ========== QUESTION ANSWERING ==========
            elif mode == "Question Answering":
                query = user_question.strip()
                with st.spinner("🔎 Retrieving relevant chunks and generating answer..."):
                    results = retrieve_with_scores(query, k=top_k)
                    context = build_context_from_results(results, redact=redact_flag)

                    prompt = (
                        "You are an intelligent assistant helping with understanding PDF documents.\n"
                        "Answer the user's question using ONLY the information present in the context below.\n"
                        "If the answer is not clearly present, say that you don't know.\n\n"
                        f"Context:\n{context}\n\n"
                        f"Question: {query}\n\n"
                        "Answer in 3–6 clear, technically sound sentences."
                    )
                    answer = generate_text(prompt, max_new_tokens=256)

            # ========== LOCAL SUMMARY ==========
            elif mode == "Local Summary (based on question)":
                query = user_question.strip()
                with st.spinner("🔎 Retrieving relevant chunks and generating local summary..."):
                    results = retrieve_with_scores(query, k=top_k)
                    context = build_context_from_results(results, redact=redact_flag)

                    prompt = (
                        "You are summarizing only the parts of the PDFs that are relevant to the following query.\n"
                        "Using ONLY the context below, write a focused summary that answers what the user is interested in.\n\n"
                        f"Context:\n{context}\n\n"
                        f"User focus: {query}\n\n"
                        "Write the summary in 4–7 sentences and stay close to the original content."
                    )
                    answer = generate_text(prompt, max_new_tokens=256)

            # ========== KEYWORD EXTRACTION ==========
            else:  # "Keyword Extraction"
                query = user_question.strip()
                with st.spinner("🔎 Retrieving relevant chunks and extracting keywords..."):
                    results = retrieve_with_scores(query, k=top_k)
                    context = build_context_from_results(results, redact=redact_flag)

                    prompt = (
                        "You are an NLP assistant performing keyword extraction.\n"
                        "From the context below, extract the most important **technical keywords and phrases** "
                        "that are relevant to the user's query. Group them into:\n"
                        " - Core concepts\n"
                        " - Methods / algorithms\n"
                        " - Datasets / tools\n\n"
                        f"Context:\n{context}\n\n"
                        f"User query: {query}\n\n"
                        "Return the answer as bullet points under the three headings."
                    )
                    answer = generate_text(prompt, max_new_tokens=256)

            # ----------------- DISPLAY ANSWER -----------------
            st.markdown("### ✅ Assistant Output")
            st.write(answer)

            # ----------------- DISPLAY CONTEXT ----------------
            if mode != "Global Summary":
                st.markdown("---")
                st.markdown("### 🔍 Retrieved Context & Similarity Scores")
                st.caption(
                    "For transparency: these are the chunks retrieved from the PDFs that the model used."
                )

                if "results" in locals():
                    with st.expander("View retrieved chunks", expanded=False):
                        for idx, (doc, score) in enumerate(results, start=1):
                            src = doc.metadata.get("source", "unknown source")
                            page = doc.metadata.get("page", "?")

                            st.markdown(
                                f"**Chunk {idx} — page {page} — similarity score: {score:.4f}**"
                            )

                            chunk_text = doc.page_content
                            if redact_flag:
                                chunk_text = redact_personal_info(chunk_text)
                            st.write(chunk_text)
                            st.markdown("---")

        except Exception as e:
            st.error(f"Something went wrong: {e}")
