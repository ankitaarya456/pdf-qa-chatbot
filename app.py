import os
import re
import tempfile
import uuid
from typing import List, Tuple

import streamlit as st
import torch
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# ======================================================
# 1. STREAMLIT CONFIG
# ======================================================
st.set_page_config(
    page_title="Multi-PDF Insight Assistant using RAG",
    page_icon="📄",
    layout="wide",
)

# ======================================================
# 2. UI STYLING (PROFESSIONAL DARK THEME)
# ======================================================
CUSTOM_CSS = """
<style>
:root {
  --primary: #22c55e;
  --bg: #020617;
  --card: #020617;
  --text: #e5e7eb;
  --border: #334155;
}

.stApp {
  background-color: var(--bg);
  color: var(--text);
}

.block-container {
  background-color: var(--card);
  border-radius: 18px;
  padding: 1.8rem 2.2rem 2.8rem 2.2rem;
  box-shadow: 0 18px 45px rgba(15,23,42,0.55);
  margin-top: 1.5rem;
}

section[data-testid="stSidebar"] {
  background-color: #020617 !important;
  border-right: 1px solid var(--border);
}

input[type="text"] {
  border-radius: 12px !important;
  border: 1px solid var(--border) !important;
  padding: 0.65rem 0.85rem !important;
  background-color: #020617 !important;
  color: var(--text) !important;
}

input[type="text"]:focus {
  border: 1px solid var(--primary) !important;
  box-shadow: 0 0 0 1px rgba(34,197,94,0.4) !important;
}

.stButton > button {
  background: linear-gradient(135deg, #22c55e, #15803d);
  color: white;
  border-radius: 999px;
  padding: 0.45rem 1.5rem;
  font-weight: 600;
  border: none;
}

.stAlert {
  border-radius: 14px;
}

div[data-testid="stExpander"] > details {
  border-radius: 14px;
  border: 1px solid var(--border);
  background-color: rgba(15,23,42,0.65);
}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# ======================================================
# 3. TITLE
# ======================================================
st.title("📄 Multi-PDF Insight Assistant using RAG")
st.caption("✅ Stable multi-document RAG with strict relevance guardrails")

# ======================================================
# 4. MODELS
# ======================================================
@st.cache_resource
def load_llm():
    model_name = "google/flan-t5-small"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    return tokenizer, model


@st.cache_resource
def load_embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )


def generate_text(prompt: str, max_new_tokens: int = 256) -> str:
    tokenizer, model = load_llm()
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=512
    )
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=max_new_tokens)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# ======================================================
# 5. HELPERS
# ======================================================
def redact_personal_info(text: str) -> str:
    patterns = [
        r"roll no\.\s*\S+",
        r"enrollment no\.\s*\S+",
        r"student name\s*:.*",
    ]
    for p in patterns:
        text = re.sub(p, "[REDACTED]", text, flags=re.IGNORECASE)
    return text


def is_relevant(results: List[Tuple[Document, float]], threshold: float = 0.75) -> bool:
    """
    Chroma similarity_search_with_score:
    LOWER score = MORE similar
    """
    if not results:
        return False
    relevant_hits = [score for _, score in results if score < threshold]
    return len(relevant_hits) >= 2

# ======================================================
# 6. SIDEBAR
# ======================================================
st.sidebar.header("📁 Upload PDFs")

uploaded_files = st.sidebar.file_uploader(
    "Upload multiple PDFs",
    type=["pdf"],
    accept_multiple_files=True
)

chunk_size = st.sidebar.slider("Chunk size", 500, 2000, 1000, 100)
chunk_overlap = st.sidebar.slider("Chunk overlap", 0, 500, 200, 50)
top_k = st.sidebar.slider("Top-K chunks", 2, 10, 6)

# ======================================================
# 7. LOAD & SPLIT PDFs
# ======================================================
def load_and_split_pdfs(files) -> List[Document]:
    all_docs = []

    for file in files:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(file.read())
            path = tmp.name

        loader = PyPDFLoader(path)
        docs = loader.load()

        for d in docs:
            d.metadata["source"] = file.name

        all_docs.extend(docs)
        os.remove(path)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    return splitter.split_documents(all_docs)

# ======================================================
# 8. VECTOR STORE
# ======================================================
vectordb = None

if uploaded_files:
    with st.spinner("📚 Processing PDFs..."):
        docs = load_and_split_pdfs(uploaded_files)

        vectordb = Chroma.from_documents(
            documents=docs,
            embedding=load_embeddings(),
            collection_name=f"pdf_rag_{uuid.uuid4().hex}"
        )

    st.success(f"✅ Loaded {len(uploaded_files)} PDFs → {len(docs)} chunks")
else:
    st.info("👈 Upload PDFs to begin")

# ======================================================
# 9. QUERY INPUT
# ======================================================
query = st.text_input("Enter your question:")

# ======================================================
# 10. RUN
# ======================================================
if st.button("Run"):
    if not vectordb:
        st.warning("Upload PDFs first")
    elif not query.strip():
        st.warning("Enter a question")
    else:
        with st.spinner("🧠 Retrieving relevant content..."):
            results = vectordb.similarity_search_with_score(query, k=top_k)

        if not is_relevant(results):
            st.error(
                "❌ **Your question does not match the uploaded documents.**\n\n"
                "👉 Please ask something clearly related to the PDF content."
            )
            st.stop()

        grouped = {}
        for doc, _ in results:
            src = doc.metadata.get("source", "Unknown PDF")
            grouped.setdefault(src, []).append(
                redact_personal_info(doc.page_content)
            )

        context = ""
        for src, texts in grouped.items():
            context += f"\n\nFrom PDF: {src}\n" + "\n".join(texts)

        prompt = f"""
Answer ONLY using the context below.
If the answer is not present, say:
"I cannot find this information in the provided documents."

Context:
{context}

Question:
{query}
"""

        with st.spinner("✍️ Generating answer..."):
            answer = generate_text(prompt)

        st.markdown("### ✅ Assistant Output")
        st.caption("Generated strictly from the uploaded PDFs")
        st.write(answer)

        st.markdown("---")
        with st.expander("🔍 Retrieved Context & Similarity Scores"):
            for i, (doc, score) in enumerate(results, 1):
                st.markdown(
                    f"**Chunk {i} | PDF: {doc.metadata.get('source')} | "
                    f"Page: {doc.metadata.get('page')} | Score: {score:.4f}**"
                )
                st.write(doc.page_content)
