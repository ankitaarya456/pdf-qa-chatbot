import os
import tempfile

import streamlit as st

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

from transformers import pipeline


# ------------------------------------------------------
# 1. Page config
# ------------------------------------------------------
st.set_page_config(
    page_title="PDF Q&A Chatbot",
    page_icon="📄",
    layout="wide",
)

# ------------------------------------------------------
# 2. Theme toggle (Light / Dark)
# ------------------------------------------------------
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

/* main container as a card */
.block-container {{
  color: var(--text-color);
  background-color: var(--card-bg);
  border-radius: 18px;
  padding: 1.5rem 2rem 2.5rem 2rem;
  box-shadow: 0 18px 45px rgba(15,23,42,0.55);
  margin-top: 1.5rem;
  margin-bottom: 2.5rem;
}}

/* nicer title spacing */
h1 {{
  margin-bottom: 0.4rem;
}}

/* description text a bit softer */
p {{
  font-size: 0.95rem;
}}

/* text input styling */
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

/* button styling */
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

/* SLIDER: green theme + rounded + glow */
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

/* remove grey tick bar behind slider */
.stSlider > div > div > div[data-testid="stTickBar"] {{
  background: transparent !important;
}}

/* expander card look */
.streamlit-expanderHeader {{
  font-weight: 600;
}}

div[data-testid="stExpander"] > details {{
  border-radius: 0.75rem;
  border: 1px solid #4b5563;
  background-color: rgba(15,23,42,0.65);
}}

/* success / warning boxes a bit softer rounded */
.stAlert {{
  border-radius: 0.75rem;
}}

/* subtle animation for status text */
.css-1iyw2u1, .css-10trblm {{
  transition: color 0.2s ease-out;
}}
</style>
"""

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# ------------------------------------------------------
# 3. Title & description
# ------------------------------------------------------
st.title("📄 PDF Q&A Chatbot (LangChain + HuggingFace + Chroma)")
st.write(
    "Upload one or more PDF files, and ask questions about their content. "
    "This app uses **LangChain**, **HuggingFace embeddings**, **Chroma vector store**, "
    "and a **question-answering model** from HuggingFace – all completely free."
)


# ------------------------------------------------------
# 4. Cache HuggingFace Models (Embeddings + QA)
# ------------------------------------------------------
@st.cache_resource(show_spinner=True)
def get_embedding_model():
    """Return a small, fast sentence-transformer model for embeddings."""
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    return HuggingFaceEmbeddings(model_name=model_name)


@st.cache_resource(show_spinner=True)
def get_qa_pipeline():
    """Return a lightweight QA pipeline from HuggingFace."""
    qa_model_name = "deepset/roberta-base-squad2"
    qa = pipeline("question-answering", model=qa_model_name, tokenizer=qa_model_name)
    return qa


# ------------------------------------------------------
# 5. Sidebar - PDF Upload + Chunk Settings
# ------------------------------------------------------
st.sidebar.header("📁 Upload PDFs & Settings")

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
    help="Larger chunks → more context but slower & heavier.",
)

chunk_overlap = st.sidebar.slider(
    "Chunk overlap (characters)",
    min_value=0,
    max_value=500,
    value=200,
    step=50,
    help="Overlap improves continuity between chunks.",
)

top_k = st.sidebar.slider(
    "Number of relevant chunks to search (top_k)",
    min_value=1,
    max_value=10,
    value=3,
    step=1,
    help="How many chunks to retrieve from the vector store.",
)


# ------------------------------------------------------
# 6. Load PDFs, Split Text
# ------------------------------------------------------
def load_and_split_pdfs(files):
    """
    1. Save uploaded PDFs to temp files
    2. Load them using PyPDFLoader
    3. Split with RecursiveCharacterTextSplitter
    """
    all_docs = []

    for file in files:
        # Save to a temporary file (because PyPDFLoader needs a file path)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(file.read())
            tmp_path = tmp_file.name

        loader = PyPDFLoader(tmp_path)
        pdf_docs = loader.load()
        all_docs.extend(pdf_docs)

        # Remove the temp file
        os.remove(tmp_path)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
    )

    chunks = splitter.split_documents(all_docs)
    return chunks


@st.cache_resource(show_spinner=True)
def build_vector_store(docs):
    """Build an in-memory Chroma vector store from documents."""
    embeddings = get_embedding_model()
    vectordb = Chroma.from_documents(documents=docs, embedding=embeddings)
    return vectordb


# ------------------------------------------------------
# 7. Main Logic: Create VectorStore Once PDFs are Uploaded
# ------------------------------------------------------
vectordb = None
retriever = None

if uploaded_files:
    with st.spinner("📚 Reading PDFs and creating vector store..."):
        docs = load_and_split_pdfs(uploaded_files)
        st.success(f"Loaded and split into {len(docs)} chunks.")

        if len(docs) == 0:
            # No readable text extracted
            st.error(
                "Could not extract any text from the uploaded PDFs.\n\n"
                "They may be scanned/image-only or empty.\n"
                "Please try another PDF that has selectable text."
            )
        else:
            vectordb = build_vector_store(docs)
            retriever = vectordb.as_retriever(search_kwargs={"k": top_k})
            st.sidebar.success("✅ Vector store ready! Ask your question below.")
else:
    st.info("👈 Please upload one or more PDF files from the sidebar to get started.")


# ------------------------------------------------------
# 8. Question Input + Answer Generation
# ------------------------------------------------------
st.subheader("💬 Ask a question about your PDFs")

user_question = st.text_input(
    "Type your question here:",
    placeholder="Example: What is the main idea of this document?",
)

if st.button("Get Answer"):
    if not uploaded_files:
        st.warning("Please upload at least one PDF first.")
    elif retriever is None:
        # This covers the case where PDFs were image-only and we showed an error above
        st.warning(
            "Vector store is not ready. "
            "Make sure you uploaded a PDF with readable/selectable text."
        )
    elif not user_question.strip():
        st.warning("Please enter a question.")
    else:
        with st.spinner("🔎 Retrieving relevant context and generating answer..."):
            try:
                # Retrieve relevant chunks
                relevant_docs = retriever.invoke(user_question)

                if not relevant_docs:
                    st.warning("No relevant information found in the PDFs.")
                else:
                    qa = get_qa_pipeline()

                    best_answer = None
                    best_score = -1.0

                    # Run QA on each chunk separately, keep best answer
                    for i, doc in enumerate(relevant_docs, start=1):
                        context_text = doc.page_content

                        result = qa(
                            {
                                "question": user_question,
                                "context": context_text,
                            }
                        )

                        if result["score"] > best_score and result["answer"].strip():
                            best_score = result["score"]
                            best_answer = {
                                "answer": result["answer"],
                                "context": context_text,
                            }

                    if best_answer is None:
                        st.warning("The model could not find a confident answer.")
                    else:
                        st.markdown("### ✅ Answer")
                        st.write(best_answer["answer"])

                        with st.expander("🔍 View context used from the PDF"):
                            st.write(best_answer["context"])

            except Exception as e:
                st.error(f"Something went wrong: {e}")
