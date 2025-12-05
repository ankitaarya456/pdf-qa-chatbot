# 📄 PDF Insight Assistant using RAG

A powerful **PDF Understanding Tool** built with **RAG (Retrieval-Augmented Generation)** and **Streamlit**.  
It allows you to upload one or more PDFs, ask questions, summarize content, and extract keywords using **retrieval + generative transformers**.

---

### 🚀 Features

✔ Upload multiple PDFs  
✔ Intelligent Question Answering using PDF content  
✔ Local & Global Summarization  
✔ Keyword Extraction grouped into categories  
✔ View **retrieved chunks + similarity score + page number**  
✔ Personal info redaction (privacy-safe)  
✔ Tunable chunk size, overlap, top-k  
✔ Light/Dark theme toggle  
✔ Uses open-source HuggingFace models  
✔ No data is stored (in-memory Chroma DB)

---

### 🧠 How It Works

This app uses a standard **RAG pipeline**:
PDF → Text Extraction → Chunking → Embedding → Vector Store → Retrieval → LLM Response
