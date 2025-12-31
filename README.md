# 📄 Multi-PDF Insight Assistant using RAG

A **context-aware and privacy-aware PDF Question Answering system** implemented using  
**Retrieval-Augmented Generation (RAG)** and **Streamlit**.

The application enables users to upload **multiple PDF documents** and interact with them using
**natural language queries**, **document summaries**, and **keyword extraction**.
All responses are generated strictly from **retrieved document content**, with transparent
chunk-level retrieval and relevance guardrails to reduce hallucinated answers.

🔗 **Live App:**  
https://pdf-app-chatbot-ankita-arya.streamlit.app/

---

## 🧠 Project Overview

This system follows a **standard RAG-based architecture**, where information retrieval precedes
language generation. Only the content retrieved from uploaded PDFs is provided as context to the
language model.

### Key characteristics:
- Multi-PDF document support
- Chunk-level transparency with similarity scores
- Relevance filtering before response generation
- Privacy-aware text handling through redaction
- Clean and professional Streamlit-based UI

If a user query is **not sufficiently relevant** to the uploaded documents, the system **blocks
response generation**.

---

## ✨ Key Capabilities

| Feature | Description |
|------|------------|
| 🔎 Question Answering | Ask natural language questions across one or more PDFs |
| 📌 Local Summary | Query-focused summaries using only relevant document sections |
| 🌍 Global Summary | High-level summary of the complete PDF collection |
| 🏷 Keyword Extraction | Extraction of technical keywords present in documents |
| 🧾 Chunk Transparency | Displays retrieved chunks with similarity scores and page numbers |
| 🛡 Relevance Guardrails | Blocks answers when retrieved content is insufficiently relevant |
| 🔐 Privacy Protection | Automatic redaction of names, roll numbers, and enrollment details |
| ⚙ Tunable Retrieval | Adjustable chunk size, overlap, and Top-K retrieval parameters |
| 🎨 UI Styling | Professional dark-themed Streamlit interface |

---

## License
This project is licensed under the Apache License 2.0.

---
## 🏗 Architecture (RAG Pipeline)

```text
PDF Upload
   ↓
Text Extraction (PyPDFLoader)
   ↓
Chunking (RecursiveCharacterTextSplitter)
   ↓
Embedding Generation (MiniLM)
   ↓
Vector Store (Chroma)
   ↓
Similarity-Based Retrieval (Top-K)
   ↓
Relevance Validation
   ↓
LLM Generation (FLAN-T5)
