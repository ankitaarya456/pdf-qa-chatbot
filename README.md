# 📄 Multi-PDF Insight Assistant using RAG

A **context-aware and privacy-aware PDF Question Answering system** implemented using **Retrieval-Augmented Generation (RAG)** and **Streamlit**.

The application allows users to upload multiple PDF documents and interact with them through **natural language questions**. All responses are generated **strictly from retrieved document content**, with transparent chunk-level retrieval and **relevance guardrails** to minimize hallucinated outputs.

🔗 **Live Application**  
https://pdf-app-chatbot-ankita-arya.streamlit.app/

---

## 🧠 Project Overview

This system follows a **standard RAG-based architecture**, where **information retrieval precedes language generation**.  
Only the content retrieved from uploaded PDF documents is supplied as context to the language model.

If a user query is **not sufficiently relevant** to the uploaded documents, the system **blocks response generation**, ensuring factual consistency and document grounding.

### Key Characteristics
- Multi-PDF document support  
- Chunk-level transparency with similarity scores  
- Relevance validation before answer generation  
- Privacy-aware text handling via redaction  
- Clean and professional Streamlit-based UI  

---

## ✨ Core Capabilities

| Feature | Description |
|------|------------|
| 🔎 Question Answering | Ask natural language questions across one or more PDFs |
| 🧾 Chunk Transparency | Displays retrieved chunks with page numbers and similarity scores |
| 🛡 Relevance Guardrails | Blocks answers when retrieved content is insufficiently relevant |
| 🔐 Privacy Protection | Automatic redaction of roll numbers, enrollment numbers, and names |
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
