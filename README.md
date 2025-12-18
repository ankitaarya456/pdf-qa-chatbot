# 📄 PDF Insight Assistant using RAG

A **context-aware, privacy-safe PDF Question Answering system** built using **Retrieval-Augmented Generation (RAG)** and **Streamlit**.  
The app allows users to upload multiple PDFs and interact with them using natural language queries, summaries, and keyword extraction — with full transparency into retrieved document chunks.

🔗 **Live App:** https://pdf-app-chatbot-ankita-arya.streamlit.app/

---

## 🧠 Key Capabilities

| Feature | Description |
|------|------------|
| 🔎 Question Answering | Ask natural language questions across one or more PDFs |
| 📌 Local Summary | Query-focused summaries using only relevant document sections |
| 🌍 Global Summary | High-level summary of entire PDF collection |
| 🏷 Keyword Extraction | Categorized technical keywords (concepts, methods, tools) |
| 🧾 Chunk Transparency | Displays retrieved chunks with similarity scores & page numbers |
| 🔐 Privacy Protection | Automatic redaction of names, roll numbers & signatures |
| 🎨 Theming | Dark / Light (Green) UI themes |
| ⚙ Tunable Retrieval | Control chunk size, overlap & Top-K retrieval |

---


### 🏗 Architecture

The app follows a standard **RAG Pipeline**:

PDF ➜ Text Extraction ➜ Chunking ➜ Embeddings ➜ Vector DB (Chroma) ➜ Retrieval ➜ LLM Output

---
## License
This project is licensed under the Apache License 2.0.
