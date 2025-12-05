# 📄 PDF Insight Assistant using RAG

A fully local, privacy-safe **PDF Analysis Tool** built using **Retrieval-Augmented Generation (RAG)** + **Streamlit**.  
Upload multiple PDFs and ask questions, extract keywords, or generate summaries using transparent chunk retrieval.

---

### 🧠 Features

| Feature | Description |
|--------|-------------|
| 🔎 Question Answering | Ask anything about one or more PDFs |
| 📌 Local Summary | Summary based only on query-specific sections |
| 🌎 Global Summary | High-level overview of entire PDFs |
| 🏷 Keyword Extraction | Categorized keywords (concepts, algorithms, tools) |
| 🧾 Chunk Display | Shows retrieved chunks + similarity score + page |
| 🔐 Privacy Safe | Personal info redaction (roll no., name, signatures) |
| 🎨 Theming | Light/Dark (Green) themes |
| ⚙ Tunable Settings | Chunk size, overlap, Top-K retrieval |

---

### 🏗 Architecture

The app follows a standard **RAG Pipeline**:

PDF ➜ Text Extraction ➜ Chunking ➜ Embeddings ➜ Vector DB (Chroma) ➜ Retrieval ➜ LLM Output
