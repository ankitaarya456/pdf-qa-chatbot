# 📄 PDF Q&A Chatbot (LangChain + HuggingFace + Chroma)

An AI-powered chatbot that lets you upload one or more PDF files and ask questions about their content.

### 🚀 Features
- Upload multiple PDFs
- Smart text extraction and splitting
- Embedding generation using Sentence Transformers
- Chroma vector database for semantic search
- HuggingFace QA model for intelligent answers
- Fully modern UI (dark/light theme, rounded sliders, gradient buttons)
- Deployable on Streamlit Cloud

---

## 🧠 How It Works
1. You upload PDF(s)
2. App extracts text and splits it into chunks
3. Each chunk → embeddings generated
4. Stored inside Chroma vector store
5. When you ask a question → relevant chunks are retrieved
6. HuggingFace QA model answers using exact PDF content

---

## 🛠️ Installation

Clone the repository:
```bash
git clone https://github.com/ankitaarya456/pdf-qa-chatbot.git
cd pdf-qa-chatbot
