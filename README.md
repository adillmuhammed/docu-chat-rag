# 📚 PDF Research Assistant

A **Retrieval-Augmented Generation (RAG)** based PDF question-answering application built with **Python, LangChain, ChromaDB, Hugging Face Embeddings, Groq, and Gradio**.

The application allows users to upload PDF documents, store their content in a vector database, and ask questions about the document using natural language.

Instead of manually searching through a long PDF, the system retrieves the most relevant sections and uses an LLM to generate a clear answer based on the retrieved information.

---

## 🎥 Project Demo

> Add your screen recording or demo GIF here.

<!-- Example:
![Project Demo](demo.gif)
-->

---

## ✨ Features

- 📄 Upload PDF documents through a Gradio interface
- 📖 Extract text from PDF files using PyPDF
- ✂️ Split documents into smaller chunks
- 🧠 Generate semantic embeddings using Hugging Face
- 🗄️ Store embeddings in ChromaDB
- 🔍 Perform semantic similarity search
- 🤖 Generate answers using Groq Llama
- 🛡️ Prevent duplicate chunks using SHA-256 based IDs
- 🔐 Keep API keys secure using environment variables
- 🎨 Simple and user-friendly Gradio interface
- 🧩 Separate PDF processing and question-answering logic

---

# 🔄 How It Works

The project consists of two main stages:

### 1. Document Ingestion

When a PDF is uploaded:

```text
📄 PDF
   ↓
PyPDFLoader
   ↓
Extract Text
   ↓
Recursive Character Text Splitter
   ↓
Text Chunks
   ↓
SHA-256 IDs
   ↓
Hugging Face Embeddings
   ↓
ChromaDB