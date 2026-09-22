# JANGO — Enterprise Document Intelligence Platform

JANGO is an enterprise document intelligence platform that lets users upload documents, process them through a Retrieval-Augmented Generation (RAG) pipeline, and ask questions against their private knowledge base.

## 🎬 Project Demo

[▶️ Watch the 20-second project demo](https://portfolio-dqyw-opal.vercel.app/videos/jango.mp4)

## ✨ Features

- 🔐 JWT-based authentication
- 📄 PDF document upload and processing
- 🧩 Document chunking and embedding
- 🔎 Semantic search with ChromaDB
- 🤖 RAG-based question answering
- 🔒 User-scoped document isolation
- 📚 Source-grounded answers
- ⚡ FastAPI backend
- ⚛️ React frontend
- 🗄️ SQLAlchemy database integration

## 🧠 RAG Pipeline

```text
PDF Upload
    ↓
Text Extraction
    ↓
Document Chunking
    ↓
Embeddings
    ↓
ChromaDB
    ↓
Semantic Search
    ↓
LLM
    ↓
Source-Grounded Answer
🛠️ Tech Stack

Frontend

React
Vite
TypeScript

Backend

Python
FastAPI
Pydantic
SQLAlchemy

AI / RAG

LangChain
ChromaDB
Ollama
Qwen3

Database & Security

SQLite / PostgreSQL
JWT
bcrypt
📌 Project

JANGO is designed around private, user-scoped document intelligence, allowing users to query their uploaded knowledge base while keeping document retrieval isolated between users.
