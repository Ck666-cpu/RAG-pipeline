# RAG-pipeline

# 🔍 RAG Chatbot Prototype

This project is a prototype of a **Retrieval-Augmented Generation (RAG) chatbot** using the LangChain framework. It loads a public dataset, embeds its content using sentence transformers, stores them in a FAISS vector store, and performs question answering using a transformer-based language model.

---

## 📚 Overview

RAG combines the strength of:
- **Retrieval-based models** for accurate context
- **Generative models** for fluent natural language answers

This notebook demonstrates:
1. Loading a HuggingFace dataset (`databricks/databricks-dolly-15k`)
2. Splitting documents into chunks
3. Embedding with `sentence-transformers`
4. Storing vectors with **FAISS**
5. Performing Q&A with a transformer-based model (`distilbert-base-uncased-distilled-squad`)

---

## 🛠️ Installation

Install required libraries:

```bash
pip install langchain torch transformers sentence-transformers datasets faiss-cpu ipywidgets
