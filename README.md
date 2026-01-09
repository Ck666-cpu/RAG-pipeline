# 🔍 FYP Workbench: Advanced Local RAG Pipeline

This project is a **Privacy-First Retrieval-Augmented Generation (RAG)** system designed for high-compliance environments. It runs entirely locally using **Ollama (LLM)** and **Qdrant (Vector DB)**, featuring a strict **MVVM Architecture** and **Role-Based Access Control (RBAC)**.

---

## 🚀 Key Features

### 1. 🧠 Corrective RAG (CRAG) with Hybrid Search
- **Retrieval:** Semantic search using `cross-encoder/ms-marco-MiniLM-L-6-v2` for reranking.
- **Fallback Logic:** Automatically switches to "General Chat Mode" if no relevant documents are found (Confidence < Threshold).
- **Context Awareness:** Remembers previous turns in the conversation using a persistent JSON memory system.

### 2. 🛡️ Role-Based Access Control (RBAC)
The system distinguishes between three user tiers:
- **Master Admin:** Can manage users (Create/Delete) but **cannot** chat or view documents (Security Principle).
- **Admin:** Can upload **Global** documents (visible to all) and **Private** documents. Full Chat access.
- **Staff:** Can only upload **Private** documents. Can view Global documents but not other users' Private documents.

### 3. 🏗️ MVVM Architecture
Designed for easy integration with any Frontend (GUI/Web):
- **Model (`fyp_service.py`, `model_db.py`):** Handles logic, database, and LLM interactions.
- **ViewModel (`view_model.py`):** Manages state, session persistence, and commands.
- **View:** (Your future UI) binds strictly to the ViewModel.

---

## 🛠️ Tech Stack

- **LLM Engine:** [Ollama](https://ollama.ai/) (Model: `tinyllama` or `phi3`)
- **Orchestration:** [LlamaIndex](https://www.llamaindex.ai/)
- **Vector Database:** [Qdrant](https://qdrant.tech/) (Running in Docker)
- **Embeddings:** `BAAI/bge-small-en-v1.5`
- **Reranker:** `cross-encoder/ms-marco-MiniLM-L-6-v2`

---

## 📦 Installation & Setup

### 1. Prerequisites
- **Python 3.10+**
- **Docker Desktop** (For Qdrant)
- **Ollama** (Installed and running)

### 2. Start Infrastructure
Start the Vector Database (Qdrant) via Docker:
```bash
docker run -p 6333:6333 -p 6334:6334 \
    -v $(pwd)/qdrant_storage:/qdrant/storage:z \
    qdrant/qdrant

```

Pull the LLM model (run in terminal):

```bash
ollama pull tinyllama
ollama serve

```

### 3. Install Dependencies

```bash
pip install -r FYP_Workbench/requirements.txt

```

---

## 🏃‍♂️ Usage

### Running the Test Simulation

To verify the entire pipeline (Auth -> RAG -> Memory), run the main test script:

```bash
python FYP_Workbench/test_auth.py

```

### Directory Structure

```
FYP_Workbench/
├── fyp_service.py      # CORE: RAG Logic, LLM calls, Hybrid Search
├── view_model.py       # CORE: State Management, Bridge to UI
├── model_db.py         # DB: Qdrant interactions & Permissions
├── user_manager.py     # AUTH: User Login/Register logic
├── history_manager.py  # MEMORY: Save/Load chat JSONs
├── data_types.py       # SHARED: Data classes (ChatMessage, SourceNode)
├── users_db.json       # STORAGE: User accounts (Auto-generated)
└── chat_histories/     # STORAGE: Conversation logs

```

---

## 🔮 Future Roadmap
* [ ] **Streaming Responses:** Enable word-by-word token streaming for faster UX.
* [ ] **GUI Integration:** Connect `ChatViewModel` to a Tkinter or Streamlit frontend.
* [ ] **Citation Highlighting:** Return exact page numbers and paragraphs for evidence.

```

```
