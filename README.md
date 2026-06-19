# Local-RAG-System

A Retrieval-Augmented Generation (RAG) system built 100% locally using HuggingFace embeddings and Ollama LLM — zero API costs!

## Project Structure

| File / Folder | Description |
|---------------|-------------|
| `ingestion_pipeline.py` | Load docs, chunk, embed, store in ChromaDB |
| `retrieval_pipeline.py` | Search vector DB for relevant chunks |
| `answer_generation.py` | Retrieve chunks + generate answer with LLM |
| `context_aware_generation.py` | Conversational RAG with memory |
| `docs/` | Your .txt documents |
| `db/chroma_db/` | Vector database (auto-generated) |
| `dpndcs/` | Virtual environment (ignored by Git) |
| `.env` | Environment variables (ignored by Git) |

## Tech Stack

| Component | Technology |
|-----------|------------|
| Embeddings | HuggingFace `sentence-transformers/all-MiniLM-L6-v2` |
| Vector DB | ChromaDB |
| LLM | Ollama (TinyLlama → upgrading to Llama 3.2) |
| Framework | LangChain |
| Language | Python 3.x |

## Evolution

| Phase | Description | Status |
|-------|-------------|--------|
| 1 | Basic ingestion, retrieval, generation | ✅ Complete |
| 2 | Context-aware / conversational RAG | ✅ Complete |
| 3 | Semantic chunking | 🔄 Planned |
| 4 | Multi-query retrieval | ⏳ Planned |
| 5 | Hybrid search (BM25 + vector) | ⏳ Planned |
| 6 | Reranking | ⏳ Planned |
| 7 | Multi-modal RAG | ⏳ Planned |

## Why Local?

- **Zero API costs** — no OpenAI bills!
- **Privacy** — your data never leaves your machine
- **Offline capable** — works without internet after setup
- **Scalable** — upgrade models as your hardware improves

## Quick Start

```bash
# 1. Activate virtual environment
.\dpndcs\Scripts\activate

# 2. Run ingestion (creates vector DB)
python ingestion_pipeline.py

# 3. Ask questions!
python answer_generation.py