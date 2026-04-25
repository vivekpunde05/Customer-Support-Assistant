# 🤖 RAG Customer Support Bot

A production-ready AI customer support system built with **LangGraph**, **Groq (free tier)**, **ChromaDB**, and **Streamlit**. Routes every user query through an intent-aware workflow — answering from a PDF knowledge base, responding directly, or escalating to a human agent when needed.

> No OpenAI. No local GPU. Runs entirely on free cloud APIs and local embeddings.

---

## ✨ Features

- **PDF Knowledge Base** — ingest any PDF and query it with semantic search
- **Intent Classification** — two-pass system: keyword rules + LLM classification
- **Smart Routing** — three paths: RAG retrieval, direct LLM answer, or HITL escalation
- **Human-in-the-Loop (HITL)** — auto-escalates sensitive queries with tracked tickets
- **LangGraph Workflow** — stateful, inspectable graph with full execution trace
- **Free Stack** — Groq (free tier LLM) + HuggingFace local embeddings + ChromaDB
- **Streamlit UI** — chat interface with live workflow trace visualisation

---

## 🏗️ Architecture

```
User Query
    │
    ▼
┌─────────────────────┐
│   LangGraph Input   │  ← Logs query, seeds AgentState
└──────────┬──────────┘
           │
    ▼
┌─────────────────────┐
│  Intent Classifier  │  ← Keyword rules → Groq/llama LLM
└──────────┬──────────┘
           │
    ┌──────┴──────┐──────────────┐
    ▼             ▼              ▼
┌────────┐  ┌─────────┐  ┌───────────┐
│  RAG   │  │ Direct  │  │   HITL    │
│  Node  │  │  Node   │  │   Node    │
└────┬───┘  └────┬────┘  └─────┬─────┘
     │            │             │
     └────────────┴─────────────┘
                  │
           ▼
    ┌─────────────┐
    │ Output Node │  ← Format answer, update history
    └─────────────┘
           │
           ▼
     Response to User
```

### Routing Logic

| Intent | Condition | Route |
|---|---|---|
| `document_query` | High confidence | RAG node |
| `general_chat` | Greeting / casual | Direct node |
| `out_of_scope` | Unrelated topic | Direct node |
| `escalation` | Keyword match | HITL node |
| Any | Confidence < 0.6 | HITL node |

---

## 📁 Project Structure

```
.
├── app.py                  # Streamlit chat UI
├── main.py                 # CLI entry point (build / chat / demo)
├── config.py               # All configuration and env vars
├── requirements.txt
├── data/
│   └── sample_knowledge_base.pdf   # Your PDF knowledge base
├── vectorstore/            # Auto-created ChromaDB store
├── modules/
│   ├── graph_workflow.py   # LangGraph graph definition
│   ├── intent_classifier.py
│   ├── rag_engine.py
│   ├── vector_store.py
│   ├── pdf_processor.py
│   ├── hitl_manager.py
│   └── llm_client.py
└── escalation_tickets.json # Auto-created HITL ticket log
```

---

## 🚀 Quick Start

### 1. Clone the repository

```bash
git clone https://github.com/your-username/rag-support-bot.git
cd rag-support-bot
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Get a free Groq API key

1. Go to [console.groq.com/keys](https://console.groq.com/keys)
2. Sign up with Google or GitHub (no credit card required)
3. Create an API key (starts with `gsk_`)

### 4. Configure environment

```bash
# Create .env file
echo "GROQ_API_KEY=gsk_your_key_here" > .env
```

### 5. Add your PDF knowledge base

Place your PDF at:
```
data/sample_knowledge_base.pdf
```

If no PDF is provided, a sample knowledge base (TechCorp policies) is automatically created as a text file fallback during the build step.

### 6. Build the vector store

```bash
python main.py --build
```

This processes the PDF, creates embeddings, and saves the ChromaDB index. Run once.

**First time?** Run the setup wizard to verify your configuration:
```bash
python main.py --setup
```

**Check API connectivity:**
```bash
python main.py --check
```

### 7. Start chatting

**Streamlit UI (recommended):**
```bash
streamlit run app.py
```

**CLI mode:**
```bash
python main.py --chat
```

**Run demo queries:**
```bash
python main.py --demo
```

---

## ⚙️ Configuration

All settings live in `config.py` and can be overridden via environment variables:

| Variable | Default | Description |
|---|---|---|
| `GROQ_API_KEY` | — | Required. Get at console.groq.com |
| `GROQ_MODEL` | `llama-3.1-8b-instant` | Groq model name |
| `CHUNK_SIZE` | `500` | PDF chunk size (tokens) |
| `CHUNK_OVERLAP` | `50` | Chunk overlap |
| `TOP_K_RETRIEVAL` | `3` | Documents retrieved per query |
| `SIMILARITY_THRESHOLD` | `0.7` | Minimum relevance score for retrieved chunks |
| `HITL_CONFIDENCE_THRESHOLD` | `0.6` | Below this → escalate |

### Alternative Groq models

```python
# In config.py or .env
GROQ_MODEL=mixtral-8x7b-32768     # Larger, slower
GROQ_MODEL=llama3-70b-8192        # Best quality
GROQ_MODEL=gemma-7b-it            # Lightweight
```

### HITL escalation keywords

Edit `HITL_ESCALATION_KEYWORDS` in `config.py` to customise what triggers human escalation:

```python
HITL_ESCALATION_KEYWORDS = [
    "refund", "complaint", "lawsuit", "legal", "fraud",
    "unauthorized charge", "account hacked", "data breach",
    "speak to human", "talk to agent", "manager", "supervisor"
]
```

---

## 🧠 How It Works

### Intent Classification (two-pass)

```
Query
  │
  ├─ Keyword check (fast, deterministic)
  │     └─ Escalation keyword? → ESCALATION (confidence: 0.95)
  │     └─ Greeting pattern?   → GENERAL_CHAT (confidence: 0.90)
  │     └─ Default             → DOCUMENT_QUERY (confidence: 0.50)
  │
  └─ LLM classification (if keyword confidence is low)
        └─ Groq/llama classifies into one of four intents
        └─ Returns confidence score
        └─ If confidence < threshold → escalate regardless of intent
```

### RAG Pipeline

```
Query → Embed (HuggingFace) → ChromaDB similarity search
      → Top-K chunks → Format context → Groq LLM → Answer
```

### HITL Escalation

When a query is escalated:
1. A ticket is created with a unique ID and logged to `escalation_tickets.json`
2. A simulated human agent response is generated
3. The user receives the response + ticket reference number
4. Tickets can be resolved, tracked, and rated via `HITLManager`

---

## 📊 Example Interactions

| Query | Route | Behaviour |
|---|---|---|
| `"What is the return policy?"` | RAG | Retrieves policy chunk, generates answer |
| `"How much is express shipping?"` | RAG | Looks up pricing table from PDF |
| `"Hello! How are you?"` | Direct | Friendly greeting, no retrieval |
| `"What's the weather today?"` | Direct | Politely declines (out of scope) |
| `"I want a refund and I'm calling my lawyer!"` | HITL | Creates ticket, escalates |
| `"My account was hacked!"` | HITL | Security escalation with ticket |

---

## 🔧 Module Reference

### `graph_workflow.py`
Defines the LangGraph `StateGraph` with all nodes and conditional edges. Call `create_agent()` to get a compiled, memory-checkpointed agent.

### `intent_classifier.py`
`IntentClassifier.classify(query)` → returns `{intent, confidence, reason, method}`. `should_escalate()` checks confidence threshold and intent type.

### `rag_engine.py`
`RAGEngine.generate_rag_answer(query, documents)` — uses retrieved chunks as context. `generate_direct_answer(query)` — no retrieval, direct LLM response.

### `vector_store.py`
`VectorStoreManager` wraps ChromaDB. `create_vectorstore(chunks)` on first run. `load_vectorstore()` + `similarity_search(query, k)` on subsequent runs.

### `pdf_processor.py`
`PDFProcessor.process(pdf_path)` → loads PDF pages and splits into overlapping chunks using `RecursiveCharacterTextSplitter`.

### `hitl_manager.py`
`HITLManager.create_ticket()` creates and persists escalation tickets. `resolve_ticket()`, `collect_feedback()`, and `get_pending_tickets()` for ticket lifecycle management.

### `llm_client.py`
`UnifiedLLM` is a LangChain-compatible wrapper around the Groq client. `get_llm()` returns a ready-to-use instance. `check_setup()` validates API key availability.

---

## 🧪 Running Tests

```bash
# Run setup wizard (first-time users)
python main.py --setup

# Check API connectivity
python main.py --check

# Run all demo routing paths
python main.py --demo

# Test individual modules
python modules/intent_classifier.py
python modules/vector_store.py
python modules/llm_client.py
```

---

## 📦 Requirements

```
langchain>=0.2.0
langchain-community>=0.2.0
langchain-core>=0.2.0
langchain-groq>=0.1.0
langchain-text-splitters>=0.2.0
langgraph>=0.0.20
chromadb>=0.4.18
pypdf>=3.17.0
sentence-transformers>=2.2.2
groq>=0.4.0
python-dotenv>=1.0.0
requests>=2.31.0
streamlit>=1.30.0
```

Python 3.9+ recommended.

---

## 🛣️ Roadmap

- [ ] Multi-PDF ingestion support
- [ ] Real human agent webhook integration
- [ ] Feedback loop for continuous improvement
- [ ] REST API endpoint (FastAPI)
- [ ] Docker deployment
- [ ] Evaluation harness with ground-truth QA pairs

---

## 🤝 Contributing

Pull requests are welcome. For major changes, open an issue first to discuss what you'd like to change.

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.

---

## 🙏 Acknowledgements

- [LangChain](https://github.com/langchain-ai/langchain) — LLM orchestration
- [LangGraph](https://github.com/langchain-ai/langgraph) — stateful agent graphs
- [Groq](https://groq.com) — blazing fast free-tier LLM inference
- [ChromaDB](https://www.trychroma.com) — open-source vector database
- [Sentence Transformers](https://www.sbert.net) — free local embeddings
