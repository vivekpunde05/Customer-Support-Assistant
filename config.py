"""
Configuration for RAG Customer Support Bot
Uses FREE cloud APIs - no OpenAI, no local models needed
"""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
VECTORSTORE_DIR = BASE_DIR / "vectorstore"
VECTORSTORE_DIR.mkdir(parents=True, exist_ok=True)

PDF_PATH = DATA_DIR / "sample_knowledge_base.pdf"

CHUNK_SIZE = 500
CHUNK_OVERLAP = 50

# ─── EMBEDDINGS: Free local (lightweight, no GPU needed) ───────────
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# ─── LLM: FREE CLOUD OPTIONS ────────────────────────────────────────
# PRIMARY: Groq API (FREE tier - 20 requests/min, no credit card)
# Get key at: https://console.groq.com/keys (instant, no approval)
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")

# Model selection (all free on Groq)
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
# Alternatives: "mixtral-8x7b-32768", "gemma-7b-it", "llama3-70b-8192"

TOP_K_RETRIEVAL = 3
SIMILARITY_THRESHOLD = 0.7

HITL_CONFIDENCE_THRESHOLD = 0.6
HITL_ESCALATION_KEYWORDS = [
    "refund", "complaint", "lawsuit", "legal", "fraud",
    "unauthorized charge", "account hacked", "data breach",
    "speak to human", "talk to agent", "manager", "supervisor"
]