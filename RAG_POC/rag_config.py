"""
Configuration centralisée du RAG POC.
Trois modes :
  - local  : Qdrant sur disque local (développement solo, pas de réseau)
  - server : Qdrant en Docker sur le serveur Hetzner (sandbox partagé, recommandé)
  - cloud  : Qdrant Cloud SaaS + clé API (optionnel, si pas de serveur propre)
"""

import os
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent / ".env")
except ImportError:
    pass

RAG_MODE = os.getenv("RAG_MODE", "local")

# ── Qdrant ────────────────────────────────────────────────────
QDRANT_LOCAL_PATH = str(Path(__file__).parent / "qdrant_db")
QDRANT_URL        = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_API_KEY    = os.getenv("QDRANT_API_KEY", "")   # vide en mode server

COLLECTION_NAME   = "atelier_ml_knowledge"

# ── Embedding (toujours local, CPU, ~90 MB) ───────────────────
EMBEDDING_MODEL   = "paraphrase-multilingual-MiniLM-L12-v2"
VECTOR_SIZE       = 384

# ── LLM (Groq — gratuit, sans GPU, multilingue) ───────────────
GROQ_API_KEY      = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL        = "llama-3.1-8b-instant"


def get_qdrant_client():
    """Retourne un client Qdrant selon le mode configuré."""
    from qdrant_client import QdrantClient

    if RAG_MODE == "local":
        return QdrantClient(path=QDRANT_LOCAL_PATH)

    if RAG_MODE == "server":
        # Qdrant tourne dans Docker sur le même serveur (réseau interne Docker)
        # QDRANT_URL = http://qdrant:6333 (défini dans docker-compose.yml)
        return QdrantClient(url=QDRANT_URL)

    if RAG_MODE == "cloud":
        if not QDRANT_URL or not QDRANT_API_KEY:
            raise ValueError("Mode cloud : QDRANT_URL et QDRANT_API_KEY requis dans .env")
        return QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)

    raise ValueError(f"RAG_MODE inconnu : '{RAG_MODE}'. Valeurs valides : local, server, cloud")
