"""
RAG – Étape 2 : Retrieval + Génération (RAG complet)
=====================================================

❝ Le RAG, c'est un assistant qui consulte vos documents avant de répondre.
  Il ne "sait" rien par lui-même sur votre projet — il cherche, puis synthétise. ❞

Pipeline complet :
  1. Question de l'utilisateur
  2. Embedding de la question (sentence-transformers, local)
  3. Recherche des N chunks les plus proches dans Qdrant
  4. Injection des chunks dans le prompt du LLM (Groq / Llama 3.1 8B)
  5. Réponse générée, fondée sur VOS documents

Mode sandbox partagé :
  → Qdrant Cloud  : base vectorielle accessible à tous les devs
  → Groq API      : LLM gratuit, sans GPU, <1 seconde de latence
  → Configurer via RAG_POC/.env (voir .env.example)
"""

from sentence_transformers import SentenceTransformer
from groq import Groq

import rag_config as cfg


# ── Prompt système ──────────────────────────────────────────────────────────

SYSTEM_PROMPT = """Tu es un assistant expert sur le projet "Atelier ML — Détection d'incohérences de prix".
Tu réponds uniquement à partir des extraits de documentation fournis dans le contexte.

Règles :
- Si le contexte contient la réponse, réponds précisément en citant la source.
- Si le contexte ne contient pas la réponse, dis-le clairement : "Je n'ai pas trouvé cette information dans les documents indexés."
- Réponds en français, de façon concise et structurée.
- Ne fabrique pas d'information absente du contexte."""


class RAG_ETAPE2_Generation:

    def __init__(self, top_k: int = 4):
        self.top_k        = top_k
        self.embed_model  = None
        self.qdrant       = None
        self.groq_client  = None

        print("=" * 60)
        print("  RAG – Étape 2 : Retrieval + Génération")
        print(f"  Mode : {cfg.RAG_MODE.upper()}")
        print("=" * 60)

    def init(self):
        """Initialise les trois composants du RAG."""
        print(f"\n🔢 Chargement embedding : {cfg.EMBEDDING_MODEL}")
        self.embed_model = SentenceTransformer(cfg.EMBEDDING_MODEL)

        print(f"🗃️  Connexion Qdrant ({cfg.RAG_MODE})...")
        self.qdrant = cfg.get_qdrant_client()
        count = self.qdrant.count(cfg.COLLECTION_NAME).count
        print(f"   → {count} chunks indexés dans '{cfg.COLLECTION_NAME}'")

        print("🤖 Connexion Groq (Llama 3.1 8B)...")
        if not cfg.GROQ_API_KEY:
            raise ValueError("GROQ_API_KEY manquant dans .env")
        self.groq_client = Groq(api_key=cfg.GROQ_API_KEY)
        print("   → OK\n")

    # ── Retrieval ─────────────────────────────────────────────────────────────

    def retrieve(self, question: str) -> list[dict]:
        """Recherche les chunks les plus pertinents pour la question."""
        query_vector = self.embed_model.encode(question).tolist()

        hits = self.qdrant.search(
            collection_name=cfg.COLLECTION_NAME,
            query_vector=query_vector,
            limit=self.top_k,
            with_payload=True,
        )

        return [
            {
                "score":  round(h.score, 3),
                "source": h.payload["source"],
                "tag":    h.payload["tag"],
                "text":   h.payload["text"],
            }
            for h in hits
        ]

    # ── Génération ────────────────────────────────────────────────────────────

    def generate(self, question: str, chunks: list[dict]) -> str:
        """Génère une réponse via Groq à partir des chunks récupérés."""
        context_parts = []
        for i, c in enumerate(chunks, 1):
            context_parts.append(
                f"[Extrait {i} — {c['source']} | score: {c['score']}]\n{c['text']}"
            )
        context = "\n\n".join(context_parts)

        user_message = f"""CONTEXTE (extraits de la documentation du projet) :
{context}

QUESTION : {question}"""

        response = self.groq_client.chat.completions.create(
            model=cfg.GROQ_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": user_message},
            ],
            temperature=0.1,
            max_tokens=800,
        )

        return response.choices[0].message.content

    # ── Pipeline principal ───────────────────────────────────────────────────

    def ask(self, question: str, verbose: bool = True) -> str:
        """Point d'entrée unique : question → réponse."""
        if verbose:
            print(f"\n❓ Question : {question}")
            print("─" * 50)

        chunks = self.retrieve(question)

        if verbose:
            print(f"🔍 {len(chunks)} chunks récupérés :")
            for c in chunks:
                print(f"   [{c['score']}] {c['source']} (tag: {c['tag']})")
                print(f"         {c['text'][:80].replace(chr(10), ' ')}...")
            print()

        answer = self.generate(question, chunks)

        if verbose:
            print(f"💬 Réponse :\n{answer}\n")

        return answer

    def demo(self):
        """Lance une démonstration avec 4 questions types."""
        demo_questions = [
            "Comment fonctionne le warm start dans XGBoost ?",
            "Quelle est la différence entre ETAPE2 et ETAPE3 ?",
            "Qu'est-ce que le seuil de confiance et pourquoi est-il utile ?",
            "Comment le modèle détecte-t-il l'origine EVOLLIS ou HYBRIS ?",
        ]

        print("\n" + "=" * 60)
        print("  DÉMONSTRATION RAG — 4 questions sur le projet")
        print("=" * 60)

        for q in demo_questions:
            self.ask(q)
            print("=" * 60)


# ── Point d'entrée ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    rag = RAG_ETAPE2_Generation(top_k=4)
    rag.init()

    import sys
    if len(sys.argv) > 1:
        question = " ".join(sys.argv[1:])
        rag.ask(question)
    else:
        rag.demo()
