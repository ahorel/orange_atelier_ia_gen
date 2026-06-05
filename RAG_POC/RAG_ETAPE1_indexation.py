"""
RAG – Étape 1 : Ingestion, Chunking, Embedding, Indexation
============================================================

❝ Un RAG (Retrieval-Augmented Generation) ne retient pas vos documents.
  Il les transforme en vecteurs numériques pour pouvoir les retrouver
  instantanément par similarité de sens, pas par mots-clés. ❞

Objectif de cette étape :
  1. Charger les documents du projet (README, commentaires, scripts Python)
  2. Découper chaque document en "chunks" cohérents
  3. Générer un embedding (vecteur numérique) par chunk via sentence-transformers
  4. Stocker tout ça dans Qdrant (base vectorielle locale, mode persistant)
  5. Vérifier l'indexation avec des requêtes de test

Pourquoi Qdrant ?
  → Base vectorielle open-source, performante, pensée pour la production
  → Mode local sans serveur (qdrant_client >= 1.7 inclut un moteur embarqué)
  → API claire : collections, points, payloads, recherche par cosinus
  → Migration facile vers un serveur Qdrant Cloud sans changer le code

Les collaborateurs pourront ensuite interroger cette base :
  → "Comment fonctionne le warm start ?"
  → "Quelle est la différence entre ETAPE2 et ETAPE3 ?"
  → "Qu'est-ce que le gap_evollis ?"
"""

import os
import re
import uuid
from pathlib import Path

from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

import rag_config as cfg


# ============================================================
# CONFIGURATION
# ============================================================

WORKSPACE_ROOT = Path(__file__).parent.parent

# Documents à indexer (relatifs à la racine du workspace)
DOCUMENT_SOURCES = [
    {"path": "Readme.MD",                                   "type": "documentation", "tag": "readme"},
    {"path": ".github/copilot-instructions.md",             "type": "documentation", "tag": "instructions"},
    # Commentaires pédagogiques — session 1
    {"path": "commentaires/commentaires_priceInconcistencyWorkshop.txt",
                                                            "type": "commentaire",   "tag": "etape1"},
    {"path": "commentaires/commentaires_priceInconcistencyWorkshop_noRuleBasedOrigin.txt",
                                                            "type": "commentaire",   "tag": "etape2"},
    {"path": "commentaires/commentaires_priceInconcistencyWorkshop_noRuleBased_auglentedFiles.txt",
                                                            "type": "commentaire",   "tag": "etape3"},
    {"path": "commentaires/commentaires_priceInconcistencyWorkshop_noRuleBased_augmentedFiles_warmStart.txt",
                                                            "type": "commentaire",   "tag": "etape4"},
    # Commentaires pédagogiques — session 2
    {"path": "commentaires/commentaires_ETAPE6_featureEngineering.txt",
                                                            "type": "commentaire",   "tag": "etape6"},
    {"path": "commentaires/commentaires_ETAPE7_metrics.txt",
                                                            "type": "commentaire",   "tag": "etape7"},
    {"path": "commentaires/commentaires_ETAPE8_confidenceThreshold.txt",
                                                            "type": "commentaire",   "tag": "etape8"},
    {"path": "commentaires/commentaires_ETAPE9_hyperparamTuning.txt",
                                                            "type": "commentaire",   "tag": "etape9"},
    {"path": "commentaires/commentaires_ETAPE10_explainability.txt",
                                                            "type": "commentaire",   "tag": "etape10"},
    # Scripts ML (déplacés dans ML_ETAPES/)
    {"path": "ML_ETAPES/ETAPE1_modelML_supervise.py",             "type": "code", "tag": "etape1"},
    {"path": "ML_ETAPES/ETAPE2_modelML_noRuleBased.py",           "type": "code", "tag": "etape2"},
    {"path": "ML_ETAPES/ETAPE3_modelML_augmentedTrainingFile.py", "type": "code", "tag": "etape3"},
    {"path": "ML_ETAPES/ETAPE4_modelML_warmStart.py",             "type": "code", "tag": "etape4"},
    {"path": "ML_ETAPES/ETAPE5_modelML_warmstart.py",             "type": "code", "tag": "etape5"},
    {"path": "ML_ETAPES/ETAPE6_modelML_featureEngineering.py",    "type": "code", "tag": "etape6"},
    {"path": "ML_ETAPES/ETAPE7_modelML_metrics.py",               "type": "code", "tag": "etape7"},
    {"path": "ML_ETAPES/ETAPE8_modelML_confidenceThreshold.py",   "type": "code", "tag": "etape8"},
    {"path": "ML_ETAPES/ETAPE9_modelML_hyperparamTuning.py",      "type": "code", "tag": "etape9"},
    {"path": "ML_ETAPES/ETAPE10_modelML_explainability.py",       "type": "code", "tag": "etape10"},
    # RAG POC lui-même
    {"path": "RAG_POC/commentaire_RAG_etape1.txt",                "type": "commentaire", "tag": "rag"},
]

# Paramètres de chunking
CHUNK_SIZE    = 500   # Taille cible d'un chunk (en caractères)
CHUNK_OVERLAP = 100   # Chevauchement entre chunks pour préserver le contexte

# Qdrant — local ou cloud selon RAG_MODE dans .env
COLLECTION_NAME   = cfg.COLLECTION_NAME

# Modèle d'embedding : multilingue (FR + EN + 50 langues), ~90 MB, 100% local
EMBEDDING_MODEL   = cfg.EMBEDDING_MODEL
VECTOR_SIZE       = cfg.VECTOR_SIZE

# Batch d'indexation (évite les pics mémoire sur de gros corpus)
BATCH_SIZE = 64


# ============================================================
# CLASSE PRINCIPALE
# ============================================================

class RAG_ETAPE1_Indexation:

    def __init__(self):
        self.documents      = []   # Documents bruts chargés
        self.chunks         = []   # Chunks {id, text, metadata}
        self.qdrant_client  = None
        self.embed_model    = None

        print("=" * 60)
        print("  RAG – Étape 1 : Indexation Qdrant")
        print("=" * 60)

    # ----------------------------------------------------------
    # 1. CHARGEMENT DES DOCUMENTS
    # ----------------------------------------------------------
    def load_documents(self):
        """Lit les fichiers texte depuis le workspace."""
        print("\n📂 Chargement des documents...")

        for source in DOCUMENT_SOURCES:
            full_path = WORKSPACE_ROOT / source["path"]

            if not full_path.exists():
                print(f"   ⚠️  Introuvable : {source['path']}")
                continue

            try:
                content = full_path.read_text(encoding="utf-8", errors="ignore")
                self.documents.append({
                    "path":    source["path"],
                    "content": content,
                    "type":    source["type"],
                    "tag":     source["tag"],
                })
                print(f"   ✅ {source['path']} ({len(content)} car.)")
            except Exception as e:
                print(f"   ❌ {source['path']} : {e}")

        print(f"\n   → {len(self.documents)} documents chargés.")

    # ----------------------------------------------------------
    # 2. CHUNKING
    # ----------------------------------------------------------
    def chunk_documents(self):
        """
        Découpe chaque document en morceaux (chunks) cohérents.

        Stratégie :
          1. Découpe sur les sauts de ligne doubles (paragraphes)
          2. Regroupe les paragraphes jusqu'à CHUNK_SIZE caractères
          3. Ajoute CHUNK_OVERLAP pour ne pas couper une idée en deux

        Qdrant exige un identifiant unique par point.
        On utilise un UUID déterministe (uuid5) basé sur le texte du chunk
        pour que la ré-indexation soit idempotente.
        """
        print("\n✂️  Découpage en chunks...")

        for doc in self.documents:
            paragraphs = self._split_into_paragraphs(doc["content"])
            doc_chunks = self._merge_paragraphs_into_chunks(paragraphs)

            for i, chunk_text in enumerate(doc_chunks):
                # UUID déterministe → même chunk = même ID à chaque run
                chunk_id = str(uuid.uuid5(
                    uuid.NAMESPACE_DNS,
                    f"{doc['path']}::{i}::{chunk_text[:64]}"
                ))
                self.chunks.append({
                    "id":        chunk_id,
                    "text":      chunk_text,
                    "source":    doc["path"],
                    "type":      doc["type"],
                    "tag":       doc["tag"],
                    "chunk_idx": i,
                })

        print(f"   → {len(self.chunks)} chunks créés.")

    def _split_into_paragraphs(self, text: str) -> list[str]:
        paragraphs = re.split(r"\n{2,}", text)
        return [p.strip() for p in paragraphs if p.strip()]

    def _merge_paragraphs_into_chunks(self, paragraphs: list[str]) -> list[str]:
        """Regroupe les paragraphes en chunks avec overlap."""
        chunks      = []
        current     = ""
        overlap_buf = ""

        for para in paragraphs:
            if len(current) + len(para) + 1 <= CHUNK_SIZE:
                current += ("\n" if current else "") + para
            else:
                if current:
                    chunks.append(current)
                    overlap_buf = current[-CHUNK_OVERLAP:] if len(current) > CHUNK_OVERLAP else current
                current = overlap_buf + ("\n" if overlap_buf else "") + para
                overlap_buf = ""

        if current:
            chunks.append(current)

        return chunks if chunks else ["\n".join(paragraphs)]

    # ----------------------------------------------------------
    # 3. INITIALISATION QDRANT
    # ----------------------------------------------------------
    def init_vector_store(self):
        """
        Initialise Qdrant en mode local ou cloud selon cfg.RAG_MODE.

        Architecture Qdrant :
          Collection → ensemble de points (un point = un chunk)
          Point      → { id, vector, payload }
            - id      : UUID unique du chunk
            - vector  : embedding 384 dimensions (float32)
            - payload : texte + métadonnées (source, type, tag)

        La distance COSINE est la plus adaptée aux embeddings textuels :
          score = 1.0 → vecteurs identiques (sens identique)
          score = 0.0 → vecteurs orthogonaux (sens sans rapport)
          score < 0   → opposés (rare en NLP)
        """
        mode_label = cfg.QDRANT_CLOUD_URL if cfg.RAG_MODE == "cloud" else cfg.QDRANT_LOCAL_PATH
        print(f"\n🗃️  Initialisation Qdrant ({cfg.RAG_MODE.upper()}) → {mode_label}")

        self.qdrant_client = cfg.get_qdrant_client()

        # Supprime la collection si elle existe (ré-indexation propre)
        if self.qdrant_client.collection_exists(COLLECTION_NAME):
            self.qdrant_client.delete_collection(COLLECTION_NAME)
            print("   ♻️  Collection existante supprimée (ré-indexation propre).")

        self.qdrant_client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(
                size=VECTOR_SIZE,
                distance=Distance.COSINE
            ),
        )

        print(f"   ✅ Collection '{COLLECTION_NAME}' créée "
              f"(dim={VECTOR_SIZE}, distance=COSINE).")

    # ----------------------------------------------------------
    # 4. EMBEDDINGS + INDEXATION
    # ----------------------------------------------------------
    def index_chunks(self):
        """
        Génère les embeddings et les insère dans Qdrant.

        Pour chaque chunk :
          texte → sentence-transformers → vecteur 384 dims → PointStruct → Qdrant

        Le payload Qdrant contient :
          text      : texte original du chunk (retourné lors de la recherche)
          source    : chemin du fichier source
          type      : "code" | "commentaire" | "documentation"
          tag       : "etape1" ... "etape10" | "readme" | "rag"
          chunk_idx : index du chunk dans le document

        ❝ Le modèle paraphrase-multilingual-MiniLM-L12-v2 comprend
          le français, l'anglais et 50 autres langues.
          "gap_evollis" et "écart EVOLLIS" → vecteurs proches
          → même sens détecté sans correspondance exacte de mots. ❞
        """
        print(f"\n🔢 Génération des embeddings ({len(self.chunks)} chunks)...")
        print(f"   Modèle  : {EMBEDDING_MODEL}")
        print("   (Téléchargement automatique au premier lancement ~90 MB)\n")

        self.embed_model = SentenceTransformer(EMBEDDING_MODEL)

        texts = [c["text"] for c in self.chunks]

        for batch_start in range(0, len(self.chunks), BATCH_SIZE):
            batch_chunks = self.chunks[batch_start:batch_start + BATCH_SIZE]
            batch_texts  = texts[batch_start:batch_start + BATCH_SIZE]

            # sentence-transformers retourne un np.ndarray (n, 384)
            embeddings = self.embed_model.encode(
                batch_texts,
                show_progress_bar=False,
                convert_to_numpy=True,
            )

            points = [
                PointStruct(
                    id=chunk["id"],
                    vector=embedding.tolist(),
                    payload={
                        "text":      chunk["text"],
                        "source":    chunk["source"],
                        "type":      chunk["type"],
                        "tag":       chunk["tag"],
                        "chunk_idx": chunk["chunk_idx"],
                    },
                )
                for chunk, embedding in zip(batch_chunks, embeddings)
            ]

            self.qdrant_client.upsert(
                collection_name=COLLECTION_NAME,
                points=points,
            )

            end = min(batch_start + BATCH_SIZE, len(self.chunks))
            print(f"   → Batch {batch_start // BATCH_SIZE + 1} indexé "
                  f"({end}/{len(self.chunks)})")

        total = self.qdrant_client.count(COLLECTION_NAME).count
        print(f"\n   ✅ {total} points indexés dans Qdrant.")

    # ----------------------------------------------------------
    # 5. VÉRIFICATION : REQUÊTES DE TEST
    # ----------------------------------------------------------
    def verify_index(self):
        """
        Effectue 3 requêtes sémantiques pour valider l'indexation.

        Avant-goût du Step 2 (Retrieval) :
          1. On encode la question → vecteur 384 dims
          2. Qdrant calcule la similarité cosinus avec tous les points
          3. Retourne les N points les plus proches
          Pas de LLM à cette étape, juste de la recherche vectorielle.
        """
        print("\n🔍 Vérification — Requêtes de test (sans LLM)\n")

        test_queries = [
            "Comment fonctionne le warm start dans XGBoost ?",
            "Quelle est la différence entre ETAPE2 et ETAPE3 ?",
            "Comment le modèle détecte-t-il l'origine de l'erreur ?",
            "Qu'est-ce que le seuil de confiance ?",
        ]

        for query in test_queries:
            print(f"  ❓ {query}")

            query_vector = self.embed_model.encode(query).tolist()

            results = self.qdrant_client.search(
                collection_name=COLLECTION_NAME,
                query_vector=query_vector,
                limit=2,
                with_payload=True,
            )

            for j, hit in enumerate(results):
                payload = hit.payload
                score   = round(hit.score, 3)
                extract = payload["text"][:120].replace("\n", " ")
                print(f"     [{j+1}] Score: {score:.3f} | "
                      f"Source: {payload['source']} (tag: {payload['tag']})")
                print(f"          Extrait: {extract}...")
            print()

    # ----------------------------------------------------------
    # PIPELINE PRINCIPAL
    # ----------------------------------------------------------
    def run(self):
        self.load_documents()
        self.chunk_documents()
        self.init_vector_store()
        self.index_chunks()
        self.verify_index()

        location = cfg.QDRANT_CLOUD_URL if cfg.RAG_MODE == "cloud" else cfg.QDRANT_LOCAL_PATH
        print("=" * 60)
        print("  ✅ Étape 1 terminée — Base Qdrant prête")
        print(f"  Mode    : {cfg.RAG_MODE.upper()}")
        print(f"  Qdrant  : {location}")
        print("  🔜 Étape 2 : python RAG_ETAPE2_generation.py")
        print("  🔜 Interface : streamlit run streamlit_app.py")
        print("=" * 60)


# ============================================================
# POINT D'ENTRÉE
# ============================================================

if __name__ == "__main__":
    indexer = RAG_ETAPE1_Indexation()
    indexer.run()
