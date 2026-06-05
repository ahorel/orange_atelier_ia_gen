"""
Interface Streamlit — RAG Sandbox partagé
Lancer avec : streamlit run RAG_POC/streamlit_app.py
"""

import streamlit as st
from RAG_ETAPE2_generation import RAG_ETAPE2_Generation

st.set_page_config(
    page_title="RAG — Atelier ML",
    page_icon="🤖",
    layout="centered",
)

st.title("🤖 RAG — Atelier ML Prix")
st.caption("Posez une question sur le projet · Powered by Qdrant Cloud + Llama 3.1 8B (Groq)")

# ── Init du RAG (mis en cache pour ne pas recharger à chaque interaction) ──

@st.cache_resource(show_spinner="Chargement du RAG...")
def load_rag():
    rag = RAG_ETAPE2_Generation(top_k=4)
    rag.init()
    return rag

try:
    rag = load_rag()
except Exception as e:
    st.error(f"Erreur d'initialisation : {e}")
    st.info("Vérifiez les variables QDRANT_URL, QDRANT_API_KEY et GROQ_API_KEY dans .env")
    st.stop()

# ── Zone de saisie ─────────────────────────────────────────────────────────

with st.form("question_form"):
    question = st.text_area(
        "Votre question",
        placeholder="Ex : Comment fonctionne le warm start ? Quelle est la différence entre ETAPE2 et ETAPE3 ?",
        height=100,
    )
    top_k = st.slider("Nombre de chunks récupérés (top-k)", 2, 8, 4)
    submitted = st.form_submit_button("Poser la question", type="primary")

# ── Questions exemples ─────────────────────────────────────────────────────

with st.expander("Exemples de questions"):
    exemples = [
        "Comment fonctionne le warm start dans XGBoost ?",
        "Quelle est la différence entre ETAPE2 et ETAPE3 ?",
        "Qu'est-ce que le seuil de confiance ?",
        "Comment le modèle détecte-t-il l'origine EVOLLIS ou HYBRIS ?",
        "Qu'est-ce que le gap_evollis ?",
        "Comment fonctionne l'hyperparameter tuning ?",
    ]
    for ex in exemples:
        if st.button(ex, key=ex):
            question = ex
            submitted = True

# ── Traitement ─────────────────────────────────────────────────────────────

if submitted and question.strip():
    rag.top_k = top_k

    with st.spinner("Recherche et génération en cours..."):
        chunks = rag.retrieve(question)
        answer = rag.generate(question, chunks)

    st.subheader("Réponse")
    st.markdown(answer)

    with st.expander(f"Sources utilisées ({len(chunks)} chunks)"):
        for i, c in enumerate(chunks, 1):
            st.markdown(f"**[{i}] `{c['source']}` — score : `{c['score']}`**")
            st.code(c["text"][:300] + ("..." if len(c["text"]) > 300 else ""), language="text")

elif submitted:
    st.warning("Entrez une question.")
