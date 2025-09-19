# streamlit_app.py
import os
import streamlit as st
import json
from subprocess import run

USE_OPENAI = bool(os.getenv("OPENAI_API_KEY"))

st.set_page_config(page_title="AI Prompt Engineering Portfolio", layout="wide")
st.title("AI Prompt Engineering Portfolio — Demo")

tabs = st.tabs(["One-pager generator", "Embeddings + RAG demo", "About"])

with tabs[0]:
    st.header("One-pager generator (sample)")
    col1, col2 = st.columns(2)
    with col1:
        project_name = st.text_input("Project / Product name", "Sales Assistant Agent")
        audience = st.text_input("Audience", "Sales managers")
        pain_points = st.text_area("Pain points (comma separated)", "slow follow-ups, low conversion")
    with col2:
        goals = st.text_area("Goals (comma separated)", "increase conversions, faster responses")
        tone = st.selectbox("Tone", ["Professional", "Conversational", "Concise"], index=0)
        max_length = st.slider("Max lines", 5, 30, 12)
    if st.button("Generate one-pager"):
        pp_list = [p.strip() for p in pain_points.split(",") if p.strip()]
        goal_list = [g.strip() for g in goals.split(",") if g.strip()]
        bullets = []
        bullets.append(f"Project: {project_name}")
        bullets.append(f"Audience: {audience}")
        bullets.append("Problem statements:")
        for p in pp_list:
            bullets.append(f"- {p}")
        bullets.append("Goals:")
        for g in goal_list:
            bullets.append(f"- {g}")
        bullets.append(f"Tone: {tone}")
        output = "\\n".join(bullets[:max_length])
        st.code(output)

with tabs[1]:
    st.header("Embeddings + RAG (minimal demo)")
    st.write("This demo uses rag/sample_docs.md and builds a small JSON vector store.")
    if st.button("(Re)build index"):
        r = run(["python", "rag/ingest.py"], capture_output=True, text=True)
        st.text(r.stdout)
        if r.stderr:
            st.error(r.stderr)
    query = st.text_input("Enter a user query for RAG", "How does the Sales Assistant work?")
    if st.button("Query RAG"):
        idx_path = "rag/vector_store.json"
        if not os.path.exists(idx_path):
            st.warning("Index not found. Click (Re)build index first.")
        else:
            with open(idx_path, "r", encoding="utf-8") as f:
                store = json.load(f)
            def sim(a,b):
                return sum(x*y for x,y in zip(a,b))
            if USE_OPENAI:
                try:
                    import openai
                    openai.api_key = os.getenv("OPENAI_API_KEY")
                    emb = openai.Embedding.create(model="text-embedding-3-small", input=query)["data"][0]["embedding"]
                except Exception as e:
                    st.error(f"OpenAI embedding error: {e}")
                    emb = [0.0]*len(store[0]["embedding"])
            else:
                emb = [float((ord(c) % 97)/97.0) for c in query][: len(store[0]["embedding"])]
                if len(emb) < len(store[0]["embedding"]):
                    emb = emb + [0.0] * (len(store[0]["embedding"]) - len(emb))
            scored = []
            for doc in store:
                score = sim(emb, doc["embedding"])
                scored.append((score, doc))
            scored.sort(reverse=True, key=lambda x: x[0])
            top = scored[:3]
            st.subheader("Top documents")
            for score, doc in top:
                st.markdown(f"**Score:** {score:.4f} — **Source:** {doc.get('source','sample')}")
                st.write(doc["text"][:1000])

with tabs[2]:
    st.header("About")
    st.write("Prototype: one-pager generator + minimal RAG demo. Edit rag/ingest.py to plug in real embedding provider.")
