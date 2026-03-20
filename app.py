# app.py
# Phase 5 — Streamlit UI
#
# Two tabs:
#   1. Ask a Question — chat interface with source citations
#   2. Evaluation Dashboard — RAGAS scores from evaluation_results.json
#
# Run:
#   streamlit run app.py
 
import json
import os
from pathlib import Path
import streamlit as st
from dotenv import load_dotenv
 
load_dotenv()
 
# ── AUTO-BUILD VECTORSTORE ON FIRST RUN ───────────────────────────────────────
# On HuggingFace Spaces the vectorstore doesn't exist yet on first boot.
# This builds it from the PDFs in data/pdfs/ before anything else runs.
if not Path("vectorstore").exists():
    with st.spinner("Building knowledge base from PDFs... (~2 min, first run only)"):
        from ingest import main as ingest_main
        ingest_main()
    st.rerun()
 
# ── PAGE CONFIG ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Public Health Policy QA",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="collapsed",
)
 
# ── CUSTOM CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display&family=DM+Sans:wght@300;400;500&display=swap');
 
  html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
 
  .stApp { background-color: #f7f5f0; }
 
  .app-header {
    background: linear-gradient(135deg, #1a3a2a 0%, #2d5a3d 100%);
    color: white;
    padding: 2rem 2.5rem;
    border-radius: 16px;
    margin-bottom: 1.5rem;
  }
  .app-header h1 {
    font-family: 'DM Serif Display', serif;
    font-size: 2.2rem;
    margin: 0 0 0.4rem 0;
    letter-spacing: -0.5px;
  }
  .app-header p {
    font-size: 0.95rem;
    opacity: 0.8;
    margin: 0;
    font-weight: 300;
  }
 
  .stTabs [data-baseweb="tab-list"] { background: transparent; gap: 8px; }
  .stTabs [data-baseweb="tab"] {
    background: white;
    border-radius: 8px;
    border: 1px solid #e0ddd6;
    padding: 0.5rem 1.2rem;
    font-weight: 500;
    color: #555;
  }
  .stTabs [aria-selected="true"] {
    background: #1a3a2a !important;
    color: white !important;
    border-color: #1a3a2a !important;
  }
 
  .stTextInput > div > div > input {
    border-radius: 10px;
    border: 2px solid #e0ddd6;
    padding: 0.75rem 1rem;
    font-size: 1rem;
    background: white;
    transition: border-color 0.2s;
  }
  .stTextInput > div > div > input:focus {
    border-color: #2d5a3d;
    box-shadow: 0 0 0 3px rgba(45,90,61,0.1);
  }
 
  .answer-box {
    background: white;
    border-left: 4px solid #2d5a3d;
    border-radius: 0 12px 12px 0;
    padding: 1.25rem 1.5rem;
    margin: 1rem 0;
    box-shadow: 0 2px 8px rgba(0,0,0,0.06);
    line-height: 1.75;
  }
 
  .source-card {
    background: #f0f4f1;
    border: 1px solid #c8d9cb;
    border-radius: 10px;
    padding: 0.85rem 1.1rem;
    margin: 0.4rem 0;
    font-size: 0.88rem;
  }
  .source-label { font-weight: 500; color: #1a3a2a; margin-bottom: 0.3rem; }
  .source-snippet { color: #666; font-style: italic; line-height: 1.5; }
 
  .score-card {
    background: white;
    border-radius: 14px;
    padding: 1.5rem;
    text-align: center;
    box-shadow: 0 2px 10px rgba(0,0,0,0.07);
    border-top: 4px solid #2d5a3d;
  }
  .score-value {
    font-family: 'DM Serif Display', serif;
    font-size: 3rem;
    color: #1a3a2a;
    line-height: 1;
    margin: 0.5rem 0;
  }
  .score-label {
    font-size: 0.85rem;
    font-weight: 500;
    text-transform: uppercase;
    letter-spacing: 1px;
    color: #888;
  }
  .score-grade {
    font-size: 0.8rem;
    margin-top: 0.4rem;
    padding: 0.2rem 0.6rem;
    border-radius: 20px;
    display: inline-block;
  }
  .grade-good  { background: #e6f4ea; color: #1e7e34; }
  .grade-ok    { background: #fff3cd; color: #856404; }
  .grade-bad   { background: #fde8e8; color: #c0392b; }
  .grade-na    { background: #f0f0f0; color: #888; }
 
  .history-item {
    background: white;
    border-radius: 10px;
    padding: 1rem 1.2rem;
    margin: 0.5rem 0;
    border: 1px solid #e8e4dd;
  }
  .history-q { font-weight: 500; color: #1a3a2a; font-size: 0.92rem; }
  .history-a { color: #666; font-size: 0.85rem; margin-top: 0.3rem; line-height: 1.4; }
 
  .stButton > button {
    background: #1a3a2a;
    color: white;
    border: none;
    border-radius: 8px;
    padding: 0.5rem 1.5rem;
    font-weight: 500;
    transition: background 0.2s;
  }
  .stButton > button:hover { background: #2d5a3d; }
 
  .info-box {
    background: #eef4f0;
    border-radius: 10px;
    padding: 1rem 1.2rem;
    font-size: 0.88rem;
    color: #444;
    line-height: 1.6;
    margin-bottom: 1rem;
  }
</style>
""", unsafe_allow_html=True)
 
 
# ── LAZY PIPELINE LOADER (cached after first load) ────────────────────────────
@st.cache_resource(show_spinner=False)
def load_pipeline():
    """
    Loads vectorstore + LLM once and caches for the session.
    Called lazily — only when user asks a question, not at startup.
    This prevents the 30-second HuggingFace Spaces health check timeout.
    """
    from rag_pipeline import (
        load_vectorstore,
        build_llm_caller,
        build_rag_chain,
    )
    vectorstore      = load_vectorstore()
    llm_caller       = build_llm_caller()
    chain, retriever = build_rag_chain(vectorstore, llm_caller)
    return chain, retriever
 
 
# ── LOAD EVALUATION RESULTS ───────────────────────────────────────────────────
def load_eval_results():
    if not Path("evaluation_results.json").exists():
        return None
    with open("evaluation_results.json") as f:
        return json.load(f)
 
 
# ── GRADE HELPER ──────────────────────────────────────────────────────────────
def grade(score):
    if score != score or score is None:
        return "~ No data", "grade-na"
    if score >= 0.8:
        return "✓ Good", "grade-good"
    if score >= 0.6:
        return "~ Acceptable", "grade-ok"
    return "✗ Needs work", "grade-bad"
 
 
# ── HEADER ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="app-header">
  <h1>🏥 Public Health Policy QA</h1>
  <p>Ask questions about your public health documents — answers grounded in evidence, with source citations</p>
</div>
""", unsafe_allow_html=True)
 
# ── TABS ──────────────────────────────────────────────────────────────────────
tab1, tab2 = st.tabs(["💬  Ask a Question", "📊  Evaluation Dashboard"])
 
 
# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — CHAT INTERFACE
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    col_main, col_history = st.columns([2, 1])
 
    with col_main:
        st.markdown("### Ask a question")
        st.markdown(
            "Type any question about the public health documents in your "
            "knowledge base. Answers are grounded in source documents only."
        )
 
        question = st.text_input(
            label="Your question",
            placeholder="e.g. What are the main public health recommendations?",
            label_visibility="collapsed",
        )
 
        ask_btn = st.button("Ask →")
 
        # Example questions
        st.markdown("**Try these:**")
        examples = [
            "What risk factors are discussed?",
            "How is health equity defined?",
            "What data collection methods are described?",
        ]
        ex_cols = st.columns(3)
        for i, ex in enumerate(examples):
            if ex_cols[i].button(ex, key=f"ex_{i}", use_container_width=True):
                question = ex
                ask_btn  = True
 
        # ── Run query ──
        if ask_btn and question.strip():
            # Load pipeline lazily — only on first question
            with st.spinner("Loading models... (first question takes ~30s)"):
                chain, retriever = load_pipeline()
 
            from rag_pipeline import answer_question
            with st.spinner("Searching documents and generating answer..."):
                result = answer_question(question.strip(), chain, retriever)
 
            # Answer
            st.markdown("#### Answer")
            st.markdown(
                f'<div class="answer-box">{result["answer"]}</div>',
                unsafe_allow_html=True,
            )
 
            # Sources
            if result["sources"]:
                st.markdown("#### Sources")
                for src in result["sources"]:
                    st.markdown(f"""
                    <div class="source-card">
                      <div class="source-label">
                        📄 {src['source']} — page {src['page']}
                      </div>
                      <div class="source-snippet">"{src['snippet']}"</div>
                    </div>
                    """, unsafe_allow_html=True)
 
            # Save to history
            if "history" not in st.session_state:
                st.session_state.history = []
            st.session_state.history.insert(0, {
                "question": question.strip(),
                "answer":   result["answer"][:180] + "...",
            })
 
        elif ask_btn and not question.strip():
            st.warning("Please enter a question first.")
 
        # Empty state
        if not (ask_btn and question.strip()):
            st.markdown("""
            <div style="text-align:center; padding: 3rem 1rem; color: #aaa;">
              <div style="font-size: 3rem; margin-bottom: 1rem;">🔍</div>
              <div style="font-size: 1rem;">
                Ask any question about your public health documents
              </div>
            </div>
            """, unsafe_allow_html=True)
 
    # History panel
    with col_history:
        st.markdown("### Recent questions")
        history = st.session_state.get("history", [])
        if not history:
            st.markdown(
                '<div class="info-box">Your question history will appear here.</div>',
                unsafe_allow_html=True,
            )
        else:
            for item in history[:8]:
                st.markdown(f"""
                <div class="history-item">
                  <div class="history-q">Q: {item['question'][:70]}...</div>
                  <div class="history-a">{item['answer'][:100]}...</div>
                </div>
                """, unsafe_allow_html=True)
            if st.button("Clear history"):
                st.session_state.history = []
                st.rerun()
 
 
# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — EVALUATION DASHBOARD
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    eval_data = load_eval_results()
 
    if eval_data is None:
        st.warning(
            "No evaluation results found. "
            "Run `python evaluate.py` first to generate scores."
        )
    else:
        ts = eval_data.get("timestamp", "Unknown")
        st.markdown(
            f'<div style="color:#888; font-size:0.85rem; margin-bottom:1rem;">'
            f'Last evaluated: {ts[:19].replace("T", " ")}</div>',
            unsafe_allow_html=True,
        )
 
        st.markdown("### Overall RAGAS scores")
        st.markdown(
            "These three metrics evaluate every stage of the RAG pipeline — "
            "retrieval quality, answer grounding, and response relevance."
        )
 
        overall = eval_data.get("overall", {})
        c1, c2, c3 = st.columns(3)
 
        metrics_info = [
            ("faithfulness",      "Faithfulness",      c1,
             "Answer only uses retrieved context — measures hallucination risk"),
            ("answer_relevancy",  "Answer relevancy",  c2,
             "Answer directly addresses what was asked"),
            ("context_precision", "Context precision", c3,
             "Retriever found the right chunks for each question"),
        ]
 
        for key, label, col, description in metrics_info:
            val = overall.get(key, float("nan"))
            g_text, g_class = grade(val)
            display = f"{val:.2f}" if val == val else "N/A"
            with col:
                st.markdown(f"""
                <div class="score-card">
                  <div class="score-label">{label}</div>
                  <div class="score-value">{display}</div>
                  <span class="score-grade {g_class}">{g_text}</span>
                  <div style="font-size:0.78rem; color:#999; margin-top:0.6rem;">
                    {description}
                  </div>
                </div>
                """, unsafe_allow_html=True)
 
        st.markdown("<br>", unsafe_allow_html=True)
 
        # Per-question breakdown
        st.markdown("### Per-question breakdown")
        questions = eval_data.get("questions", [])
        answers   = eval_data.get("answers",   [])
        per_q     = eval_data.get("per_question", [])
 
        if per_q:
            for i, (q, a, row) in enumerate(zip(questions, answers, per_q)):
                with st.expander(f"Q{i+1}: {q[:70]}..."):
                    st.markdown(f"**Answer:** {a[:400]}...")
                    st.markdown("**Scores:**")
                    m_cols = st.columns(3)
                    for j, (mkey, mlabel) in enumerate([
                        ("faithfulness",      "Faithfulness"),
                        ("answer_relevancy",  "Answer relevancy"),
                        ("context_precision", "Context precision"),
                    ]):
                        val = row.get(mkey, float("nan"))
                        g_text, g_class = grade(val)
                        display = f"{val:.3f}" if val == val else "N/A"
                        m_cols[j].markdown(
                            f"**{mlabel}:** {display} "
                            f'<span class="score-grade {g_class}" '
                            f'style="font-size:0.75rem; padding:0.15rem 0.4rem;">'
                            f'{g_text}</span>',
                            unsafe_allow_html=True,
                        )
 
        # Interpretation guide
        st.markdown("---")
        st.markdown("### How to interpret these scores")
        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown("""
            **Faithfulness < 0.7**
            → LLM may be going beyond the retrieved context.
            Fix: make the prompt more restrictive, reduce chunk size.
 
            **Answer relevancy < 0.7**
            → Answers are vague or off-topic.
            Fix: tune the prompt, try a larger LLM.
            """)
        with col_b:
            st.markdown("""
            **Context precision < 0.7**
            → Retriever is fetching irrelevant chunks.
            Fix: better embedding model, adjust TOP_K,
            or switch from similarity to MMR search.
 
            **All scores low?**
            → Start with context precision — if retrieval is poor,
            the LLM has nothing good to work with.
            """)
 
# ── FOOTER ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    '<div style="text-align:center; color:#aaa; font-size:0.8rem;">'
    'RAG Health QA · Built with LangChain · FAISS · HuggingFace · RAGAS'
    '</div>',
    unsafe_allow_html=True,
)
 
