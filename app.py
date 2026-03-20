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
import math
from pathlib import Path
import streamlit as st
from dotenv import load_dotenv

# Import our pipeline
from rag_pipeline import (
    load_vectorstore,
    build_llm_caller,
    build_rag_chain,
    answer_question,
)

load_dotenv()

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

  /* Force light theme base — overrides Streamlit dark mode */
  html, body, [class*="css"], .stApp {
    font-family: 'DM Sans', sans-serif !important;
    color: #222222 !important;
  }

  /* Main background */
  .stApp {
    background-color: #f7f5f0 !important;
  }

  /* Force all generic text dark */
  p, span, div, label, li, td, th, h1, h2, h3, h4, h5, h6 {
    color: #222222;
  }

  /* Streamlit markdown containers */
  .stMarkdown, .stMarkdown p, .stMarkdown span, .stMarkdown div {
    color: #222222 !important;
  }

  /* Streamlit text elements */
  .element-container, .stText {
    color: #222222 !important;
  }

  /* Header */
  .app-header {
    background: linear-gradient(135deg, #1a3a2a 0%, #2d5a3d 100%);
    color: white !important;
    padding: 2rem 2.5rem;
    border-radius: 16px;
    margin-bottom: 1.5rem;
  }
  .app-header h1 {
    font-family: 'DM Serif Display', serif;
    font-size: 2.2rem;
    margin: 0 0 0.4rem 0;
    letter-spacing: -0.5px;
    color: white !important;
  }
  .app-header p {
    font-size: 0.95rem;
    opacity: 0.8;
    margin: 0;
    font-weight: 300;
    color: white !important;
  }

  /* Tab styling */
  .stTabs [data-baseweb="tab-list"] {
    background: transparent;
    gap: 8px;
  }
  .stTabs [data-baseweb="tab"] {
    background: white !important;
    border-radius: 8px !important;
    border: 1px solid #e0ddd6 !important;
    padding: 0.5rem 1.2rem !important;
    font-weight: 500 !important;
    color: #333333 !important;
  }
  .stTabs [aria-selected="true"] {
    background: #1a3a2a !important;
    color: white !important;
    border-color: #1a3a2a !important;
  }

  /* Question input */
  .stTextInput > div > div > input {
    border-radius: 10px !important;
    border: 2px solid #e0ddd6 !important;
    padding: 0.75rem 1rem !important;
    font-size: 1rem !important;
    background: white !important;
    color: #222222 !important;
    transition: border-color 0.2s;
  }
  .stTextInput > div > div > input:focus {
    border-color: #2d5a3d !important;
    box-shadow: 0 0 0 3px rgba(45,90,61,0.1) !important;
  }
  .stTextInput > div > div > input::placeholder {
    color: #999999 !important;
  }

  /* Answer box */
  .answer-box {
    background: white;
    border-left: 4px solid #2d5a3d;
    border-radius: 0 12px 12px 0;
    padding: 1.25rem 1.5rem;
    margin: 1rem 0;
    box-shadow: 0 2px 8px rgba(0,0,0,0.06);
    line-height: 1.75;
    color: #222222 !important;
  }

  /* Source card */
  .source-card {
    background: #f0f4f1;
    border: 1px solid #c8d9cb;
    border-radius: 10px;
    padding: 0.85rem 1.1rem;
    margin: 0.4rem 0;
    font-size: 0.88rem;
  }
  .source-label {
    font-weight: 500;
    color: #1a3a2a !important;
    margin-bottom: 0.3rem;
  }
  .source-snippet {
    color: #444444 !important;
    font-style: italic;
    line-height: 1.5;
  }

  /* Score card */
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
    color: #1a3a2a !important;
    line-height: 1;
    margin: 0.5rem 0;
  }
  .score-label {
    font-size: 0.85rem;
    font-weight: 500;
    text-transform: uppercase;
    letter-spacing: 1px;
    color: #555555 !important;
  }
  .score-grade {
    font-size: 0.8rem;
    margin-top: 0.4rem;
    padding: 0.2rem 0.6rem;
    border-radius: 20px;
    display: inline-block;
  }
  .grade-good { background: #e6f4ea; color: #1e7e34 !important; }
  .grade-ok   { background: #fff3cd; color: #856404 !important; }
  .grade-bad  { background: #fde8e8; color: #c0392b !important; }

  /* History item */
  .history-item {
    background: white;
    border-radius: 10px;
    padding: 1rem 1.2rem;
    margin: 0.5rem 0;
    border: 1px solid #e8e4dd;
    cursor: pointer;
  }
  .history-q {
    font-weight: 500;
    color: #1a3a2a !important;
    font-size: 0.92rem;
  }
  .history-a {
    color: #444444 !important;
    font-size: 0.85rem;
    margin-top: 0.3rem;
    line-height: 1.4;
  }

  /* Spinner override */
  .stSpinner > div {
    border-top-color: #2d5a3d !important;
  }

  /* All buttons — light outlined by default (example/secondary buttons) */
  .stButton > button {
    background: white !important;
    color: #1a3a2a !important;
    border: 2px solid #1a3a2a !important;
    border-radius: 8px !important;
    padding: 0.5rem 1.5rem !important;
    font-weight: 500 !important;
    transition: background 0.2s, color 0.2s;
  }
  .stButton > button:hover {
    background: #f0f4f1 !important;
    color: #1a3a2a !important;
  }

  /* Primary buttons (Ask →, Clear history) */
  .stButton > button[kind="primary"] {
    background: #1a3a2a !important;
    color: white !important;
    border: none !important;
  }
  .stButton > button[kind="primary"]:hover {
    background: #2d5a3d !important;
    color: white !important;
  }

  /* Warning / info boxes */
  .stWarning, .stInfo {
    color: #222222 !important;
  }

  /* Expander */
  .streamlit-expanderHeader {
    color: #222222 !important;
    background: white !important;
  }
  .streamlit-expanderContent {
    color: #222222 !important;
    background: white !important;
  }

  /* Sidebar info */
  .info-box {
    background: #eef4f0;
    border-radius: 10px;
    padding: 1rem 1.2rem;
    font-size: 0.88rem;
    color: #333333 !important;
    line-height: 1.6;
    margin-bottom: 1rem;
  }

  /* Section headings rendered by st.markdown */
  .stMarkdown h3 {
    color: #1a3a2a !important;
  }

  /* Footer */
  .footer-text {
    color: #888888 !important;
    text-align: center;
    font-size: 0.8rem;
  }
</style>
""", unsafe_allow_html=True)


# ── LOAD PIPELINE (cached so it only loads once) ──────────────────────────────
@st.cache_resource(show_spinner=False)
def load_pipeline():
    with st.spinner("Loading RAG pipeline... (first load takes ~30s)"):
        vectorstore      = load_vectorstore()
        llm_caller       = build_llm_caller()
        chain, retriever = build_rag_chain(vectorstore, llm_caller)
    return chain, retriever


# ── LOAD EVALUATION RESULTS ───────────────────────────────────────────────────
def load_eval_results():
    results_file = "evaluation_results.json"
    if not Path(results_file).exists():
        return None
    with open(results_file) as f:
        return json.load(f)


# ── HELPER: grade a score ─────────────────────────────────────────────────────
def grade(score):
    if score != score or score is None:  # nan check
        return "~ No data", "grade-bad"
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
            "Type any question about the public health documents in your knowledge base. "
            "Answers are grounded in the source documents — no hallucination."
        )

        question = st.text_input(
            label="Your question",
            placeholder="e.g. What are the main public health recommendations?",
            label_visibility="collapsed",
        )

        ask_btn = st.button("Ask →", use_container_width=False, type="primary")

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

        if ask_btn and question.strip():
            chain, retriever = load_pipeline()

            with st.spinner("Searching documents and generating answer..."):
                result = answer_question(question.strip(), chain, retriever)

            st.markdown("#### Answer")
            st.markdown(
                f'<div class="answer-box">{result["answer"]}</div>',
                unsafe_allow_html=True,
            )

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

            if "history" not in st.session_state:
                st.session_state.history = []
            st.session_state.history.insert(0, {
                "question": question.strip(),
                "answer":   result["answer"][:180] + "...",
            })

        elif ask_btn and not question.strip():
            st.warning("Please enter a question first.")

        if not (ask_btn and question.strip()):
            st.markdown("""
            <div style="text-align:center; padding: 3rem 1rem; color: #888888;">
              <div style="font-size: 3rem; margin-bottom: 1rem;">🔍</div>
              <div style="font-size: 1rem; color: #888888;">
                Ask any question about your public health documents
              </div>
            </div>
            """, unsafe_allow_html=True)

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

            if st.button("Clear history", type="primary"):
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
            f'<div style="color:#666666; font-size:0.85rem; margin-bottom:1rem;">'
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
                  <div style="font-size:0.78rem; color:#666666; margin-top:0.6rem;">
                    {description}
                  </div>
                </div>
                """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

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

        st.markdown("---")
        st.markdown("### How to interpret these scores")

        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown("""
            **Faithfulness < 0.7**
            → The LLM may be going beyond the retrieved context.
            Fix: make the prompt more restrictive, reduce chunk size.

            **Answer relevancy < 0.7**
            → Answers are vague or off-topic.
            Fix: tune the prompt, try a larger/better LLM.
            """)
        with col_b:
            st.markdown("""
            **Context precision < 0.7**
            → Retriever is fetching irrelevant chunks.
            Fix: try a better embedding model, adjust TOP_K,
            or switch from similarity to MMR search.

            **All scores low?**
            → Start with context precision — if retrieval is poor,
            the LLM has nothing good to work with.
            """)


# ── FOOTER ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    '<div class="footer-text">'
    'RAG Health QA · Built with LangChain · FAISS · HuggingFace · RAGAS'
    '</div>',
    unsafe_allow_html=True,
)