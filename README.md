---
title: RAG Health QA
emoji: 🏥
colorFrom: green
colorTo: green
sdk: streamlit
sdk_version: 1.38.0
app_file: app.py
pinned: false
---

<div align="center">

# 🏥 Public Health Policy QA System

### Retrieval-Augmented Generation on WHO Health Policy Documents

[![Live Demo](https://img.shields.io/badge/🚀_Live_Demo-HuggingFace_Spaces-orange?style=for-the-badge)](https://vinayverma3218-rag-health-qa.hf.space)
[![GitHub](https://img.shields.io/badge/GitHub-Repository-black?style=for-the-badge&logo=github)](https://github.com/VinayKumar945/Public-Health-Policy-QA)
[![Python](https://img.shields.io/badge/Python-3.11+-blue?style=for-the-badge&logo=python)](https://python.org)
[![LangChain](https://img.shields.io/badge/LangChain-0.2.17-green?style=for-the-badge)](https://langchain.com)

*Ask questions about WHO public health documents. Get grounded answers with page-level citations and production evaluation metrics.*

</div>

---

## 📌 What This Project Does

Most RAG demos stop at "it works." This one measures *how well* it works.

This system answers questions about WHO public health policy documents using a full Retrieval-Augmented Generation (RAG) pipeline — with RAGAS evaluation metrics to quantify quality at every stage of the pipeline, not just demo it.

**Two-tab Streamlit UI:**
- 💬 **Ask a Question** — type any health policy question, get a grounded answer with source citations (file name + page number)
- 📊 **Evaluation Dashboard** — live RAGAS scores showing faithfulness, answer relevancy, and context precision

---

## 🏗️ Architecture

```
PDF Documents (450+ pages)
        │
        ▼
┌─────────────────────┐
│   PyMuPDF (fitz)    │  ← Extract text page by page
└─────────────────────┘
        │
        ▼
┌─────────────────────┐
│  RecursiveCharacter │  ← Chunk: 500 chars, 50 overlap
│   TextSplitter      │
└─────────────────────┘
        │
        ▼
┌─────────────────────┐
│  all-MiniLM-L6-v2   │  ← Embed: 384-dimensional vectors
│  (sentence-         │
│   transformers)     │
└─────────────────────┘
        │
        ▼
┌─────────────────────┐
│   FAISS Index       │  ← Store: 3,018 chunks indexed
│   (saved to disk)   │
└─────────────────────┘

At query time:
User Question → Embed → FAISS similarity search (top-4)
             → LangChain LCEL prompt → Zephyr-7B LLM
             → Grounded answer + page citations
```

---

## 📊 Evaluation Results (RAGAS)

| Metric | Score | Meaning |
|--------|-------|---------|
| **Answer Relevancy** | 0.68 | Answer addresses the question asked |
| **Context Precision** | 0.23 | Retriever finds relevant chunks |
| **Faithfulness** | N/A | Credits exhausted mid-run |

> **Why these metrics matter:** Most RAG projects show demos. RAGAS lets you *measure* quality — faithfulness catches hallucination, answer relevancy catches vague responses, context precision diagnoses retriever issues. A score of 0.23 on context precision tells me exactly what to fix next: MMR search and smaller chunk sizes.

---

## 🛠️ Tech Stack

| Layer | Tool | Why |
|-------|------|-----|
| PDF parsing | PyMuPDF | Fast, accurate text extraction |
| Chunking | LangChain RecursiveCharacterTextSplitter | Sentence-aware splitting |
| Embeddings | `all-MiniLM-L6-v2` | Free, 384-dim, strong semantic search |
| Vector store | FAISS | In-process, no server needed, saves to disk |
| LLM | Zephyr-7B via HuggingFace Inference API | Free, no OpenAI key required |
| Orchestration | LangChain LCEL | Composable, model-agnostic pipeline |
| Evaluation | RAGAS v0.1.21 | Reference-free LLM evaluation |
| UI | Streamlit | Rapid ML app deployment |
| Hosting | HuggingFace Spaces | Free, live public URL |

---

## 🚀 Quick Start

### Prerequisites
- Python 3.11+
- HuggingFace account + API token ([get one here](https://huggingface.co/settings/tokens))

### 1. Clone and install
```bash
git clone https://github.com/VinayKumar945/Public-Health-Policy-QA
cd Public-Health-Policy-QA
pip install -r requirements.txt
```

### 2. Set your HuggingFace token
```bash
echo "HUGGINGFACEHUB_API_TOKEN=hf_your_token_here" > .env
```

### 3. Add your PDFs
```bash
mkdir -p data/pdfs
# Copy your PDF documents into data/pdfs/
```

### 4. Build the vector store (run once)
```bash
python ingest.py
```

### 5. Run the app
```bash
streamlit run app.py
```

---

## 📁 Project Structure

```
├── app.py                    # Streamlit UI (chat + evaluation dashboard)
├── ingest.py                 # PDF → chunks → embeddings → FAISS index
├── rag_pipeline.py           # LangChain LCEL chain + answer function
├── evaluate.py               # RAGAS evaluation pipeline
├── evaluation_results.json   # Saved RAGAS scores (loaded by dashboard)
├── data/
│   └── pdfs/                 # Source PDF documents
├── vectorstore/
│   ├── index.faiss           # FAISS vector index (3,018 chunks)
│   └── index.pkl             # Metadata (source, page number)
└── requirements.txt          # Pinned dependencies
```

---

## 🔍 Key Design Decisions

**Why FAISS over ChromaDB?**
FAISS runs in-process with no server overhead. For a single-user demo, it's faster to load and simpler to deploy. For production multi-user, I'd migrate to Pinecone or Weaviate.

**Why same embedding model at index and query time?**
Documents and queries must live in the same vector space. Using different models breaks similarity search — like translating documents into French but searching in Spanish.

**Why `ragas==0.1.21` pinned?**
RAGAS v0.2+ was rewritten to require OpenAI as the judge LLM. v0.1.21 supports custom LLM wrappers, enabling free evaluation with Zephyr-7B. In production with a budget, I'd upgrade and use GPT-4-mini as judge.

**Why `temperature=0.1`?**
Low temperature = deterministic, factual responses. For public health QA where accuracy matters, creativity is a liability.

**Why lazy loading on HuggingFace Spaces?**
HuggingFace has a 30-second health check — if the app doesn't respond, it gets killed. Loading models only on first user question lets the app boot instantly, pass the health check, then load when needed.

---

## 📈 What I'd Improve Next

| Improvement | Expected Impact |
|-------------|----------------|
| Switch similarity search → MMR | Higher context precision (currently 0.23) |
| Reduce chunk size 500 → 300 chars | More precise retrieval |
| Add re-ranking with cross-encoder | Better top-4 chunk selection |
| Upgrade to RAGAS v0.2 + GPT-4-mini judge | More reliable faithfulness scores |
| Add query caching | Faster responses for repeated questions |

---

## 💡 Technical Challenges Solved

**HuggingFace deprecated their inference endpoint mid-project** (got a 410 Gone error). Traced it through LangChain source code, found the hardcoded URL, switched to `InferenceClient` which auto-routes to the new endpoint.

**Dependency conflict:** `ragas==0.1.21` requires `langchain-core<0.3` but `langchain-huggingface` requires `>=0.3`. Solved by using `langchain_community.embeddings` instead — these two versions cannot coexist.

**HuggingFace Spaces rejected binary files** (FAISS index). Solved by committing with Git LFS tracking for `.faiss` and `.pkl` files.

---

## 📚 References

- [RAGAS: Automated Evaluation of RAG Pipelines](https://arxiv.org/abs/2309.15217)
- [LangChain Expression Language (LCEL)](https://python.langchain.com/docs/expression_language/)
- [FAISS: A Library for Efficient Similarity Search](https://engineering.fb.com/2017/03/29/data-infrastructure/faiss-a-library-for-efficient-similarity-search/)
- [Sentence Transformers: all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)
- [HuggingFaceH4/zephyr-7b-beta](https://huggingface.co/HuggingFaceH4/zephyr-7b-beta)

---

## 👤 Author

**Vinay Kumar** — Data Scientist

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue?style=flat&logo=linkedin)](https://linkedin.com/in/vinaykumar945)
[![GitHub](https://img.shields.io/badge/GitHub-Follow-black?style=flat&logo=github)](https://github.com/VinayKumar945)
[![Live Demo](https://img.shields.io/badge/Live_Demo-Try_it-orange?style=flat)](https://vinayverma3218-rag-health-qa.hf.space)

---

<div align="center">
<sub>Built with LangChain · FAISS · HuggingFace · RAGAS · Streamlit</sub>
</div>
