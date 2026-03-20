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

## 🏥 Public Health Policy QA

A Retrieval-Augmented Generation (RAG) app for querying public health policy documents.

## What it does

- **Ask questions** about uploaded public health PDFs and get grounded, cited answers
- **Evaluation Dashboard** showing RAGAS scores (faithfulness, answer relevancy, context precision)

## Stack

- **LangChain** — RAG orchestration
- **FAISS** — vector similarity search
- **HuggingFace** — embeddings (`all-MiniLM-L6-v2`) + LLM (`zephyr-7b-beta`)
- **RAGAS** — pipeline evaluation metrics
- **Streamlit** — UI

## Setup (running locally)
```bash
pip install -r requirements.txt

# Add your HuggingFace token to .env
echo "HUGGINGFACEHUB_API_TOKEN=hf_your_token_here" > .env

# Put PDFs in data/pdfs/ then ingest
python ingest.py

# Run the app
streamlit run app.py
```

## Running on this Space

The Space requires a `HUGGINGFACEHUB_API_TOKEN` secret set in **Settings → Variables and secrets**.

If no vectorstore is present, the app will show instructions to run `ingest.py` first.
