# ingest.py
# Phase 2 — Data ingestion for the RAG pipeline
#
# What this script does:
# 1. Reads every PDF from data/pdfs/
# 2. Pulls text out page by page
# 3. Splits the text into smaller overlapping chunks
# 4. Creates embeddings for each chunk
# 5. Saves everything into a FAISS vector store on disk
#
# Run this once before starting the app:
#     python ingest.py

import os
from pathlib import Path
from dotenv import load_dotenv

import fitz  # PyMuPDF — reads PDFs
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

# Load values from .env if present
load_dotenv()

# Config
PDF_FOLDER   = "data/pdfs"          # folder where your PDFs live
VECTORSTORE  = "vectorstore"        # where FAISS will be saved
EMBED_MODEL  = "sentence-transformers/all-MiniLM-L6-v2"  # free, no API key needed
CHUNK_SIZE   = 500    # words per chunk (roughly)
CHUNK_OVERLAP = 50    # words shared between neighbouring chunks

def load_pdfs(folder: str) -> list[dict]:
    """
    Read all PDFs from the given folder and extract text page by page.
    """
    documents = []
    pdf_files = list(Path(folder).glob("*.pdf"))

    if not pdf_files:
        print(f"[!] No PDFs found in '{folder}'. Add some PDFs and re-run.")
        return []

    print(f"[+] Found {len(pdf_files)} PDF(s):")
    for pdf_path in pdf_files:
        print(f"    → {pdf_path.name}")
        doc = fitz.open(pdf_path)          
        for page_num, page in enumerate(doc):
            text = page.get_text()         
            if text.strip():               
                documents.append({
                    "text":   text,
                    "source": pdf_path.name,
                    "page":   page_num + 1  # human-friendly page number
                })
        doc.close()

    total_pages = len(documents)
    print(f"[+] Extracted text from {total_pages} pages total.\n")
    return documents


def split_into_chunks(documents: list[dict]) -> list:
    """
    Split page text into smaller overlapping chunks.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""]  
    )

    all_chunks = []
    for doc in documents:
        # split the text
        pieces = splitter.split_text(doc["text"])
        for piece in pieces:
            all_chunks.append({
                "text":   piece,
                "source": doc["source"],
                "page":   doc["page"]
            })

    print(f"[+] Split {len(documents)} pages into {len(all_chunks)} chunks.\n")
    return all_chunks



def build_vectorstore(chunks: list[dict]) -> None:
    """
    Turn each chunk into an embedding and store everything in FAISS.
    """
    print(f"[+] Loading embedding model: {EMBED_MODEL}")
    print("    (First run downloads ~80MB — this is normal)\n")

    embeddings = HuggingFaceEmbeddings(
        model_name=EMBED_MODEL,
        model_kwargs={"device": "cpu"},   # change to "cuda" if you have a GPU
        encode_kwargs={"normalize_embeddings": True}
    )

    texts = [chunk["text"] for chunk in chunks]
    metadatas = [
        {"source": chunk["source"], "page": chunk["page"]}
        for chunk in chunks
    ]

    print(f"[+] Embedding {len(texts)} chunks — this may take 1-2 minutes...")
    
    vectorstore = FAISS.from_texts(
        texts=texts,
        embedding=embeddings,
        metadatas=metadatas
    )

    os.makedirs(VECTORSTORE, exist_ok=True)
    vectorstore.save_local(VECTORSTORE)
    print(f"[+] Vector store saved to '{VECTORSTORE}/'")
    print(f"    Files created: {VECTORSTORE}/index.faiss, {VECTORSTORE}/index.pkl\n")



def main():
    print("=" * 55)
    print("  RAG Health QA — Ingestion Pipeline")
    print("=" * 55 + "\n")
    # Step 1: Load PDFs and extract text
    documents = load_pdfs(PDF_FOLDER)
    if not documents:
        return

    # Step 2: Chunk them
    chunks = split_into_chunks(documents)

    # Step 3: Embed + save
    build_vectorstore(chunks)

    print("=" * 55)
    print("  Done! You can now run the RAG pipeline.")
    print("  Next: python rag_pipeline.py  (or  streamlit run app.py)")
    print("=" * 55)


if __name__ == "__main__":
    main()