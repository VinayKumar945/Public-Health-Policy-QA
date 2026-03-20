# rag_pipeline.py
# Phase 3 — RAG Pipeline
#
# Run to test:
#   python rag_pipeline.py

import os
from urllib import response
from dotenv import load_dotenv

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from huggingface_hub import InferenceClient   

load_dotenv()

# ── CONFIG ────────────────────────────────────────────────────────────────────
VECTORSTORE = "vectorstore"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL   = "HuggingFaceH4/zephyr-7b-beta"
TOP_K       = 4


# ── PROMPT TEMPLATE ───────────────────────────────────────────────────────────
PROMPT_TEMPLATE = """You are a public health policy expert assistant.
Use ONLY the context provided below to answer the question.
If the answer is not in the context, say "I don't have enough information in the provided documents to answer this."
Do not make up information. Be concise and precise.

Context:
{context}

Question: {question}

Answer:"""


# ── STEP 1: Load vector store ─────────────────────────────────────────────────
def load_vectorstore():
    """
    Loads the FAISS index using the same embedding model.
    """
    print("[+] Loading embedding model...")
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBED_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )

    print("[+] Loading FAISS vector store from disk...")
    vectorstore = FAISS.load_local(
        VECTORSTORE,
        embeddings,
        allow_dangerous_deserialization=True
    )
    print("[+] Vector store loaded.\n")
    return vectorstore


# ── STEP 2: Build LLM caller using InferenceClient ───────────────────────────
def build_llm_caller():
    """
    Returns a function: prompt → answer.
    """
    hf_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")
    if not hf_token:
        raise ValueError(
            "HUGGINGFACEHUB_API_TOKEN not found in .env\n"
            "Get yours at: huggingface.co → Settings → Access Tokens"
        )

    print(f"[+] Connecting to LLM: {LLM_MODEL}")

    client = InferenceClient(
        model=LLM_MODEL,
        token=hf_token,
    )

    def call_llm(prompt: str) -> str:
        # Convert to plain string for InferenceClient
        if hasattr(prompt, "text"):
            prompt = prompt.text
        else:
            prompt = str(prompt)

        response = client.chat_completion(
            messages=[{"role": "user", "content": prompt}],
            max_tokens=512,
            temperature=0.1,
        )
        return response.choices[0].message.content.strip()

    print("[+] LLM ready.\n")
    return call_llm


# ── STEP 3: Build the full RAG chain ──────────────────────────────────────────
def build_rag_chain(vectorstore, llm_caller):
    """
    retriever → prompt → LLM → output
    """
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": TOP_K}
    )

    prompt = PromptTemplate(
        template=PROMPT_TEMPLATE,
        input_variables=["context", "question"]
    )

    def format_docs(docs):
        return "\n\n---\n\n".join(doc.page_content for doc in docs)
    
    llm_runnable = RunnableLambda(llm_caller)

    chain = (
        {
            "context":  retriever | format_docs,
            "question": RunnablePassthrough()
        }
        | prompt
        | llm_runnable
        | StrOutputParser()
    )

    return chain, retriever


# ── STEP 4: Answer function — reused by app.py ────────────────────────────────
def answer_question(question: str, chain, retriever) -> dict:
    """
    Returns answer and sources.
    """
    answer = chain.invoke(question)

    source_docs = retriever.invoke(question)
    sources = []
    seen = set()
    for doc in source_docs:
        key = (doc.metadata.get("source", "unknown"),
               doc.metadata.get("page", "?"))
        if key not in seen:
            seen.add(key)
            sources.append({
                "source":  doc.metadata.get("source", "unknown"),
                "page":    doc.metadata.get("page", "?"),
                "snippet": doc.page_content[:200] + "..."
            })

    return {
        "answer":  answer.strip(),
        "sources": sources
    }


# ── MAIN ──────────────────────────────────────────────────────────────────────
def main():
    print("=" * 55)
    print("  RAG Health QA — Pipeline Test")
    print("=" * 55 + "\n")

    vectorstore      = load_vectorstore()
    llm_caller       = build_llm_caller()
    chain, retriever = build_rag_chain(vectorstore, llm_caller)

    test_questions = [
        "What are the main public health recommendations in this document?",
        "What risk factors are discussed?",
        "What interventions or policies are recommended?"
    ]

    for question in test_questions:
        print(f"Question: {question}")
        print("-" * 50)
        result = answer_question(question, chain, retriever)

        print(f"Answer:\n{result['answer']}\n")
        print("Sources:")
        for src in result["sources"]:
            print(f"  • {src['source']} — page {src['page']}")
            print(f"    \"{src['snippet']}\"")
        print("\n" + "=" * 55 + "\n")


if __name__ == "__main__":
    main()