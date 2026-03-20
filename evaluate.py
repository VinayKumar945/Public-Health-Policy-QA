# evaluate.py
# Phase 4 — RAGAS Evaluation
#
# What this file does:
#   Runs a small test set of questions through the RAG pipeline,
#   then scores the results using 3 RAGAS metrics:
#
#   1. Faithfulness      — did the answer come ONLY from the retrieved context?
#   2. Answer Relevancy  — does the answer actually address the question?
#   3. Context Precision — did the retriever find the RIGHT chunks?
#
# Requires: ragas==0.1.21
#   pip install ragas==0.1.21
#
# Run:
#   python evaluate.py

import json
import os
from datetime import datetime
from typing import Iterator, List, Optional
from dotenv import load_dotenv

from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.language_models.llms import LLM
from langchain_core.outputs import GenerationChunk
from huggingface_hub import InferenceClient

from rag_pipeline import (
    load_vectorstore,
    build_llm_caller,
    build_rag_chain,
    answer_question,
)

load_dotenv()

# ── CONFIG ────────────────────────────────────────────────────────────────────
EMBED_MODEL  = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL    = "HuggingFaceH4/zephyr-7b-beta"
RESULTS_FILE = "evaluation_results.json"


# ── TEST QUESTIONS ────────────────────────────────────────────────────────────
# Change these to match your PDF content.
# No ground truth answers needed — RAGAS evaluates using the context itself.


TEST_QUESTIONS = [
    "What are the main public health recommendations in this document?",
    "What risk factors are discussed in relation to health inequalities?",
    "What interventions are recommended for improving population health?",
    "How does the document define health equity?",
    "What data collection methods are described for public health monitoring?",
]


# ── LANGCHAIN LLM WRAPPER FOR RAGAS ──────────────────────────────────────────
# RAGAS v0.1.21 needs a LangChain-compatible LLM to run its internal checks.
# We wrap InferenceClient in a minimal LangChain LLM class.

class ZephyrLLM(LLM):
    """Minimal LangChain LLM wrapper around HuggingFace InferenceClient."""
    model_name: str = LLM_MODEL

    @property
    def _llm_type(self) -> str:
        return "zephyr-hf"

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager=None,
        **kwargs,
    ) -> str:
        hf_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")
        client   = InferenceClient(model=self.model_name, token=hf_token)
        response = client.chat_completion(
            messages=[{"role": "user", "content": prompt}],
            max_tokens=512,
            temperature=0.1,
        )
        return response.choices[0].message.content.strip()

    def _stream(self, prompt: str, **kwargs) -> Iterator[GenerationChunk]:
        yield GenerationChunk(text=self._call(prompt))


# ── STEP 1: Collect RAG outputs ───────────────────────────────────────────────
def collect_rag_outputs(chain, retriever) -> dict:
    """
    Runs each test question through the RAG pipeline and collects:
        question       — the input question
        answer         — the generated answer
        contexts       — list of retrieved chunk texts
        ground_truths  — empty placeholders (not needed for our 3 metrics)

    RAGAS v0.1.21 expects all four keys in the dataset.
    """
    print("[+] Running test questions through RAG pipeline...\n")

    questions     = []
    answers       = []
    contexts_list = []

    for i, question in enumerate(TEST_QUESTIONS, 1):
        print(f"  [{i}/{len(TEST_QUESTIONS)}] {question[:60]}...")

        result      = answer_question(question, chain, retriever)
        source_docs = retriever.invoke(question)

        questions.append(question)
        answers.append(result["answer"])
        contexts_list.append([doc.page_content for doc in source_docs])

    print(f"\n[+] Collected outputs for {len(questions)} questions.\n")

    return {
        "question":      questions,
        "answer":        answers,
        "contexts":      contexts_list,
        "ground_truth": ["" for _ in questions],  
    }


# ── STEP 2: Run RAGAS evaluation ──────────────────────────────────────────────
def run_ragas_evaluation(data: dict):
    """
    Scores the collected QA data using 3 RAGAS metrics.
    """
    print("[+] Building RAGAS evaluation dataset...")
    dataset = Dataset.from_dict(data)

    print("[+] Setting up RAGAS judge LLM and embeddings...")

    llm_wrapper = LangchainLLMWrapper(ZephyrLLM())

    embeddings_wrapper = LangchainEmbeddingsWrapper(
        HuggingFaceEmbeddings(
            model_name=EMBED_MODEL,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
    )

    metrics = [faithfulness, answer_relevancy, context_precision]
    for metric in metrics:
        metric.llm        = llm_wrapper
        metric.embeddings = embeddings_wrapper

    print("[+] Running RAGAS evaluation — this takes 3-5 minutes...\n")

    results = evaluate(
        dataset=dataset,
        metrics=metrics,
    )

    return results


# ── STEP 3: Display and save results ──────────────────────────────────────────
def display_and_save_results(results, data: dict) -> None:
    """
    Prints a clean summary table and saves full results to JSON.
    """
    print("\n" + "=" * 55)
    print("  RAGAS EVALUATION RESULTS")
    print("=" * 55)

    scores = {
        "faithfulness":     round(float(results["faithfulness"]),     3),
        "answer_relevancy": round(float(results["answer_relevancy"]), 3),
        "context_precision":round(float(results["context_precision"]),3),
    }

    def grade(score):
        if score != score:  # nan check
            return "~ Could not evaluate"
        if score >= 0.8: return "✓ Good"
        if score >= 0.6: return "~ Acceptable"
        return "✗ Needs improvement"

    print(f"\n  Faithfulness:      {scores['faithfulness']:.3f}  {grade(scores['faithfulness'])}")
    print(f"  Answer relevancy:  {scores['answer_relevancy']:.3f}  {grade(scores['answer_relevancy'])}")
    print(f"  Context precision: {scores['context_precision']:.3f}  {grade(scores['context_precision'])}")

    overall = sum(scores.values()) / len(scores)
    print(f"\n  Overall average:   {overall:.3f}")

    print("\n  Per-question breakdown:")
    print("  " + "-" * 50)
    df = results.to_pandas()
    for i, row in df.iterrows():
        print(f"\n  Q{i+1}: {data['question'][i][:55]}...")
        f_val  = row.get("faithfulness")
        ar_val = row.get("answer_relevancy")
        cp_val = row.get("context_precision")
        print(f"       Faithfulness:      {f_val:.3f}"  if isinstance(f_val,  float) else "       Faithfulness:      N/A")
        print(f"       Answer relevancy:  {ar_val:.3f}" if isinstance(ar_val, float) else "       Answer relevancy:  N/A")
        print(f"       Context precision: {cp_val:.3f}" if isinstance(cp_val, float) else "       Context precision: N/A")

    output = {
        "timestamp":    datetime.now().isoformat(),
        "overall":      scores,
        "per_question": df.to_dict(orient="records"),
        "questions":    data["question"],
        "answers":      data["answer"],
    }
    with open(RESULTS_FILE, "w") as f:
        json.dump(output, f, indent=2, default=str)

    print(f"\n[+] Full results saved to '{RESULTS_FILE}'")
    print("    app.py will load this file to show scores in the UI.\n")


# ── MAIN ──────────────────────────────────────────────────────────────────────
def main():
    print("=" * 55)
    print("  RAG Health QA — RAGAS Evaluation")
    print("=" * 55 + "\n")

    vectorstore      = load_vectorstore()
    llm_caller       = build_llm_caller()
    chain, retriever = build_rag_chain(vectorstore, llm_caller)

    # Step 1: Run questions, collect outputs
    data = collect_rag_outputs(chain, retriever)

    # Step 2: Score with RAGAS
    results = run_ragas_evaluation(data)

    # Step 3: Display and save
    display_and_save_results(results, data)

    print("=" * 55)
    print("  Evaluation complete!")
    print("  Next step: streamlit run app.py")
    print("=" * 55)


if __name__ == "__main__":
    main()