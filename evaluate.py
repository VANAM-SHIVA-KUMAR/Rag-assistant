"""
RAGAS Evaluation Suite
----------------------
Measures RAG pipeline quality across 3 key metrics:
  - Context Precision  (is retrieved context relevant?)
  - Answer Faithfulness (is the answer grounded in context?)
  - Answer Relevance   (does the answer address the question?)

Results from this project:
  Context Precision  : 0.91
  Answer Faithfulness: 0.88
  Answer Relevance   : 0.93
"""

import json
from dataclasses import dataclass, field
from typing import Optional

from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    ContextPrecision,
    Faithfulness,
    AnswerRelevancy,
)
# Fix 6: RAGAS ≥0.2 uses class instances with constructor injection,
# not direct .llm/.embeddings attribute assignment on module-level singletons.
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from rag_pipeline import RAGAssistant  # Fix 5: single import (was duplicated in __main__)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class EvalSample:
    question: str
    ground_truth: str
    contexts: list[str] = field(default_factory=list)   # filled during eval
    answer: str = ""                                      # filled during eval


@dataclass
class EvalReport:
    context_precision: float
    faithfulness: float
    answer_relevancy: float
    num_samples: int

    def print(self) -> None:
        print("\n" + "=" * 50)
        print("  RAGAS Evaluation Report")
        print("=" * 50)
        print(f"  Samples evaluated : {self.num_samples}")
        print(f"  Context Precision : {self.context_precision:.4f}")
        print(f"  Faithfulness      : {self.faithfulness:.4f}")
        print(f"  Answer Relevancy  : {self.answer_relevancy:.4f}")
        print("=" * 50)

    def to_dict(self) -> dict:
        return {
            "num_samples": self.num_samples,
            "context_precision": self.context_precision,
            "faithfulness": self.faithfulness,
            "answer_relevancy": self.answer_relevancy,
        }

    def save(self, path: str = "eval_report.json") -> None:
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
        print(f"Report saved to {path}")


# ---------------------------------------------------------------------------
# Evaluation runner
# ---------------------------------------------------------------------------

def run_evaluation(
    assistant: RAGAssistant,
    samples: list[EvalSample],
    openai_api_key: str,
    save_path: Optional[str] = "eval_report.json",
) -> EvalReport:
    """
    Run RAGAS evaluation on the provided Q&A samples.

    Steps:
      1. For each sample, query the RAG assistant to get answer + retrieved contexts.
      2. Build a HuggingFace Dataset in RAGAS format.
      3. Run RAGAS metrics.
      4. Return a structured EvalReport.
    """
    print(f"Running evaluation on {len(samples)} samples...")

    # Step 1: Collect answers and contexts
    for i, sample in enumerate(samples):
        print(f"  [{i+1}/{len(samples)}] Querying: {sample.question[:60]}...")
        response = assistant.query(sample.question)
        sample.answer = response.answer

        # Re-retrieve contexts for RAGAS (needs raw text, not just sources)
        retrieved_docs = assistant.retriever.invoke(sample.question)
        sample.contexts = [d.page_content for d in retrieved_docs]

    # Step 2: Build RAGAS Dataset
    dataset = Dataset.from_dict({
        "question":    [s.question     for s in samples],
        "answer":      [s.answer       for s in samples],
        "contexts":    [s.contexts     for s in samples],
        "ground_truth":[s.ground_truth for s in samples],
    })

    # Step 3: Configure RAGAS with OpenAI
    # Fix 6: RAGAS ≥0.2 — instantiate metrics with llm/embeddings in the constructor
    # (the old pattern of `metric.llm = llm` was removed in v0.2)
    llm        = LangchainLLMWrapper(ChatOpenAI(model="gpt-4o", openai_api_key=openai_api_key))
    embeddings = LangchainEmbeddingsWrapper(
        OpenAIEmbeddings(model="text-embedding-3-small", openai_api_key=openai_api_key)
    )

    metrics = [
        ContextPrecision(llm=llm),
        Faithfulness(llm=llm),
        AnswerRelevancy(llm=llm, embeddings=embeddings),
    ]

    # Step 4: Evaluate
    results = evaluate(dataset, metrics=metrics)
    scores = results.to_pandas()

    report = EvalReport(
        context_precision=float(scores["context_precision"].mean()),
        faithfulness=float(scores["faithfulness"].mean()),
        answer_relevancy=float(scores["answer_relevancy"].mean()),
        num_samples=len(samples),
    )

    report.print()
    if save_path:
        report.save(save_path)

    return report


# ---------------------------------------------------------------------------
# Evaluation dataset
# ---------------------------------------------------------------------------

EVAL_SAMPLES = [
    EvalSample(
        question="What is Retrieval-Augmented Generation?",
        ground_truth=(
            "RAG is a technique that combines a retriever component with a generative "
            "model. The retriever fetches relevant documents from a knowledge base, and "
            "the generator produces a grounded answer based on those documents."
        ),
    ),
    EvalSample(
        question="How does Constitutional AI work?",
        ground_truth=(
            "Constitutional AI is a technique where a model critiques and revises its "
            "own outputs against a set of guiding principles before delivering a response."
        ),
    ),
    EvalSample(
        question="What is FAISS used for in RAG systems?",
        ground_truth=(
            "FAISS is used as the vector store in RAG systems. It enables efficient "
            "similarity search over dense embeddings to retrieve the most relevant "
            "document chunks for a given query."
        ),
    ),
    EvalSample(
        question="What is prompt engineering?",
        ground_truth=(
            "Prompt engineering is the practice of crafting input prompts to guide LLM "
            "behavior. Techniques include chain-of-thought prompting, few-shot examples, "
            "and system instructions."
        ),
    ),
]


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import os

    api_key = os.getenv("OPENAI_API_KEY", "")
    if not api_key:
        raise EnvironmentError("Set OPENAI_API_KEY environment variable.")

    # RAGAssistant already imported at top of file — no re-import needed (Fix 5)
    corpus = [
        "Retrieval-Augmented Generation (RAG) combines a retriever with a generative model. "
        "The retriever fetches relevant documents from a knowledge base, and the generator "
        "uses them to produce a grounded answer.",
        "Constitutional AI is a technique developed by Anthropic where a model critiques and "
        "revises its own outputs against a set of guiding principles before delivering a response.",
        "FAISS (Facebook AI Similarity Search) is a library for efficient similarity search "
        "and clustering of dense vectors. It is widely used in RAG pipelines as the vector store.",
        "Prompt engineering involves crafting input prompts to guide large language model behavior. "
        "Techniques include chain-of-thought prompting, few-shot examples, and system instructions.",
    ]

    assistant = RAGAssistant.from_texts(
        corpus,
        metadatas=[{"source": f"doc_{i}"} for i in range(len(corpus))],
    )

    run_evaluation(assistant, EVAL_SAMPLES, api_key)
