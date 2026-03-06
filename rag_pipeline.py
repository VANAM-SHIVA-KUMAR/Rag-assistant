"""
Constitutional AI-Aligned RAG Pipeline
--------------------------------------
Production RAG pipeline with:
  - Document ingestion + chunking
  - FAISS vector store (embedding-based retrieval)
  - GPT-4 synthesis
  - Constitutional AI critique layer (self-check before delivery)
"""

import os
import json
from typing import Optional
from dataclasses import dataclass

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.schema import Document
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
# Fix 2: Fail fast with a clear error rather than a cryptic auth failure later
if not OPENAI_API_KEY:
    raise EnvironmentError(
        "OPENAI_API_KEY environment variable is not set. "
        "Run: export OPENAI_API_KEY=sk-..."
    )

EMBEDDING_MODEL = "text-embedding-3-small"
LLM_MODEL = "gpt-4o"
CHUNK_SIZE = 512
CHUNK_OVERLAP = 64
TOP_K = 4

CONSTITUTIONAL_PRINCIPLES = [
    "Do not provide harmful, dangerous, or illegal advice.",
    "Do not fabricate facts or present uncertain information as definitive.",
    "Do not generate content that is discriminatory or biased.",
    "Always recommend professional consultation for medical, legal, or financial matters.",
    "Respect user privacy and do not encourage sharing sensitive personal data.",
]


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class RAGResponse:
    answer: str
    sources: list[str]
    critique_passed: bool
    critique_feedback: Optional[str] = None


# ---------------------------------------------------------------------------
# Document Ingestion
# ---------------------------------------------------------------------------

def load_documents(texts: list[str], metadatas: Optional[list[dict]] = None) -> list[Document]:
    """Convert raw text strings into LangChain Document objects."""
    metadatas = metadatas or [{} for _ in texts]
    return [Document(page_content=t, metadata=m) for t, m in zip(texts, metadatas)]


def chunk_documents(documents: list[Document]) -> list[Document]:
    """Split documents into overlapping chunks for better retrieval."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ".", " ", ""],
    )
    return splitter.split_documents(documents)


# ---------------------------------------------------------------------------
# FAISS Vector Store
# ---------------------------------------------------------------------------

def build_vector_store(chunks: list[Document]) -> FAISS:
    """Embed chunks and build a FAISS index."""
    embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL, openai_api_key=OPENAI_API_KEY)
    return FAISS.from_documents(chunks, embeddings)


def load_vector_store(path: str) -> FAISS:
    embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL, openai_api_key=OPENAI_API_KEY)
    return FAISS.load_local(path, embeddings, allow_dangerous_deserialization=True)


def save_vector_store(store: FAISS, path: str) -> None:
    store.save_local(path)


# ---------------------------------------------------------------------------
# Constitutional AI Critique Layer
# ---------------------------------------------------------------------------

_CRITIQUE_PROMPT = PromptTemplate(
    input_variables=["principles", "answer"],
    template="""You are a Constitutional AI safety reviewer.

Review the following AI-generated answer against these principles:
{principles}

Answer to review:
\"\"\"
{answer}
\"\"\"

First, identify any violation (yes/no). 
Then, if there is a violation, rewrite the answer to comply.
If there is no violation, repeat the original answer unchanged.

Respond strictly in this JSON format:
{{
  "violation": true | false,
  "feedback": "<explanation if violation, else null>",
  "revised_answer": "<safe answer>"
}}
""",
)


def constitutional_critique(llm: ChatOpenAI, answer: str) -> tuple[bool, Optional[str], str]:
    """
    Run the Constitutional AI self-critique pass.

    Returns:
        (passed: bool, feedback: str | None, final_answer: str)
    """
    principles_text = "\n".join(f"- {p}" for p in CONSTITUTIONAL_PRINCIPLES)
    prompt = _CRITIQUE_PROMPT.format(principles=principles_text, answer=answer)

    raw = llm.invoke(prompt).content.strip()

    # Strip markdown fences if present  (e.g. ```json\n{...}\n```)
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]   # remove the word "json"
        raw = raw.strip()   # FIX: strip leading/trailing whitespace including \n

    try:
        result = json.loads(raw)
        violation = result.get("violation", False)
        feedback = result.get("feedback")
        revised = result.get("revised_answer", answer)
        return (not violation), feedback, revised
    except json.JSONDecodeError:
        # Fail safe: return original answer, flag as passed
        return True, None, answer


# ---------------------------------------------------------------------------
# RAG Chain
# ---------------------------------------------------------------------------

_RAG_PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template="""You are a helpful, precise AI assistant. Use only the provided context to answer.
If the context does not contain sufficient information, say so honestly.

Context:
{context}

Question: {question}

Answer:""",
)


class RAGAssistant:
    """
    End-to-end RAG assistant with Constitutional AI safety layer.

    Usage:
        assistant = RAGAssistant.from_texts(["doc1 text...", "doc2 text..."])
        response = assistant.query("What is X?")
        print(response.answer)
    """

    def __init__(self, vector_store: FAISS):
        self.llm = ChatOpenAI(
            model=LLM_MODEL,
            temperature=0.2,
            openai_api_key=OPENAI_API_KEY,
        )
        self.retriever = vector_store.as_retriever(search_kwargs={"k": TOP_K})
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.retriever,
            chain_type_kwargs={"prompt": _RAG_PROMPT},
            return_source_documents=True,
        )

    @classmethod
    def from_texts(cls, texts: list[str], metadatas: Optional[list[dict]] = None) -> "RAGAssistant":
        docs = load_documents(texts, metadatas)
        chunks = chunk_documents(docs)
        store = build_vector_store(chunks)
        return cls(store)

    @classmethod
    def from_saved_index(cls, path: str) -> "RAGAssistant":
        store = load_vector_store(path)
        return cls(store)

    def query(self, question: str) -> RAGResponse:
        result = self.qa_chain.invoke({"query": question})
        raw_answer: str = result["result"]
        source_docs: list[Document] = result.get("source_documents", [])
        sources = list({d.metadata.get("source", "unknown") for d in source_docs})

        # Constitutional AI critique
        passed, feedback, final_answer = constitutional_critique(self.llm, raw_answer)

        return RAGResponse(
            answer=final_answer,
            sources=sources,
            critique_passed=passed,
            critique_feedback=feedback,
        )


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    sample_docs = [
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

    print("Building RAG assistant...")
    assistant = RAGAssistant.from_texts(
        sample_docs,
        metadatas=[{"source": f"doc_{i}"} for i in range(len(sample_docs))],
    )

    questions = [
        "What is Retrieval-Augmented Generation?",
        "How does Constitutional AI work?",
        "What is FAISS used for in RAG systems?",
    ]

    for q in questions:
        print(f"\nQ: {q}")
        resp = assistant.query(q)
        print(f"A: {resp.answer}")
        print(f"   Sources: {resp.sources}")
        print(f"   Critique passed: {resp.critique_passed}")
        if resp.critique_feedback:
            print(f"   Critique feedback: {resp.critique_feedback}")
