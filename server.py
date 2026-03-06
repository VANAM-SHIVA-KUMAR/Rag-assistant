"""
FastAPI server for the Constitutional AI-Aligned RAG Assistant
--------------------------------------------------------------
Endpoints:
  POST /ingest   - Add documents to the knowledge base
  POST /query    - Query the RAG assistant
  GET  /health   - Health check
"""

import os
import asyncio
from typing import Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from rag_pipeline import (
    RAGAssistant, RAGResponse,
    load_documents, chunk_documents, build_vector_store,
)
from langchain_community.vectorstores import FAISS

app = FastAPI(
    title="Constitutional AI-Aligned RAG Assistant",
    description="Production RAG pipeline with Constitutional AI safety layer and RAGAS evaluation.",
    version="1.0.0",
)

# Global state
_assistant: Optional[RAGAssistant] = None
_corpus: list[str] = []
_metadatas: list[dict] = []
_vector_store: Optional[FAISS] = None


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------

class IngestRequest(BaseModel):
    texts: list[str] = Field(..., min_length=1)
    metadatas: Optional[list[dict]] = None

class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000)

class QueryResponse(BaseModel):
    answer: str
    sources: list[str]
    critique_passed: bool
    critique_feedback: Optional[str] = None


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
def health():
    return {"status": "ok", "index_size": len(_corpus)}


@app.post("/ingest")
def ingest(req: IngestRequest):
    """
    Add documents to the knowledge base.
    Fix 4: Uses FAISS.merge_from() to incrementally add new chunks
    instead of rebuilding the entire index from scratch each time.
    """
    global _assistant, _corpus, _metadatas, _vector_store

    metas = req.metadatas or [{"source": f"doc_{len(_corpus)+i}"} for i in range(len(req.texts))]

    # Build a small index just for the NEW documents
    new_docs   = load_documents(req.texts, metas)
    new_chunks = chunk_documents(new_docs)
    new_store  = build_vector_store(new_chunks)

    if _vector_store is None:
        # First ingest — use as-is
        _vector_store = new_store
    else:
        # Incremental add: merge new index into existing one (O(new) not O(all))
        _vector_store.merge_from(new_store)

    _corpus.extend(req.texts)
    _metadatas.extend(metas)
    _assistant = RAGAssistant(_vector_store)

    return {"message": f"Ingested {len(req.texts)} documents. Total corpus: {len(_corpus)}."}


# Fix 3: async endpoint with run_in_executor so the event loop isn't blocked
# during the (potentially slow) LLM + critique calls.
@app.post("/query", response_model=QueryResponse)
async def query(req: QueryRequest):
    if _assistant is None:
        raise HTTPException(status_code=400, detail="No documents ingested yet. Call /ingest first.")

    loop = asyncio.get_running_loop()
    response: RAGResponse = await loop.run_in_executor(
        None, lambda: _assistant.query(req.question)
    )
    return QueryResponse(
        answer=response.answer,
        sources=response.sources,
        critique_passed=response.critique_passed,
        critique_feedback=response.critique_feedback,
    )


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)
