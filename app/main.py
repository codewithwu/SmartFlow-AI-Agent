import logging
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from sse_starlette.sse import EventSourceResponse

from app.config import settings
from app.schemas.models import (
    ChatRequest,
    ChatResponse,
    IntermediateStep,
    DocumentUploadResponse,
    CollectionInfo,
    HealthResponse,
)
from app.agent.supervisor import SupervisorAgent
from app.rag.document_processor import DocumentProcessor
from app.rag.vector_store import VectorStoreManager
from app.memory.short_term import clear_session, list_sessions

logger = logging.getLogger("smartflow")

# --- Global singletons (initialized lazily) ---
_supervisor: SupervisorAgent | None = None
_doc_processor: DocumentProcessor | None = None
_vector_store: VectorStoreManager | None = None


def get_supervisor() -> SupervisorAgent:
    global _supervisor
    if _supervisor is None:
        _supervisor = SupervisorAgent()
    return _supervisor


def get_doc_processor() -> DocumentProcessor:
    global _doc_processor
    if _doc_processor is None:
        _doc_processor = DocumentProcessor()
    return _doc_processor


def get_vector_store() -> VectorStoreManager:
    global _vector_store
    if _vector_store is None:
        _vector_store = VectorStoreManager()
    return _vector_store


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info(
        "SmartFlow AI Agent starting | provider=%s model=%s",
        settings.LLM_PROVIDER,
        settings.OPENAI_MODEL if settings.LLM_PROVIDER == "openai" else settings.OLLAMA_MODEL,
    )
    yield
    logger.info("SmartFlow AI Agent shutting down")


app = FastAPI(
    title="SmartFlow AI Agent",
    description="智能业务流助手 - 支持 ReAct / Plan-Execute / 多Agent协作 / RAG",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ======================== Chat ========================

@app.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Process a chat message through the agent system."""
    try:
        supervisor = get_supervisor()
        result = supervisor.invoke(
            query=request.message,
            session_id=request.session_id,
            mode=request.agent_mode,
            use_rag=request.use_rag,
            collection_name=request.collection_name,
        )
        steps = [
            IntermediateStep(
                tool=s.get("tool", ""),
                tool_input=s.get("tool_input", ""),
                output=s.get("output", ""),
            )
            for s in result.get("intermediate_steps", [])
        ]
        return ChatResponse(
            response=result["response"],
            intermediate_steps=steps,
            sources=result.get("sources", []),
            agent_mode=result.get("agent_mode", ""),
        )
    except Exception as e:
        logger.exception("Chat error")
        raise HTTPException(status_code=500, detail=f"Agent execution error: {e}")


# ======================== Documents / RAG ========================

@app.post("/api/documents/upload", response_model=DocumentUploadResponse)
async def upload_document(
    file: UploadFile = File(...),
    collection_name: str = Form(default="default"),
):
    """Upload a document to the knowledge base."""
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file provided")

    allowed_ext = (".pdf", ".txt", ".md")
    if not any(file.filename.lower().endswith(ext) for ext in allowed_ext):
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type. Allowed: {', '.join(allowed_ext)}",
        )

    try:
        content = await file.read()
        processor = get_doc_processor()
        docs = processor.load_bytes(content, file.filename)

        store = get_vector_store()
        num_chunks = store.add_documents(docs, collection_name)

        return DocumentUploadResponse(
            collection_name=collection_name,
            num_chunks=num_chunks,
            message=f"Successfully uploaded {file.filename}: {num_chunks} chunks indexed.",
        )
    except Exception as e:
        logger.exception("Document upload error")
        raise HTTPException(status_code=500, detail=f"Upload failed: {e}")


@app.get("/api/documents/collections", response_model=list[CollectionInfo])
async def list_collections():
    """List all knowledge base collections."""
    store = get_vector_store()
    collections = store.list_collections()
    return [CollectionInfo(name=c["name"], count=c["count"]) for c in collections]


@app.delete("/api/documents/collections/{name}")
async def delete_collection(name: str):
    """Delete a knowledge base collection."""
    store = get_vector_store()
    success = store.delete_collection(name)
    if not success:
        raise HTTPException(status_code=404, detail=f"Collection '{name}' not found")
    return {"message": f"Collection '{name}' deleted."}


# ======================== Memory ========================

@app.post("/api/memory/clear")
async def clear_memory(session_id: str = "default"):
    """Clear conversation memory for a session."""
    existed = clear_session(session_id)
    return {
        "message": f"Session '{session_id}' cleared."
        if existed
        else f"Session '{session_id}' not found (already empty)."
    }


@app.get("/api/memory/sessions")
async def get_sessions():
    """List all active sessions."""
    return {"sessions": list_sessions()}


# ======================== Health ========================

@app.get("/api/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    model = (
        settings.OPENAI_MODEL
        if settings.LLM_PROVIDER == "openai"
        else settings.OLLAMA_MODEL
    )
    return HealthResponse(
        status="ok",
        llm_provider=settings.LLM_PROVIDER,
        model=model,
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app.main:app",
        host=settings.API_HOST,
        port=settings.API_PORT,
        reload=True,
    )
