from typing import Optional
from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    message: str = Field(..., description="User message")
    session_id: str = Field(default="default", description="Session ID for conversation tracking")
    agent_mode: str = Field(default="auto", description="Agent mode: react, plan_execute, or auto")
    use_rag: bool = Field(default=False, description="Whether to use RAG knowledge base")
    collection_name: str = Field(default="default", description="RAG collection name to search")


class IntermediateStep(BaseModel):
    tool: str = ""
    tool_input: str = ""
    output: str = ""


class ChatResponse(BaseModel):
    response: str
    intermediate_steps: list[IntermediateStep] = []
    sources: list[str] = []
    agent_mode: str = ""


class DocumentUploadResponse(BaseModel):
    collection_name: str
    num_chunks: int
    message: str


class CollectionInfo(BaseModel):
    name: str
    count: int


class HealthResponse(BaseModel):
    status: str
    llm_provider: str
    model: str
