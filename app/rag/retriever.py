from typing import Optional

from langchain_core.documents import Document

from app.rag.vector_store import VectorStoreManager


class RAGRetriever:
    """High-level retrieval interface that wraps VectorStoreManager."""

    def __init__(self, vector_store: VectorStoreManager):
        self._store = vector_store

    def retrieve(
        self, query: str, collection_name: str, k: int = 4
    ) -> list[Document]:
        """Retrieve relevant documents for a query."""
        return self._store.similarity_search(query, collection_name, k=k)

    def retrieve_as_context(
        self, query: str, collection_name: str, k: int = 4
    ) -> str:
        """Retrieve documents and format them as a context string for the LLM."""
        docs = self.retrieve(query, collection_name, k=k)
        if not docs:
            return ""

        parts = ["以下是从知识库中检索到的相关内容:\n"]
        for i, doc in enumerate(docs, 1):
            source = doc.metadata.get("source", "未知")
            page = doc.metadata.get("page", "")
            ref = f"[来源: {source}"
            if page:
                ref += f", 第{page}页"
            ref += "]"
            parts.append(f"--- 片段 {i} {ref} ---\n{doc.page_content}\n")
        return "\n".join(parts)
