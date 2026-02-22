import uuid
from typing import Optional

import chromadb
from chromadb.config import Settings as ChromaSettings

from app.config import settings
from app.llm.provider import get_embeddings


class LongTermMemory:
    """ChromaDB-based long-term semantic memory for storing important conversation
    summaries and user preferences."""

    COLLECTION_NAME = "long_term_memory"

    def __init__(self):
        self._client = chromadb.PersistentClient(
            path=settings.CHROMA_PERSIST_DIR,
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        self._collection = self._client.get_or_create_collection(
            name=self.COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )
        self._embeddings = None

    def _get_embeddings(self):
        if self._embeddings is None:
            self._embeddings = get_embeddings()
        return self._embeddings

    def save_memory(
        self, session_id: str, content: str, metadata: Optional[dict] = None
    ) -> str:
        """Save a piece of memory (e.g. conversation summary) to long-term storage."""
        doc_id = str(uuid.uuid4())
        meta = {"session_id": session_id}
        if metadata:
            meta.update(metadata)

        embedding = self._get_embeddings().embed_query(content)
        self._collection.add(
            ids=[doc_id],
            embeddings=[embedding],
            documents=[content],
            metadatas=[meta],
        )
        return doc_id

    def search_memory(
        self, query: str, session_id: Optional[str] = None, k: int = 3
    ) -> list[dict]:
        """Search long-term memory by semantic similarity."""
        where_filter = {"session_id": session_id} if session_id else None
        embedding = self._get_embeddings().embed_query(query)

        results = self._collection.query(
            query_embeddings=[embedding],
            n_results=k,
            where=where_filter,
        )

        memories = []
        if results and results["documents"]:
            for doc, meta, dist in zip(
                results["documents"][0],
                results["metadatas"][0],
                results["distances"][0],
            ):
                memories.append(
                    {"content": doc, "metadata": meta, "score": 1 - dist}
                )
        return memories

    def clear(self, session_id: Optional[str] = None) -> None:
        """Clear memories. If session_id given, only clear that session."""
        if session_id:
            existing = self._collection.get(where={"session_id": session_id})
            if existing["ids"]:
                self._collection.delete(ids=existing["ids"])
        else:
            self._client.delete_collection(self.COLLECTION_NAME)
            self._collection = self._client.get_or_create_collection(
                name=self.COLLECTION_NAME,
                metadata={"hnsw:space": "cosine"},
            )
