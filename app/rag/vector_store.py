from typing import Optional

import chromadb
from chromadb.config import Settings as ChromaSettings
from langchain_core.documents import Document

from app.config import settings
from app.llm.provider import get_embeddings


class VectorStoreManager:
    """Manages ChromaDB collections for RAG document storage and retrieval."""

    def __init__(self):
        self._client = chromadb.PersistentClient(
            path=settings.CHROMA_PERSIST_DIR,
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        self._embeddings = None

    def _get_embeddings(self):
        if self._embeddings is None:
            self._embeddings = get_embeddings()
        return self._embeddings

    def add_documents(self, docs: list[Document], collection_name: str) -> int:
        """Add documents to a named collection. Returns number of chunks added."""
        collection = self._client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        embeddings_model = self._get_embeddings()

        texts = [doc.page_content for doc in docs]
        metadatas = [doc.metadata for doc in docs]
        embeddings = embeddings_model.embed_documents(texts)
        ids = [f"{collection_name}_{i}" for i in range(collection.count(), collection.count() + len(docs))]

        collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=texts,
            metadatas=metadatas,
        )
        return len(docs)

    def similarity_search(
        self, query: str, collection_name: str, k: int = 4
    ) -> list[Document]:
        """Search a collection for documents similar to the query."""
        try:
            collection = self._client.get_collection(name=collection_name)
        except Exception:
            return []

        embedding = self._get_embeddings().embed_query(query)
        results = collection.query(query_embeddings=[embedding], n_results=k)

        docs = []
        if results and results["documents"]:
            for text, meta in zip(results["documents"][0], results["metadatas"][0]):
                docs.append(Document(page_content=text, metadata=meta))
        return docs

    def list_collections(self) -> list[dict]:
        """List all collections with document counts."""
        collections = self._client.list_collections()
        result = []
        for col in collections:
            c = self._client.get_collection(col.name)
            result.append({"name": col.name, "count": c.count()})
        return result

    def delete_collection(self, collection_name: str) -> bool:
        """Delete a collection by name."""
        try:
            self._client.delete_collection(collection_name)
            return True
        except Exception:
            return False
