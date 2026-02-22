import os
from typing import Optional

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


class DocumentProcessor:
    """Loads and chunks PDF / TXT documents for RAG ingestion."""

    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", "。", "！", "？", ".", " ", ""],
        )

    def load_file(self, file_path: str) -> list[Document]:
        """Load a file and return chunked Documents with metadata."""
        ext = os.path.splitext(file_path)[1].lower()
        if ext == ".pdf":
            return self._load_pdf(file_path)
        elif ext in (".txt", ".md"):
            return self._load_text(file_path)
        else:
            raise ValueError(f"Unsupported file type: {ext}. Supported: .pdf, .txt, .md")

    def load_bytes(self, content: bytes, filename: str) -> list[Document]:
        """Load from raw bytes (for file uploads)."""
        ext = os.path.splitext(filename)[1].lower()
        if ext == ".pdf":
            return self._load_pdf_bytes(content, filename)
        elif ext in (".txt", ".md"):
            text = content.decode("utf-8", errors="ignore")
            doc = Document(page_content=text, metadata={"source": filename})
            return self.splitter.split_documents([doc])
        else:
            raise ValueError(f"Unsupported file type: {ext}")

    def _load_pdf(self, file_path: str) -> list[Document]:
        from pypdf import PdfReader

        reader = PdfReader(file_path)
        docs = []
        for i, page in enumerate(reader.pages):
            text = page.extract_text() or ""
            if text.strip():
                docs.append(
                    Document(
                        page_content=text,
                        metadata={"source": os.path.basename(file_path), "page": i + 1},
                    )
                )
        return self.splitter.split_documents(docs)

    def _load_pdf_bytes(self, content: bytes, filename: str) -> list[Document]:
        import io
        from pypdf import PdfReader

        reader = PdfReader(io.BytesIO(content))
        docs = []
        for i, page in enumerate(reader.pages):
            text = page.extract_text() or ""
            if text.strip():
                docs.append(
                    Document(
                        page_content=text,
                        metadata={"source": filename, "page": i + 1},
                    )
                )
        return self.splitter.split_documents(docs)

    def _load_text(self, file_path: str) -> list[Document]:
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()
        doc = Document(
            page_content=text,
            metadata={"source": os.path.basename(file_path)},
        )
        return self.splitter.split_documents([doc])
