from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.embeddings import Embeddings

from app.config import settings


def get_chat_model() -> BaseChatModel:
    """Get the chat model based on the configured provider."""
    if settings.LLM_PROVIDER == "ollama":
        from langchain_ollama import ChatOllama

        return ChatOllama(
            base_url=settings.OLLAMA_BASE_URL,
            model=settings.OLLAMA_MODEL,
            temperature=settings.LLM_TEMPERATURE,
            num_predict=settings.LLM_MAX_TOKENS,
        )
    else:
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            api_key=settings.OPENAI_API_KEY,
            base_url=settings.OPENAI_BASE_URL,
            model=settings.OPENAI_MODEL,
            temperature=settings.LLM_TEMPERATURE,
            max_tokens=settings.LLM_MAX_TOKENS,
        )


def get_embeddings() -> Embeddings:
    """Get the embedding model based on the configured provider."""
    if settings.LLM_PROVIDER == "ollama":
        from langchain_ollama import OllamaEmbeddings

        return OllamaEmbeddings(
            base_url=settings.OLLAMA_BASE_URL,
            model=settings.OLLAMA_EMBEDDING_MODEL,
        )
    else:
        from langchain_openai import OpenAIEmbeddings

        return OpenAIEmbeddings(
            api_key=settings.OPENAI_API_KEY,
            base_url=settings.OPENAI_BASE_URL,
            model=settings.OPENAI_EMBEDDING_MODEL,
        )
