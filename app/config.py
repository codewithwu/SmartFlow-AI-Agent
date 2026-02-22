from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # LLM Provider: "openai" or "ollama"
    LLM_PROVIDER: str = "openai"

    # OpenAI
    OPENAI_API_KEY: str = ""
    OPENAI_BASE_URL: str = "https://api.openai.com/v1"
    OPENAI_MODEL: str = "gpt-4o-mini"
    OPENAI_EMBEDDING_MODEL: str = "text-embedding-3-small"

    # Ollama
    OLLAMA_BASE_URL: str = "http://localhost:11434"
    OLLAMA_MODEL: str = "llama3.1"
    OLLAMA_EMBEDDING_MODEL: str = "nomic-embed-text"

    # LLM parameters
    LLM_TEMPERATURE: float = 0.7
    LLM_MAX_TOKENS: int = 4096

    # ChromaDB
    CHROMA_PERSIST_DIR: str = "./data/chroma_db"

    # Memory
    SHORT_TERM_MAX_MESSAGES: int = 20

    # API
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000

    # Frontend
    BACKEND_URL: str = "http://localhost:8000"

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}


settings = Settings()
