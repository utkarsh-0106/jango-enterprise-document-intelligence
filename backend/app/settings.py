import secrets

from pydantic import ConfigDict, model_validator
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    ENVIRONMENT: str = "development"
    DATABASE_URL: str = "sqlite:///./sql_app.db"
    SECRET_KEY: str | None = None
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30

    FRONTEND_ORIGIN: str = "http://localhost:5173"

    OLLAMA_BASE_URL: str = "http://localhost:11434"
    OLLAMA_EMBEDDING_MODEL: str = "nomic-embed-text"
    OLLAMA_CHAT_MODEL: str = "qwen3:8b"

    CHROMA_PERSIST_DIRECTORY: str = "./storage/chroma"
    DOCUMENT_STORAGE_DIRECTORY: str = "./storage/documents"

    model_config = ConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    @model_validator(mode="after")
    def validate_secret_key(self):
        if self.SECRET_KEY in (None, "", "your-secret-key-here"):
            if self.ENVIRONMENT.lower() == "production":
                raise ValueError(
                    "SECRET_KEY must be set to a non-placeholder value in production"
                )
            self.SECRET_KEY = secrets.token_urlsafe(32)
        return self


settings = Settings()
