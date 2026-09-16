from pathlib import Path
from typing import Iterable

from langchain_chroma import Chroma
from langchain_core.documents import Document as LangChainDocument

from backend.app.services.ai_config import (
    AIProviderConfigError,
    normalize_provider,
    require_gemini_api_key,
)
from backend.app.settings import settings


COLLECTION_NAME = "enterprise_documents"


def _get_embeddings():
    provider = normalize_provider(settings.EMBEDDING_PROVIDER)

    if provider == "gemini":
        from langchain_google_genai import GoogleGenerativeAIEmbeddings

        return GoogleGenerativeAIEmbeddings(
            model=settings.GEMINI_EMBEDDING_MODEL,
            google_api_key=require_gemini_api_key(),
        )

    if provider == "ollama":
        from langchain_ollama import OllamaEmbeddings

        return OllamaEmbeddings(
            model=settings.OLLAMA_EMBEDDING_MODEL,
            base_url=settings.OLLAMA_BASE_URL,
        )

    raise AIProviderConfigError(
        f"Unsupported EMBEDDING_PROVIDER '{settings.EMBEDDING_PROVIDER}'. "
        "Use 'ollama' or 'gemini'."
    )


def get_embeddings():
    return _get_embeddings()


def get_vector_store() -> Chroma:
    Path(settings.CHROMA_PERSIST_DIRECTORY).mkdir(
        parents=True,
        exist_ok=True,
    )

    return Chroma(
        collection_name=COLLECTION_NAME,
        persist_directory=settings.CHROMA_PERSIST_DIRECTORY,
        embedding_function=get_embeddings(),
    )


def add_chunks(chunks: Iterable[LangChainDocument]) -> list[str]:
    chunks = list(chunks)

    if not chunks:
        return []

    vector_store = get_vector_store()

    return vector_store.add_documents(chunks)


def delete_document_vectors(document_id: int) -> None:
    vector_store = get_vector_store()

    collection = vector_store._collection

    result = collection.get(
        where={"document_id": int(document_id)}
    )

    ids = result.get("ids", [])

    if ids:
        collection.delete(ids=ids)


def similarity_search(
    query: str,
    user_id: int,
    k: int = 5,
):
    vector_store = get_vector_store()

    return vector_store.similarity_search_with_score(
        query,
        k=k,
        filter={"user_id": int(user_id)},
    )
