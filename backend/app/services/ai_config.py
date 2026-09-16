from backend.app.settings import settings


class AIProviderConfigError(ValueError):
    """Raised when the selected AI provider is misconfigured."""


def normalize_provider(value: str) -> str:
    return (value or "").strip().lower()


def require_gemini_api_key() -> str:
    key = (settings.GEMINI_API_KEY or "").strip()
    if not key:
        raise AIProviderConfigError(
            "GEMINI_API_KEY must be set when LLM_PROVIDER or "
            "EMBEDDING_PROVIDER is 'gemini'. Ollama is not used as a fallback."
        )
    return key
