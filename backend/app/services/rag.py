from fastapi import HTTPException
from langchain_core.prompts import ChatPromptTemplate

from backend.app.schemas.rag import RagRequest, RagResponse
from backend.app.services.ai_config import (
    AIProviderConfigError,
    normalize_provider,
    require_gemini_api_key,
)
from backend.app.services.vector_store import similarity_search
from backend.app.settings import settings


def _get_llm():
    provider = normalize_provider(settings.LLM_PROVIDER)

    if provider == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI

        return ChatGoogleGenerativeAI(
            model=settings.GEMINI_CHAT_MODEL,
            google_api_key=require_gemini_api_key(),
            temperature=0,
        )

    if provider == "ollama":
        from langchain_ollama import ChatOllama

        return ChatOllama(
            model=settings.OLLAMA_CHAT_MODEL,
            base_url=settings.OLLAMA_BASE_URL,
            temperature=0,
        )

    raise AIProviderConfigError(
        f"Unsupported LLM_PROVIDER '{settings.LLM_PROVIDER}'. "
        "Use 'ollama' or 'gemini'."
    )


def query_rag(
    rag_request: RagRequest,
    user_id: int,
) -> RagResponse:

    question = rag_request.question.strip()

    if not question:
        raise HTTPException(
            status_code=400,
            detail="Question cannot be empty",
        )

    top_k = max(
        1,
        min(rag_request.top_k or 5, 20),
    )

    try:
        results = similarity_search(
            query=question,
            user_id=user_id,
            k=top_k,
        )
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Document retrieval failed: {exc}",
        ) from exc

    if not results:
        return RagResponse(
            answer="I don't know based on the uploaded documents.",
            sources=[],
        )

    context_parts = []
    sources = []

    for document, score in results:

        metadata = document.metadata or {}

        document_id = metadata.get("document_id")
        filename = metadata.get(
            "filename",
            "Unknown",
        )
        page_number = metadata.get(
            "page_number",
            "Unknown",
        )

        context_parts.append(
            f"""
Document: {filename}
Page: {page_number}

Content:
{document.page_content}
"""
        )

        sources.append(
            {
                "id": document_id,
                "filename": filename,
                "page_number": page_number,
                "content": document.page_content,
                "score": float(score),
            }
        )

    context = "\n\n---\n\n".join(context_parts)

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """
You are an enterprise document assistant.

Answer the user's question ONLY using the supplied
document context.

Rules:

- Do not invent facts.
- If the answer is not present in the context,
  say you don't know.
- Use only information supported by the context.
- Mention the relevant document filename and page
  when appropriate.
- Keep the answer clear and concise.

Document context:

{context}
""",
            ),
            (
                "human",
                "{question}",
            ),
        ]
    )

    try:

        llm = _get_llm()

        messages = prompt.invoke(
            {
                "context": context,
                "question": question,
            }
        )

        response = llm.invoke(messages)

        content = response.content

        if isinstance(content, str):
            answer = content.strip()
        elif isinstance(content, list):
            parts = []
            for block in content:
                if isinstance(block, dict):
                    text = block.get("text")
                    if text:
                        parts.append(str(text))
                elif isinstance(block, str):
                    parts.append(block)
            answer = "\n".join(parts).strip()
        else:
            answer = str(content).strip()

        return RagResponse(
            answer=answer,
            sources=sources,
        )

    except Exception as exc:

        raise HTTPException(
            status_code=500,
            detail=f"RAG generation failed: {exc}",
        ) from exc
