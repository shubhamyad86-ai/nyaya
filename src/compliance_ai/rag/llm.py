"""LLM providers: local Ollama (free) or Azure OpenAI, chosen by config or env."""
from __future__ import annotations

import os

from langchain_core.language_models.chat_models import BaseChatModel


def get_llm(provider: str, ollama_model: str = "llama3.1:8b") -> BaseChatModel | None:
    """Return a chat model, or None for retrieval-only mode.

    Azure OpenAI reads AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_API_KEY,
    AZURE_OPENAI_DEPLOYMENT and AZURE_OPENAI_API_VERSION from the environment.
    """
    provider = os.environ.get("LLM_PROVIDER", provider).lower()
    if provider == "ollama":
        from langchain_ollama import ChatOllama

        return ChatOllama(
            model=os.environ.get("OLLAMA_MODEL", ollama_model),
            base_url=os.environ.get("OLLAMA_HOST", "http://localhost:11434"),
            temperature=0,
        )
    if provider == "azure":
        from langchain_openai import AzureChatOpenAI

        return AzureChatOpenAI(
            azure_deployment=os.environ["AZURE_OPENAI_DEPLOYMENT"],
            api_version=os.environ.get("AZURE_OPENAI_API_VERSION", "2024-10-21"),
            temperature=0,
        )
    if provider == "none":
        return None
    raise ValueError(f"unknown LLM provider: {provider}")
