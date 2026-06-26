"""LLM provider factory - returns appropriate LLM based on configuration."""


from langchain_anthropic import ChatAnthropic
from langchain_core.language_models import BaseChatModel
from langchain_openai import (
    AzureChatOpenAI,
    AzureOpenAIEmbeddings,
    ChatOpenAI,
    OpenAIEmbeddings,
)

from src.config.settings import app_config, settings


def get_llm(model: str | None = None) -> BaseChatModel:
    """
    Get LLM instance based on configured provider.

    Args:
        model: Model name (provider-specific). If None,
               uses settings.chat_model

    Returns:
        BaseChatModel instance

    Raises:
        ValueError: If provider is not supported
    """
    model = model or app_config.chat_model

    if settings.llm_provider == "openai":
        return ChatOpenAI(
            api_key=settings.openai_api_key,
            model=model,
            temperature=0,  # Deterministic for structured output
        )

    elif settings.llm_provider == "azure":
        return AzureChatOpenAI(
            api_key=settings.azure_openai_api_key,
            azure_endpoint=settings.azure_openai_endpoint,
            api_version=settings.openai_api_version,
            deployment_name=model,  # Azure uses deployment names
        )

    elif settings.llm_provider == "anthropic":
        return ChatAnthropic(
            api_key=settings.anthropic_api_key,
            model=model,
            temperature=0,
        )

    else:
        raise ValueError(
            f"Unsupported LLM provider: {settings.llm_provider}. "
            "Supported providers: openai, azure, anthropic"
        )


def get_embedding_model(model: str | None = None):
    """
    Get embedding model for vector search.

    Args:
        model: Embedding model name. If None, uses
               settings.embedding_model

    Returns:
        Embedding model instance

    Raises:
        ValueError: If embedding model cannot be configured
    """
    model = model or app_config.embedding_model

    if settings.llm_provider == "openai":
        return OpenAIEmbeddings(
            api_key=settings.openai_api_key,
            model=model,
        )

    elif settings.llm_provider == "azure":
        return AzureOpenAIEmbeddings(
            api_key=settings.azure_openai_api_key,
            azure_endpoint=settings.azure_openai_endpoint,
            deployment=model,  # Azure deployment name
        )

    else:
        # Fallback to OpenAI for other providers
        # Anthropic doesn't have embeddings, so we use OpenAI
        if settings.openai_api_key:
            return OpenAIEmbeddings(
                api_key=settings.openai_api_key,
                model=model,
            )
        raise ValueError(
            "Embedding model requires OpenAI or Azure OpenAI configuration. "
            "For Anthropic provider, please also set OPENAI_API_KEY for embeddings."
        )


# Made with Bob
