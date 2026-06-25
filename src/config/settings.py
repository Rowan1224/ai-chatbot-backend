"""Application settings and configuration loader."""

from typing import Any, Dict, List, Optional

import yaml
from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # LLM Provider Configuration
    llm_provider: str = Field(
        default="openai", description="LLM provider: openai, azure, anthropic"
    )

    # OpenAI Configuration (optional - only needed if llm_provider="openai")
    openai_api_key: Optional[str] = Field(default=None, description="OpenAI API key")

    # Azure OpenAI Configuration (optional - only needed if llm_provider="azure")
    azure_openai_api_key: Optional[str] = Field(default=None, description="Azure OpenAI API key")
    azure_openai_endpoint: Optional[str] = Field(
        default=None, description="Azure OpenAI endpoint"
    )
    openai_api_version: str = Field(
        default="2024-12-01-preview", description="Azure OpenAI API version"
    )

    # Anthropic Configuration (optional - only needed if llm_provider="anthropic")
    anthropic_api_key: Optional[str] = Field(default=None, description="Anthropic API key")

    # Model Configuration
    chat_model: str = Field(default="gpt-4", description="Chat model name")
    embedding_model: str = Field(
        default="text-embedding-3-small", description="Embedding model name"
    )

    # LangSmith Configuration (optional - for tracing)
    langsmith_tracing: bool = Field(default=False, description="Enable LangSmith tracing")
    langsmith_endpoint: Optional[str] = Field(
        default=None, description="LangSmith API endpoint"
    )
    langsmith_api_key: Optional[str] = Field(default=None, description="LangSmith API key")
    langsmith_project: str = Field(default="default", description="LangSmith project name")

    # MongoDB Configuration
    mongodb_url: str = Field(..., description="MongoDB connection URL")
    mongodb_db_name: str = Field(default="chatbot", description="MongoDB database name")

    # Redis Configuration
    redis_url: str = Field(default="redis://localhost:6379", description="Redis connection URL")
    redis_host: str = Field(default="localhost", description="Redis connection Host")
    redis_port: int = Field(default=6379, description="Redis connection Port")
    redis_password: Optional[str] = Field(default=None, description="Redis connection Password")
    use_redis_checkpointer: bool = Field(
        default=True,
        description="Use Redis for LangGraph checkpointing (False = InMemory for dev/test)"
    )

    # API Configuration
    api_key: str = Field(default="dev-api-key-12345", description="API key for authentication")
    api_host: str = Field(default="0.0.0.0", description="API host")
    api_port: int = Field(default=8000, description="API port")

    # Application Settings
    config_path: str = Field(
        default="src/config/prompt_config.yaml", description="Path to prompt configuration file"
    )
    log_level: str = Field(default="INFO", description="Logging level")

    # Vector Search Configuration
    vector_search_provider: str = Field(
        default="atlas", description="Vector search provider (atlas, local, pgvector)"
    )

    # Duplicate Detection
    duplicate_detection_enabled: bool = Field(
        default=True, description="Enable duplicate detection"
    )
    semantic_search_enabled: bool = Field(
        default=True, description="Enable semantic similarity search (requires embeddings)"
    )
    similarity_threshold: float = Field(
        default=0.85, description="Similarity threshold for duplicate detection"
    )
    lookback_days: int = Field(
        default=30, description="Number of days to look back for duplicates"
    )

    @model_validator(mode="after")
    def validate_llm_config(self) -> "Settings":
        """Validate that required fields exist for selected LLM provider."""
        if self.llm_provider == "openai":
            if not self.openai_api_key:
                raise ValueError("OPENAI_API_KEY is required when LLM_PROVIDER=openai")

        elif self.llm_provider == "azure":
            if not self.azure_openai_api_key:
                raise ValueError("AZURE_OPENAI_API_KEY is required when LLM_PROVIDER=azure")
            if not self.azure_openai_endpoint:
                raise ValueError("AZURE_OPENAI_ENDPOINT is required when LLM_PROVIDER=azure")

        elif self.llm_provider == "anthropic":
            if not self.anthropic_api_key:
                raise ValueError("ANTHROPIC_API_KEY is required when LLM_PROVIDER=anthropic")

        else:
            raise ValueError(
                f"Invalid LLM_PROVIDER: {self.llm_provider}. "
                "Must be one of: openai, azure, anthropic"
            )

        return self


class PromptConfig:
    """Prompt configuration loaded from YAML file."""

    def __init__(self, config_path: str) -> None:
        """Initialize prompt configuration from YAML file."""
        with open(config_path, "r") as f:
            self.config: Dict[str, Any] = yaml.safe_load(f)

    @property
    def config_version(self) -> str:
        """Get configuration version."""
        return self.config.get("config_version", "v1.0")

    @property
    def system_prompt(self) -> str:
        """Get system prompt."""
        return self.config.get("system_prompt", "")

    @property
    def fields(self) -> List[Dict[str, Any]]:
        """Get field definitions."""
        return self.config.get("fields", [])

    @property
    def duplicate_detection(self) -> Dict[str, Any]:
        """Get duplicate detection settings."""
        return self.config.get("duplicate_detection", {})

    @property
    def privacy(self) -> Dict[str, Any]:
        """Get privacy settings."""
        return self.config.get("privacy", {})


# Global settings instance
settings = Settings()
prompt_config = PromptConfig(settings.config_path)

# Made with Bob
