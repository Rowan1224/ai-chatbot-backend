"""Application settings and configuration loader."""

from typing import Any

import yaml
from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Infrastructure and secret settings loaded from environment variables.

    Only values that are environment-specific (infrastructure endpoints,
    credentials) or secret (API keys) live here.  Behavioural configuration
    such as model selection, feature flags, and tuning thresholds lives in
    app_config.yaml.  The LLM system prompt lives in prompt_config.yaml.
    """

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # LLM Provider — infrastructure-level switch that determines which
    # credentials are required.  Model names live in app_config.yaml.
    llm_provider: str = Field(
        default="openai", description="LLM provider: openai, azure, anthropic"
    )

    # OpenAI credentials (required when llm_provider=openai)
    openai_api_key: str | None = Field(default=None, description="OpenAI API key")

    # Azure OpenAI credentials (required when llm_provider=azure)
    azure_openai_api_key: str | None = Field(default=None, description="Azure OpenAI API key")
    azure_openai_endpoint: str | None = Field(
        default=None, description="Azure OpenAI endpoint"
    )
    openai_api_version: str = Field(
        default="2024-12-01-preview", description="Azure OpenAI API version"
    )

    # Anthropic credentials (required when llm_provider=anthropic)
    anthropic_api_key: str | None = Field(default=None, description="Anthropic API key")

    # LangSmith tracing (optional)
    langsmith_tracing: bool = Field(default=False, description="Enable LangSmith tracing")
    langsmith_endpoint: str | None = Field(
        default=None, description="LangSmith API endpoint"
    )
    langsmith_api_key: str | None = Field(default=None, description="LangSmith API key")
    langsmith_project: str = Field(default="default", description="LangSmith project name")

    # PostgreSQL — connection string includes credentials and host
    postgresql_url: str = Field(..., description="PostgreSQL connection URL")

    # Redis — infrastructure endpoint
    redis_url: str = Field(default="redis://localhost:6379", description="Redis connection URL")
    redis_host: str = Field(default="localhost", description="Redis host")
    redis_port: int = Field(default=6379, description="Redis port")
    redis_password: str | None = Field(default=None, description="Redis password")
    use_redis_checkpointer: bool = Field(
        default=True,
        description="Use Redis for LangGraph checkpointing (False = InMemory for dev/test)"
    )

    # API secret key — no default so it must be set explicitly in production
    api_key: str = Field(..., description="API key for authentication")

    # Server bind config
    api_host: str = Field(default="0.0.0.0", description="API host")
    api_port: int = Field(default=8000, description="API port")

    # Paths to YAML config files
    prompt_config_path: str = Field(
        default="src/config/prompt_config.yaml",
        description="Path to prompt configuration YAML",
    )
    app_config_path: str = Field(
        default="src/config/app_config.yaml",
        description="Path to application configuration YAML",
    )

    @model_validator(mode="after")
    def validate_llm_config(self) -> "Settings":
        """Validate that required credentials exist for the selected LLM provider."""
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
    """Prompt configuration loaded from prompt_config.yaml.

    Owns only the system prompt and its version.  Everything else
    (models, feature flags, thresholds, privacy) lives in AppConfig.
    """

    def __init__(self, config_path: str) -> None:
        with open(config_path) as f:
            self._config: dict[str, Any] = yaml.safe_load(f)

    @property
    def config_version(self) -> str:
        return self._config.get("config_version", "v1.0")

    @property
    def system_prompt(self) -> str:
        return self._config.get("system_prompt", "")


class AppConfig:
    """Application behaviour configuration loaded from app_config.yaml.

    Owns model selection, feature flags, tuning thresholds, and privacy
    settings.  No secrets; safe to commit to version control.
    """

    def __init__(self, config_path: str) -> None:
        with open(config_path) as f:
            self._config: dict[str, Any] = yaml.safe_load(f)

    # --- LLM models ---

    @property
    def chat_model(self) -> str:
        return self._config.get("llm", {}).get("chat_model", "gpt-4")

    @property
    def embedding_model(self) -> str:
        return self._config.get("llm", {}).get("embedding_model", "text-embedding-3-small")

    # --- Duplicate detection ---

    @property
    def duplicate_detection_enabled(self) -> bool:
        return self._config.get("duplicate_detection", {}).get("enabled", True)

    @property
    def semantic_search_enabled(self) -> bool:
        return self._config.get("duplicate_detection", {}).get("semantic_search_enabled", True)

    @property
    def similarity_threshold(self) -> float:
        return self._config.get("duplicate_detection", {}).get("similarity_threshold", 0.85)

    @property
    def lookback_days(self) -> int:
        return self._config.get("duplicate_detection", {}).get("lookback_days", 90)

    @property
    def fuzzy_threshold(self) -> float:
        return self._config.get("duplicate_detection", {}).get("fuzzy_threshold", 0.3)

    @property
    def candidate_limit(self) -> int:
        return self._config.get("duplicate_detection", {}).get("candidate_limit", 50)

    @property
    def vector_search_provider(self) -> str:
        return self._config.get("duplicate_detection", {}).get("vector_search_provider", "pgvector")

    # --- Privacy ---

    @property
    def pii_fields(self) -> list[str]:
        return self._config.get("privacy", {}).get("pii_fields", [])

    # --- CORS ---

    @property
    def rate_limit_rpm(self) -> int:
        return self._config.get(
            "rate_limiting", {}
        ).get("requests_per_minute", 30)

    @property
    def cors_origins(self) -> list[str]:
        return self._config.get("cors", {}).get("origins", [])

    # --- Logging ---

    @property
    def log_level(self) -> str:
        return self._config.get("logging", {}).get("level", "INFO")


# Global singletons — imported across the application
settings = Settings()
prompt_config = PromptConfig(settings.prompt_config_path)
app_config = AppConfig(settings.app_config_path)

# Made with Bob
