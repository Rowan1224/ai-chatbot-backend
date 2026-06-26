"""Unit tests for src/core/llm.py — LLM provider factory."""

import pytest
from unittest.mock import MagicMock, patch

from src.core.llm import get_llm, get_embedding_model


class TestGetLlm:
    """Tests for get_llm() factory function."""

    def test_openai_provider_returns_chat_openai(self):
        with patch("src.core.llm.settings") as mock_settings, \
             patch("src.core.llm.app_config") as mock_cfg:
            mock_settings.llm_provider = "openai"
            mock_settings.openai_api_key = "test-key"
            mock_cfg.chat_model = "gpt-4"

            with patch("src.core.llm.ChatOpenAI") as mock_cls:
                mock_cls.return_value = MagicMock()
                result = get_llm()

            mock_cls.assert_called_once_with(
                api_key="test-key", model="gpt-4", temperature=0
            )
            assert result is mock_cls.return_value

    def test_openai_provider_with_explicit_model(self):
        with patch("src.core.llm.settings") as mock_settings, \
             patch("src.core.llm.app_config") as mock_cfg:
            mock_settings.llm_provider = "openai"
            mock_settings.openai_api_key = "test-key"
            mock_cfg.chat_model = "gpt-4"

            with patch("src.core.llm.ChatOpenAI") as mock_cls:
                mock_cls.return_value = MagicMock()
                get_llm(model="gpt-3.5-turbo")

            mock_cls.assert_called_once_with(
                api_key="test-key", model="gpt-3.5-turbo", temperature=0
            )

    def test_azure_provider_returns_azure_chat_openai(self):
        with patch("src.core.llm.settings") as mock_settings, \
             patch("src.core.llm.app_config") as mock_cfg:
            mock_settings.llm_provider = "azure"
            mock_settings.azure_openai_api_key = "azure-key"
            mock_settings.azure_openai_endpoint = "https://my.openai.azure.com/"
            mock_settings.openai_api_version = "2024-12-01-preview"
            mock_cfg.chat_model = "gpt-4-deployment"

            with patch("src.core.llm.AzureChatOpenAI") as mock_cls:
                mock_cls.return_value = MagicMock()
                result = get_llm()

            mock_cls.assert_called_once()
            assert result is mock_cls.return_value

    def test_anthropic_provider_returns_chat_anthropic(self):
        with patch("src.core.llm.settings") as mock_settings, \
             patch("src.core.llm.app_config") as mock_cfg:
            mock_settings.llm_provider = "anthropic"
            mock_settings.anthropic_api_key = "anth-key"
            mock_cfg.chat_model = "claude-3-opus-20240229"

            with patch("src.core.llm.ChatAnthropic") as mock_cls:
                mock_cls.return_value = MagicMock()
                result = get_llm()

            mock_cls.assert_called_once_with(
                api_key="anth-key",
                model="claude-3-opus-20240229",
                temperature=0,
            )
            assert result is mock_cls.return_value

    def test_unsupported_provider_raises_value_error(self):
        with patch("src.core.llm.settings") as mock_settings, \
             patch("src.core.llm.app_config") as mock_cfg:
            mock_settings.llm_provider = "unknown-provider"
            mock_cfg.chat_model = "some-model"

            with pytest.raises(ValueError, match="Unsupported LLM provider"):
                get_llm()

    def test_model_defaults_to_settings_chat_model(self):
        with patch("src.core.llm.settings") as mock_settings, \
             patch("src.core.llm.app_config") as mock_cfg:
            mock_settings.llm_provider = "openai"
            mock_settings.openai_api_key = "test-key"
            mock_cfg.chat_model = "gpt-4-turbo"

            with patch("src.core.llm.ChatOpenAI") as mock_cls:
                mock_cls.return_value = MagicMock()
                get_llm()  # no model arg

            call_kwargs = mock_cls.call_args
            assert call_kwargs.kwargs["model"] == "gpt-4-turbo"


class TestGetEmbeddingModel:
    """Tests for get_embedding_model() factory function."""

    def test_openai_provider_returns_openai_embeddings(self):
        with patch("src.core.llm.settings") as mock_settings, \
             patch("src.core.llm.app_config") as mock_cfg:
            mock_settings.llm_provider = "openai"
            mock_settings.openai_api_key = "test-key"
            mock_cfg.embedding_model = "text-embedding-3-small"

            with patch("src.core.llm.OpenAIEmbeddings") as mock_cls:
                mock_cls.return_value = MagicMock()
                result = get_embedding_model()

            mock_cls.assert_called_once_with(
                api_key="test-key", model="text-embedding-3-small"
            )
            assert result is mock_cls.return_value

    def test_openai_provider_with_explicit_model(self):
        with patch("src.core.llm.settings") as mock_settings, \
             patch("src.core.llm.app_config") as mock_cfg:
            mock_settings.llm_provider = "openai"
            mock_settings.openai_api_key = "test-key"
            mock_cfg.embedding_model = "text-embedding-3-small"

            with patch("src.core.llm.OpenAIEmbeddings") as mock_cls:
                mock_cls.return_value = MagicMock()
                get_embedding_model(model="text-embedding-ada-002")

            mock_cls.assert_called_once_with(
                api_key="test-key", model="text-embedding-ada-002"
            )

    def test_azure_provider_returns_azure_openai_embeddings(self):
        with patch("src.core.llm.settings") as mock_settings, \
             patch("src.core.llm.app_config") as mock_cfg:
            mock_settings.llm_provider = "azure"
            mock_settings.azure_openai_api_key = "azure-key"
            mock_settings.azure_openai_endpoint = "https://my.openai.azure.com/"
            mock_cfg.embedding_model = "text-embedding-ada-002"

            with patch("src.core.llm.AzureOpenAIEmbeddings") as mock_cls:
                mock_cls.return_value = MagicMock()
                result = get_embedding_model()

            mock_cls.assert_called_once()
            assert result is mock_cls.return_value

    def test_anthropic_with_openai_key_uses_openai_embeddings(self):
        """Anthropic has no embedding API; falls back to OpenAI if key present."""
        with patch("src.core.llm.settings") as mock_settings, \
             patch("src.core.llm.app_config") as mock_cfg:
            mock_settings.llm_provider = "anthropic"
            mock_settings.openai_api_key = "fallback-openai-key"
            mock_cfg.embedding_model = "text-embedding-3-small"

            with patch("src.core.llm.OpenAIEmbeddings") as mock_cls:
                mock_cls.return_value = MagicMock()
                result = get_embedding_model()

            mock_cls.assert_called_once()
            assert result is mock_cls.return_value

    def test_anthropic_without_openai_key_raises(self):
        """Anthropic without any OpenAI key → ValueError."""
        with patch("src.core.llm.settings") as mock_settings, \
             patch("src.core.llm.app_config") as mock_cfg:
            mock_settings.llm_provider = "anthropic"
            mock_settings.openai_api_key = None
            mock_cfg.embedding_model = "text-embedding-3-small"

            with pytest.raises(ValueError, match="Embedding model requires"):
                get_embedding_model()

    def test_model_defaults_to_settings_embedding_model(self):
        with patch("src.core.llm.settings") as mock_settings, \
             patch("src.core.llm.app_config") as mock_cfg:
            mock_settings.llm_provider = "openai"
            mock_settings.openai_api_key = "test-key"
            mock_cfg.embedding_model = "text-embedding-ada-002"

            with patch("src.core.llm.OpenAIEmbeddings") as mock_cls:
                mock_cls.return_value = MagicMock()
                get_embedding_model()

            call_kwargs = mock_cls.call_args
            assert call_kwargs.kwargs["model"] == "text-embedding-ada-002"


# Made with Bob
