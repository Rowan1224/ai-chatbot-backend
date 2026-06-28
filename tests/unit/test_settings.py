"""Unit tests for settings.py — Settings validators, PromptConfig, and AppConfig."""

from unittest.mock import mock_open, patch

import pytest
import yaml


class TestSettingsValidation:
    """Tests for Settings.validate_llm_config model validator."""

    def test_openai_requires_api_key(self):
        """OPENAI_API_KEY must be set when LLM_PROVIDER=openai."""
        from pydantic import ValidationError

        with pytest.raises((ValidationError, Exception)):
            with patch.dict(
                "os.environ",
                {
                    "LLM_PROVIDER": "openai",
                    "OPENAI_API_KEY": "",
                    "POSTGRESQL_URL": "postgresql://localhost/test",
                    "API_KEY": "test-key",
                },
                clear=False,
            ):
                import importlib

                import src.config.settings as settings_mod
                importlib.reload(settings_mod)

    def test_azure_requires_api_key_and_endpoint(self):
        """Both AZURE_OPENAI_API_KEY and AZURE_OPENAI_ENDPOINT are required for azure."""
        from pydantic import ValidationError

        with pytest.raises((ValidationError, Exception)):
            with patch.dict(
                "os.environ",
                {
                    "LLM_PROVIDER": "azure",
                    "AZURE_OPENAI_API_KEY": "",
                    "AZURE_OPENAI_ENDPOINT": "",
                    "POSTGRESQL_URL": "postgresql://localhost/test",
                    "OPENAI_API_KEY": "",
                    "API_KEY": "test-key",
                },
                clear=False,
            ):
                import importlib

                import src.config.settings as settings_mod
                importlib.reload(settings_mod)

    def test_anthropic_requires_api_key(self):
        with pytest.raises((Exception,)):
            with patch.dict(
                "os.environ",
                {
                    "LLM_PROVIDER": "anthropic",
                    "ANTHROPIC_API_KEY": "",
                    "POSTGRESQL_URL": "postgresql://localhost/test",
                    "OPENAI_API_KEY": "",
                    "API_KEY": "test-key",
                },
                clear=False,
            ):
                import importlib

                import src.config.settings as settings_mod
                importlib.reload(settings_mod)

    def test_invalid_provider_raises(self):
        with pytest.raises((Exception,)):
            with patch.dict(
                "os.environ",
                {
                    "LLM_PROVIDER": "cohere",
                    "POSTGRESQL_URL": "postgresql://localhost/test",
                    "OPENAI_API_KEY": "x",
                    "API_KEY": "test-key",
                },
                clear=False,
            ):
                import importlib

                import src.config.settings as settings_mod
                importlib.reload(settings_mod)


class TestSettingsValidator:
    """Test Settings validator logic directly without full reload."""

    def test_openai_missing_key_raises_value_error(self):
        """Validator raises when provider=openai and key is None."""
        from pydantic import ValidationError

        from src.config.settings import Settings

        with pytest.raises((ValidationError, ValueError)):
            Settings(
                llm_provider="openai",
                openai_api_key=None,
                postgresql_url="postgresql://localhost/test",
                api_key="test-key",
            )

    def test_azure_missing_key_raises_value_error(self):
        from pydantic import ValidationError

        from src.config.settings import Settings

        with pytest.raises((ValidationError, ValueError)):
            Settings(
                llm_provider="azure",
                azure_openai_api_key=None,
                azure_openai_endpoint="https://my.openai.azure.com/",
                postgresql_url="postgresql://localhost/test",
                api_key="test-key",
            )

    def test_azure_missing_endpoint_raises_value_error(self):
        from pydantic import ValidationError

        from src.config.settings import Settings

        with pytest.raises((ValidationError, ValueError)):
            Settings(
                llm_provider="azure",
                azure_openai_api_key="some-key",
                azure_openai_endpoint=None,
                postgresql_url="postgresql://localhost/test",
                api_key="test-key",
            )

    def test_anthropic_missing_key_raises_value_error(self):
        from pydantic import ValidationError

        from src.config.settings import Settings

        with pytest.raises((ValidationError, ValueError)):
            Settings(
                llm_provider="anthropic",
                anthropic_api_key=None,
                postgresql_url="postgresql://localhost/test",
                api_key="test-key",
            )

    def test_invalid_provider_raises_value_error(self):
        from pydantic import ValidationError

        from src.config.settings import Settings

        with pytest.raises((ValidationError, ValueError)):
            Settings(
                llm_provider="cohere",
                openai_api_key="x",
                postgresql_url="postgresql://localhost/test",
                api_key="test-key",
            )


class TestPromptConfig:
    """Tests for PromptConfig — system prompt and version only."""

    def _make_config(self, data: dict):
        """Return a PromptConfig loaded from a fake YAML file."""
        from src.config.settings import PromptConfig

        yaml_str = yaml.dump(data)
        with patch("builtins.open", mock_open(read_data=yaml_str)):
            return PromptConfig("fake_path.yaml")

    def test_config_version_from_yaml(self):
        cfg = self._make_config({"config_version": "v3.0"})
        assert cfg.config_version == "v3.0"

    def test_config_version_default(self):
        """Defaults to 'v1.0' when key is missing."""
        cfg = self._make_config({})
        assert cfg.config_version == "v1.0"

    def test_system_prompt_from_yaml(self):
        cfg = self._make_config({"system_prompt": "You are a helpful bot."})
        assert cfg.system_prompt == "You are a helpful bot."

    def test_system_prompt_default_empty(self):
        cfg = self._make_config({})
        assert cfg.system_prompt == ""


class TestAppConfig:
    """Tests for AppConfig — models, duplicate detection, privacy, logging."""

    def _make_config(self, data: dict):
        """Return an AppConfig loaded from a fake YAML file."""
        from src.config.settings import AppConfig

        yaml_str = yaml.dump(data)
        with patch("builtins.open", mock_open(read_data=yaml_str)):
            return AppConfig("fake_path.yaml")

    # --- LLM models ---

    def test_chat_model_from_yaml(self):
        cfg = self._make_config({"llm": {"chat_model": "gpt-4o"}})
        assert cfg.chat_model == "gpt-4o"

    def test_chat_model_default(self):
        cfg = self._make_config({})
        assert cfg.chat_model == "gpt-4"

    def test_embedding_model_from_yaml(self):
        cfg = self._make_config({"llm": {"embedding_model": "text-embedding-ada-002"}})
        assert cfg.embedding_model == "text-embedding-ada-002"

    def test_embedding_model_default(self):
        cfg = self._make_config({})
        assert cfg.embedding_model == "text-embedding-3-small"

    # --- Duplicate detection ---

    def test_duplicate_detection_enabled_from_yaml(self):
        cfg = self._make_config({"duplicate_detection": {"enabled": False}})
        assert cfg.duplicate_detection_enabled is False

    def test_duplicate_detection_enabled_default(self):
        cfg = self._make_config({})
        assert cfg.duplicate_detection_enabled is True

    def test_similarity_threshold_from_yaml(self):
        cfg = self._make_config({"duplicate_detection": {"similarity_threshold": 0.9}})
        assert cfg.similarity_threshold == 0.9

    def test_similarity_threshold_default(self):
        cfg = self._make_config({})
        assert cfg.similarity_threshold == 0.85

    def test_lookback_days_from_yaml(self):
        cfg = self._make_config({"duplicate_detection": {"lookback_days": 30}})
        assert cfg.lookback_days == 30

    def test_lookback_days_default(self):
        cfg = self._make_config({})
        assert cfg.lookback_days == 90

    def test_fuzzy_threshold_from_yaml(self):
        cfg = self._make_config({"duplicate_detection": {"fuzzy_threshold": 0.5}})
        assert cfg.fuzzy_threshold == 0.5

    def test_fuzzy_threshold_default(self):
        cfg = self._make_config({})
        assert cfg.fuzzy_threshold == 0.3

    def test_candidate_limit_from_yaml(self):
        cfg = self._make_config({"duplicate_detection": {"candidate_limit": 100}})
        assert cfg.candidate_limit == 100

    def test_candidate_limit_default(self):
        cfg = self._make_config({})
        assert cfg.candidate_limit == 50

    def test_semantic_search_enabled_from_yaml(self):
        cfg = self._make_config({"duplicate_detection": {"semantic_search_enabled": False}})
        assert cfg.semantic_search_enabled is False

    def test_vector_search_provider_from_yaml(self):
        cfg = self._make_config({"duplicate_detection": {"vector_search_provider": "local"}})
        assert cfg.vector_search_provider == "local"

    def test_vector_search_provider_default(self):
        cfg = self._make_config({})
        assert cfg.vector_search_provider == "pgvector"

    # --- Privacy ---

    def test_pii_fields_from_yaml(self):
        cfg = self._make_config({"privacy": {"pii_fields": ["name", "employee_id"]}})
        assert cfg.pii_fields == ["name", "employee_id"]

    def test_pii_fields_default_empty(self):
        cfg = self._make_config({})
        assert cfg.pii_fields == []

    # --- Logging ---

    def test_log_level_from_yaml(self):
        cfg = self._make_config({"logging": {"level": "DEBUG"}})
        assert cfg.log_level == "DEBUG"

    def test_log_level_default(self):
        cfg = self._make_config({})
        assert cfg.log_level == "INFO"

    # --- CORS ---

    def test_cors_origins_from_yaml(self):
        cfg = self._make_config({"cors": {"origins": ["https://app.example.com"]}})
        assert cfg.cors_origins == ["https://app.example.com"]

    def test_cors_origins_default_empty(self):
        cfg = self._make_config({})
        assert cfg.cors_origins == []

    # --- Guardrails ---

    def test_guardrails_enabled_from_yaml(self):
        cfg = self._make_config({"guardrails": {"enabled": False}})
        assert cfg.guardrails_enabled is False

    def test_guardrails_enabled_default_true(self):
        cfg = self._make_config({})
        assert cfg.guardrails_enabled is True

    def test_pii_redaction_enabled_from_yaml(self):
        cfg = self._make_config({"guardrails": {"pii_redaction_enabled": False}})
        assert cfg.pii_redaction_enabled is False

    def test_pii_redaction_enabled_default_true(self):
        cfg = self._make_config({})
        assert cfg.pii_redaction_enabled is True

    def test_injection_detection_enabled_from_yaml(self):
        cfg = self._make_config(
            {"guardrails": {"injection_detection_enabled": False}}
        )
        assert cfg.injection_detection_enabled is False

    def test_injection_detection_enabled_default_true(self):
        cfg = self._make_config({})
        assert cfg.injection_detection_enabled is True

    def test_all_guardrail_flags_can_be_set_together(self):
        cfg = self._make_config({
            "guardrails": {
                "enabled": True,
                "pii_redaction_enabled": False,
                "injection_detection_enabled": False,
            }
        })
        assert cfg.guardrails_enabled is True
        assert cfg.pii_redaction_enabled is False
        assert cfg.injection_detection_enabled is False



