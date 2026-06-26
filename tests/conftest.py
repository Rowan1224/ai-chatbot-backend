"""Shared pytest fixtures and configuration."""

import os
from unittest.mock import AsyncMock, MagicMock

import pytest

# Minimum env vars required by pydantic-settings at import time.
# These are overridden per-test or per-suite where real values matter.
os.environ.setdefault("OPENAI_API_KEY", "test-key")
os.environ.setdefault("API_KEY", "test-api-key")
os.environ.setdefault("LLM_PROVIDER", "openai")
os.environ.setdefault(
    "POSTGRESQL_URL",
    "postgresql://test:test@localhost:5432/test_chatbot",
)
os.environ.setdefault("REDIS_URL", "redis://localhost:6379")


@pytest.fixture
def mock_redis_client():
    """Create a mock Redis client."""
    client = MagicMock()
    return client


@pytest.fixture
def mock_llm():
    """Create a mock LLM."""
    llm = AsyncMock()
    llm.ainvoke = AsyncMock()
    llm.with_structured_output = MagicMock(return_value=llm)
    return llm


@pytest.fixture
def mock_embedding_model():
    """Create a mock embedding model."""
    model = AsyncMock()
    model.aembed_query = AsyncMock(return_value=[0.1] * 1536)
    return model


@pytest.fixture
def sample_request_data():
    """Sample request data for testing."""
    return {
        "request_type": "infrastructure-provisioning",
        "target_environment": "development",
        "business_justification": "Need to test new feature",
        "name": "John Doe",
        "employee_id": "EMP12345",
    }


@pytest.fixture
def sample_request_data_no_pii():
    """Sample request data without PII."""
    return {
        "request_type": "infrastructure-provisioning",
        "target_environment": "development",
        "business_justification": "Need to test new feature",
    }


@pytest.fixture
def sample_embedding():
    """Sample embedding vector."""
    return [0.1] * 1536


@pytest.fixture
def sample_conversation_state():
    """Sample conversation state matching ConversationState TypedDict."""
    return {
        "messages": [],
        "collected_data": {},
        "is_ready": False,
        "is_complete": False,
        "duplicate_warning": [],
        "config_version": "v2.0",
        "awaiting_duplicate_decision": False,
        "duplicate_decision": None,
    }


# Made with Bob
