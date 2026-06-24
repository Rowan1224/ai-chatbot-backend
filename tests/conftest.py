"""Shared pytest fixtures and configuration."""

import os
import pytest
from unittest.mock import MagicMock, AsyncMock

# Set test environment variables
os.environ["MONGODB_URL"] = "mongodb://localhost:27017/test_db"
os.environ["REDIS_URL"] = "redis://localhost:6379"
os.environ["OPENAI_API_KEY"] = "test-key"
os.environ["API_KEY"] = "test-api-key"
os.environ["LLM_PROVIDER"] = "openai"
os.environ["CHAT_MODEL"] = "gpt-4"
os.environ["EMBEDDING_MODEL"] = "text-embedding-3-small"


@pytest.fixture
def mock_mongodb_client():
    """Create a mock MongoDB client."""
    client = MagicMock()
    client.get_collection = MagicMock()
    return client


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
        "employee_id": "EMP12345"
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
    """Sample conversation state."""
    return {
        "messages": [],
        "collected_data": {},
        "is_ready": False,
        "is_complete": False,
        "duplicate_warning": [],
        "config_version": "v1",
        "update_existing": False,
        "existing_request_id": None,
        "temp_pii": {},
        "awaiting_confirmation": False,
        "pii_collected": False,
        "awaiting_duplicate_decision": False,
    }

# Made with Bob
