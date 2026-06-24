"""Simplified unit tests that actually work with the current implementation."""

import pytest
from unittest.mock import MagicMock, AsyncMock, patch
from langchain_core.messages import AIMessage, HumanMessage

from src.core.workflow import ConversationWorkflow
from src.core.schema import RequestSchema, get_text_fields, schema_to_dict


class TestSchemaBasics:
    """Basic schema tests that work."""
    
    def test_get_text_fields_excludes_pii(self):
        """Test that PII fields are excluded."""
        text_fields = get_text_fields()
        
        # Should NOT include PII
        assert "name" not in text_fields
        assert "employee_id" not in text_fields
        
        # Should include business fields
        assert len(text_fields) > 0
    
    def test_schema_to_dict_works(self):
        """Test schema conversion."""
        data = {
            "request_type": "infrastructure-provisioning",
            "target_environment": "development",
            "business_justification": "Test",
            "name": "John Doe",
            "employee_id": "EMP123"
        }
        
        instance = RequestSchema(**data)
        result = schema_to_dict(instance)
        
        assert isinstance(result, dict)
        assert "request_type" in result
        assert "name" in result


class TestWorkflowHelpers:
    """Test workflow helper methods."""
    
    @pytest.fixture
    def workflow(self):
        """Create workflow with mocked clients."""
        mongodb = MagicMock()
        redis = MagicMock()
        return ConversationWorkflow(mongodb, redis)
    
    def test_get_message_content_finds_human_message(self, workflow):
        """Test extracting content from HumanMessage."""
        messages = [
            AIMessage(content="Bot message"),
            HumanMessage(content="User message"),
        ]
        
        content = workflow._get_message_content(messages)
        assert content == "User message"
    
    def test_get_message_content_empty_list(self, workflow):
        """Test with empty message list."""
        content = workflow._get_message_content([])
        assert content == ""
    
    def test_parse_pii_valid(self, workflow):
        """Test parsing valid PII."""
        success, name, emp_id, error = workflow._parse_pii("John Doe,EMP12345")
        
        assert success is True
        assert name == "John Doe"
        assert emp_id == "EMP12345"
        assert error == ""
    
    def test_parse_pii_invalid_format(self, workflow):
        """Test parsing invalid PII format."""
        success, name, emp_id, error = workflow._parse_pii("Invalid")
        
        assert success is False
        assert name == ""
        assert emp_id == ""
        assert error != ""
    
    def test_parse_pii_missing_emp_prefix(self, workflow):
        """Test PII without EMP prefix."""
        success, name, emp_id, error = workflow._parse_pii("John Doe,12345")
        
        assert success is False
    
    def test_format_pii_request_message(self, workflow):
        """Test PII request message formatting."""
        message = workflow._format_pii_request_message()
        
        assert "format" in message.lower()
        assert "," in message
    
    def test_format_pii_confirmation_message(self, workflow):
        """Test PII confirmation message formatting."""
        message = workflow._format_pii_confirmation_message("John Doe", "EMP123")
        
        assert "John Doe" in message
        assert "EMP123" in message
        assert "confirm" in message.lower()


class TestWorkflowRouting:
    """Test workflow routing logic."""
    
    @pytest.fixture
    def workflow(self):
        """Create workflow."""
        mongodb = MagicMock()
        redis = MagicMock()
        return ConversationWorkflow(mongodb, redis)
    
    def test_route_entry_to_chat(self, workflow):
        """Test routing to chat."""
        state = {
            "is_complete": False,
            "pii_collected": False,
            "awaiting_confirmation": False,
            "awaiting_duplicate_decision": False,
        }
        
        route = workflow.route_entry(state)
        assert route == "chat"
    
    def test_route_entry_to_collect_pii(self, workflow):
        """Test routing to collect_pii."""
        state = {
            "is_complete": True,
            "pii_collected": False,
            "awaiting_confirmation": False,
            "awaiting_duplicate_decision": False,
        }
        
        route = workflow.route_entry(state)
        assert route == "collect_pii"
    
    def test_route_entry_to_confirm_pii(self, workflow):
        """Test routing to confirm_pii."""
        state = {
            "is_complete": True,
            "pii_collected": False,
            "awaiting_confirmation": True,
            "awaiting_duplicate_decision": False,
        }
        
        route = workflow.route_entry(state)
        assert route == "confirm_pii"
    
    def test_route_entry_to_duplicate_decision(self, workflow):
        """Test routing to handle_duplicate_decision."""
        state = {
            "is_complete": True,
            "pii_collected": True,
            "awaiting_confirmation": False,
            "awaiting_duplicate_decision": True,
        }
        
        route = workflow.route_entry(state)
        assert route == "handle_duplicate_decision"
    
    def test_should_extract_when_ready(self, workflow):
        """Test should_extract routing."""
        state = {"is_ready": True}
        assert workflow.should_extract(state) == "extract"
        
        state = {"is_ready": False}
        assert workflow.should_extract(state) == "__end__"
    
    def test_should_collect_pii_routing(self, workflow):
        """Test should_collect_pii routing."""
        state = {"pii_collected": False}
        assert workflow.should_collect_pii(state) == "collect_pii"
        
        state = {"pii_collected": True}
        assert workflow.should_collect_pii(state) == "duplicate_check"
    
    def test_should_save_no_duplicates(self, workflow):
        """Test should_save with no duplicates."""
        state = {
            "duplicate_warning": [],
            "awaiting_duplicate_decision": False,
        }
        assert workflow.should_save(state) == "save"
    
    def test_should_save_with_duplicates_awaiting(self, workflow):
        """Test should_save with duplicates awaiting decision."""
        state = {
            "duplicate_warning": [{"id": "123"}],
            "awaiting_duplicate_decision": True,
        }
        assert workflow.should_save(state) == "__end__"
    
    def test_after_duplicate_decision_valid(self, workflow):
        """Test after_duplicate_decision routing."""
        state = {"awaiting_duplicate_decision": False}
        assert workflow.after_duplicate_decision(state) == "save"
        
        state = {"awaiting_duplicate_decision": True}
        assert workflow.after_duplicate_decision(state) == "__end__"


class TestWorkflowInitialization:
    """Test workflow initialization."""
    
    def test_workflow_init(self):
        """Test that workflow initializes correctly."""
        mongodb = MagicMock()
        redis = MagicMock()
        
        workflow = ConversationWorkflow(mongodb, redis)
        
        assert workflow.mongodb_client is not None
        assert workflow.redis_client is not None
        assert workflow.llm is not None
        assert workflow.embedding_model is not None
        assert workflow.llm_chat is not None
        assert workflow.llm_extract is not None
        assert workflow.llm_duplicate_decision is not None
    
    def test_workflow_build_graph(self):
        """Test that graph builds without errors."""
        mongodb = MagicMock()
        redis = MagicMock()
        
        workflow = ConversationWorkflow(mongodb, redis)
        graph = workflow.build_graph()
        
        assert graph is not None
    
    def test_workflow_compile(self):
        """Test that workflow compiles."""
        mongodb = MagicMock()
        redis = MagicMock()
        
        workflow = ConversationWorkflow(mongodb, redis)
        workflow.compile()
        
        assert workflow.app is not None


@pytest.mark.asyncio
class TestDatabaseClients:
    """Test database client initialization."""
    
    async def test_mongodb_client_init(self):
        """Test MongoDB client can be created."""
        from src.core.database import MongoDBClient
        
        client = MongoDBClient()
        assert client is not None
        assert client.client is None  # Not connected yet
        assert client.db is None
    
    def test_redis_client_init(self):
        """Test Redis client can be created."""
        from src.core.database import RedisClient
        
        client = RedisClient()
        assert client is not None
        assert client.client is None  # Not connected yet


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

# Made with Bob
