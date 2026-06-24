"""Unit tests for workflow module."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from src.core.workflow import (
    ConversationWorkflow,
    ConversationState,
    ChatResponse,
    DuplicateDecision,
)


class TestConversationWorkflow:
    """Test ConversationWorkflow class."""
    
    @pytest.fixture
    def mock_clients(self):
        """Create mock database clients."""
        mongodb_client = MagicMock()
        redis_client = MagicMock()
        return mongodb_client, redis_client
    
    @pytest.fixture
    def workflow(self, mock_clients):
        """Create workflow instance with mocked clients."""
        mongodb_client, redis_client = mock_clients
        return ConversationWorkflow(mongodb_client, redis_client)
    
    def test_init(self, workflow):
        """Test workflow initialization."""
        assert workflow.mongodb_client is not None
        assert workflow.redis_client is not None
        assert workflow.llm is not None
        assert workflow.embedding_model is not None
        assert workflow.llm_chat is not None
        assert workflow.llm_extract is not None
        assert workflow.llm_duplicate_decision is not None
    
    def test_get_message_content_with_human_message(self, workflow):
        """Test extracting content from HumanMessage."""
        messages = [
            SystemMessage(content="System prompt"),
            AIMessage(content="Bot response"),
            HumanMessage(content="User input"),
        ]
        
        content = workflow._get_message_content(messages)
        assert content == "User input"
    
    def test_get_message_content_no_human_message(self, workflow):
        """Test when no HumanMessage exists."""
        messages = [
            SystemMessage(content="System prompt"),
            AIMessage(content="Bot response"),
        ]
        
        content = workflow._get_message_content(messages)
        assert content == ""  # Returns empty string, not None
    
    def test_get_message_content_multiple_human_messages(self, workflow):
        """Test that it returns the last HumanMessage."""
        messages = [
            HumanMessage(content="First message"),
            AIMessage(content="Bot response"),
            HumanMessage(content="Second message"),
        ]
        
        content = workflow._get_message_content(messages)
        assert content == "Second message"


class TestPIIParsing:
    """Test PII parsing functionality."""
    
    @pytest.fixture
    def workflow(self):
        """Create workflow instance."""
        mongodb_client = MagicMock()
        redis_client = MagicMock()
        return ConversationWorkflow(mongodb_client, redis_client)
    
    def test_parse_pii_valid_format(self, workflow):
        """Test parsing valid PII format."""
        user_input = "John Doe,EMP12345"
        
        success, name, employee_id, error_msg = workflow._parse_pii(user_input)
        
        assert success is True
        assert name == "John Doe"
        assert employee_id == "EMP12345"
        assert error_msg == ""
    
    def test_parse_pii_with_spaces(self, workflow):
        """Test parsing PII with extra spaces."""
        user_input = "  Jane Smith  ,  EMP67890  "
        
        success, name, employee_id, error_msg = workflow._parse_pii(user_input)
        
        assert success is True
        assert name == "Jane Smith"
        assert employee_id == "EMP67890"
    
    def test_parse_pii_invalid_format_no_comma(self, workflow):
        """Test parsing invalid format without comma."""
        user_input = "John Doe EMP12345"
        
        success, name, employee_id, error_msg = workflow._parse_pii(user_input)
        
        assert success is False
        assert name == ""
        assert employee_id == ""
        assert "exactly 2 values" in error_msg
    
    def test_parse_pii_invalid_format_too_many_parts(self, workflow):
        """Test parsing invalid format with too many commas."""
        user_input = "John,Doe,EMP12345"
        
        success, name, employee_id, error_msg = workflow._parse_pii(user_input)
        
        assert success is False
        assert name == ""
        assert employee_id == ""
        assert "exactly 2 values" in error_msg
    
    def test_parse_pii_empty_parts(self, workflow):
        """Test parsing with empty parts."""
        user_input = ",EMP12345"
        
        success, name, employee_id, error_msg = workflow._parse_pii(user_input)
        
        assert success is False
        assert "at least 2 characters" in error_msg


class TestPIIFormatting:
    """Test PII message formatting."""
    
    @pytest.fixture
    def workflow(self):
        """Create workflow instance."""
        mongodb_client = MagicMock()
        redis_client = MagicMock()
        return ConversationWorkflow(mongodb_client, redis_client)
    
    def test_format_pii_request_message(self, workflow):
        """Test formatting PII request message."""
        message = workflow._format_pii_request_message()
        
        assert "name" in message.lower()
        assert "employee" in message.lower()
        assert "," in message  # Format instruction
    
    def test_format_pii_confirmation_message(self, workflow):
        """Test formatting PII confirmation message."""
        message = workflow._format_pii_confirmation_message("John Doe", "EMP12345")
        
        assert "John Doe" in message
        assert "EMP12345" in message
        assert "yes" in message.lower()


class TestCollectPIINode:
    """Test collect_pii_node functionality."""
    
    @pytest.fixture
    def workflow(self):
        """Create workflow instance."""
        mongodb_client = MagicMock()
        redis_client = MagicMock()
        return ConversationWorkflow(mongodb_client, redis_client)
    
    def test_collect_pii_valid_input(self, workflow):
        """Test collecting PII with valid input."""
        state = {
            "messages": [HumanMessage(content="John Doe,EMP12345")],
            "collected_data": {},
            "temp_pii": {},
            "awaiting_confirmation": False,
            "pii_collected": False,
        }
        
        result = workflow.collect_pii_node(state)  # NOT async
        
        assert result["temp_pii"]["name"] == "John Doe"
        assert result["temp_pii"]["employee_id"] == "EMP12345"
        assert result["awaiting_confirmation"] is True
        assert len(result["messages"]) == 1
    
    def test_collect_pii_invalid_input(self, workflow):
        """Test collecting PII with invalid input."""
        state = {
            "messages": [HumanMessage(content="Invalid format")],
            "collected_data": {},
            "temp_pii": {},
            "awaiting_confirmation": False,
            "pii_collected": False,
        }
        
        result = workflow.collect_pii_node(state)  # NOT async
        
        assert result["awaiting_confirmation"] is False
        assert result["pii_collected"] is False
        # Error message should be present
        assert len(result["messages"]) == 1


class TestConfirmPIINode:
    """Test confirm_pii_node functionality."""
    
    @pytest.fixture
    def workflow(self):
        """Create workflow instance."""
        mongodb_client = MagicMock()
        redis_client = MagicMock()
        return ConversationWorkflow(mongodb_client, redis_client)
    
    def test_confirm_pii_yes(self, workflow):
        """Test confirming PII with 'yes'."""
        state = {
            "messages": [HumanMessage(content="yes")],
            "collected_data": {},
            "temp_pii": {
                "name": "John Doe",
                "employee_id": "EMP12345"
            },
            "awaiting_confirmation": True,
            "pii_collected": False,
        }
        
        result = workflow.confirm_pii_node(state)  # NOT async
        
        assert result["collected_data"]["name"] == "John Doe"
        assert result["collected_data"]["employee_id"] == "EMP12345"
        assert result["pii_collected"] is True
        assert result["awaiting_confirmation"] is False
    
    def test_confirm_pii_corrections(self, workflow):
        """Test providing corrections."""
        state = {
            "messages": [HumanMessage(content="Jane Smith,EMP67890")],
            "collected_data": {},
            "temp_pii": {
                "name": "John Doe",
                "employee_id": "EMP12345"
            },
            "awaiting_confirmation": True,
            "pii_collected": False,
        }
        
        result = workflow.confirm_pii_node(state)  # NOT async
        
        # Should update temp_pii and ask for confirmation again
        assert result["temp_pii"]["name"] == "Jane Smith"
        assert result["temp_pii"]["employee_id"] == "EMP67890"
        assert result["awaiting_confirmation"] is True
        assert result["pii_collected"] is False
    
    def test_confirm_pii_invalid_response(self, workflow):
        """Test invalid confirmation response."""
        state = {
            "messages": [HumanMessage(content="maybe")],
            "collected_data": {},
            "temp_pii": {
                "name": "John Doe",
                "employee_id": "EMP12345"
            },
            "awaiting_confirmation": True,
            "pii_collected": False,
        }
        
        result = workflow.confirm_pii_node(state)  # NOT async
        
        # Should ask for clarification
        assert result["awaiting_confirmation"] is True
        assert result["pii_collected"] is False
        assert "yes" in result["messages"][0].content.lower()


class TestRoutingLogic:
    """Test workflow routing logic."""
    
    @pytest.fixture
    def workflow(self):
        """Create workflow instance."""
        mongodb_client = MagicMock()
        redis_client = MagicMock()
        return ConversationWorkflow(mongodb_client, redis_client)
    
    def test_route_entry_to_chat(self, workflow):
        """Test routing to chat node."""
        state = {
            "is_complete": False,
            "pii_collected": False,
            "awaiting_confirmation": False,
            "awaiting_duplicate_decision": False,
        }
        
        route = workflow.route_entry(state)
        assert route == "chat"
    
    def test_route_entry_to_collect_pii(self, workflow):
        """Test routing to collect_pii node."""
        state = {
            "is_complete": True,
            "pii_collected": False,
            "awaiting_confirmation": False,
            "awaiting_duplicate_decision": False,
        }
        
        route = workflow.route_entry(state)
        assert route == "collect_pii"
    
    def test_route_entry_to_confirm_pii(self, workflow):
        """Test routing to confirm_pii node."""
        state = {
            "is_complete": True,
            "pii_collected": False,
            "awaiting_confirmation": True,
            "awaiting_duplicate_decision": False,
        }
        
        route = workflow.route_entry(state)
        assert route == "confirm_pii"
    
    def test_route_entry_to_handle_duplicate_decision(self, workflow):
        """Test routing to handle_duplicate_decision node."""
        state = {
            "is_complete": True,
            "pii_collected": True,
            "awaiting_confirmation": False,
            "awaiting_duplicate_decision": True,
        }
        
        route = workflow.route_entry(state)
        assert route == "handle_duplicate_decision"
    
    def test_should_extract_ready(self, workflow):
        """Test should_extract when ready."""
        state = {"is_ready": True}
        
        route = workflow.should_extract(state)
        assert route == "extract"
    
    def test_should_extract_not_ready(self, workflow):
        """Test should_extract when not ready."""
        state = {"is_ready": False}
        
        route = workflow.should_extract(state)
        assert route == "__end__"
    
    def test_should_collect_pii_not_collected(self, workflow):
        """Test should_collect_pii when PII not collected."""
        state = {"pii_collected": False}
        
        route = workflow.should_collect_pii(state)
        assert route == "collect_pii"
    
    def test_should_collect_pii_already_collected(self, workflow):
        """Test should_collect_pii when PII already collected."""
        state = {"pii_collected": True}
        
        route = workflow.should_collect_pii(state)
        assert route == "duplicate_check"
    
    def test_should_save_no_duplicates(self, workflow):
        """Test should_save when no duplicates."""
        state = {
            "duplicate_warning": [],
            "awaiting_duplicate_decision": False,
        }
        
        route = workflow.should_save(state)
        assert route == "save"
    
    def test_should_save_duplicates_awaiting_decision(self, workflow):
        """Test should_save when duplicates found and awaiting decision."""
        state = {
            "duplicate_warning": [{"id": "123"}],
            "awaiting_duplicate_decision": True,
        }
        
        route = workflow.should_save(state)
        assert route == "__end__"
    
    def test_should_save_duplicates_decision_made(self, workflow):
        """Test should_save when duplicates found but decision made."""
        state = {
            "duplicate_warning": [{"id": "123"}],
            "awaiting_duplicate_decision": False,
        }
        
        route = workflow.should_save(state)
        assert route == "save"
    
    def test_after_duplicate_decision_valid(self, workflow):
        """Test after_duplicate_decision when decision is valid."""
        state = {"awaiting_duplicate_decision": False}
        
        route = workflow.after_duplicate_decision(state)
        assert route == "save"
    
    def test_after_duplicate_decision_invalid(self, workflow):
        """Test after_duplicate_decision when decision is invalid."""
        state = {"awaiting_duplicate_decision": True}
        
        route = workflow.after_duplicate_decision(state)
        assert route == "__end__"


class TestChatResponse:
    """Test ChatResponse model."""
    
    def test_chat_response_creation(self):
        """Test creating ChatResponse."""
        response = ChatResponse(
            response="Hello, how can I help?",
            is_ready=False
        )
        
        assert response.response == "Hello, how can I help?"
        assert response.is_ready is False
    
    def test_chat_response_validation(self):
        """Test ChatResponse validation."""
        with pytest.raises(Exception):  # Pydantic validation error
            ChatResponse(response="Test")  # Missing is_ready


class TestDuplicateDecision:
    """Test DuplicateDecision model."""
    
    def test_duplicate_decision_creation(self):
        """Test creating DuplicateDecision."""
        decision = DuplicateDecision(
            choice="update",
            reasoning="User wants to update existing request"
        )
        
        assert decision.choice == "update"
        assert decision.reasoning == "User wants to update existing request"
    
    def test_duplicate_decision_validation(self):
        """Test DuplicateDecision validation."""
        with pytest.raises(Exception):  # Pydantic validation error
            DuplicateDecision(choice="update")  # Missing reasoning


class TestBuildGraph:
    """Test graph building."""
    
    @pytest.fixture
    def workflow(self):
        """Create workflow instance."""
        mongodb_client = MagicMock()
        redis_client = MagicMock()
        return ConversationWorkflow(mongodb_client, redis_client)
    
    def test_build_graph_creates_nodes(self, workflow):
        """Test that build_graph creates all required nodes."""
        graph = workflow.build_graph()
        
        # Verify graph is created
        assert graph is not None
        
        # Note: We can't easily test internal node structure without
        # accessing private attributes, but we can verify it doesn't error
    
    def test_compile_creates_app(self, workflow):
        """Test that compile creates the app."""
        workflow.compile()
        
        assert workflow.app is not None

# Made with Bob
