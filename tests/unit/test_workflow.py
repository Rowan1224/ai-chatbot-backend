"""Unit tests for simplified workflow module."""

import pytest
from unittest.mock import MagicMock

from langgraph.checkpoint.memory import InMemorySaver

from src.core.workflow import (
    ConversationWorkflow,
    ConversationState,
    ChatResponse,
    DuplicateDecision,
)


class TestConversationWorkflow:
    """Test ConversationWorkflow class."""
    
    @pytest.fixture
    def workflow(self):
        """Create workflow instance with mocked clients."""
        mongodb_client = MagicMock()
        return ConversationWorkflow(mongodb_client)
    
    def test_init(self, workflow):
        """Test workflow initialization."""
        assert workflow.mongodb_client is not None
        assert workflow.llm is not None
        assert workflow.embedding_model is not None
        assert workflow.llm_chat is not None
        assert workflow.llm_extract is not None
        assert workflow.llm_duplicate_decision is not None


class TestRoutingLogic:
    """Test workflow routing logic."""
    
    @pytest.fixture
    def workflow(self):
        """Create workflow instance."""
        mongodb_client = MagicMock()
        return ConversationWorkflow(mongodb_client)
    
    def test_route_entry_to_chat(self, workflow):
        """Test routing to chat node."""
        state = {
            "awaiting_duplicate_decision": False,
        }
        
        route = workflow.route_entry(state)
        assert route == "chat"
    
    def test_route_entry_to_handle_duplicate_decision(self, workflow):
        """Test routing to handle_duplicate_decision node."""
        state = {
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
        return ConversationWorkflow(mongodb_client)
    
    def test_build_graph_creates_nodes(self, workflow):
        """Test that build_graph creates all required nodes."""
        graph = workflow.build_graph()
        
        # Verify graph is created
        assert graph is not None
    
    def test_compile_creates_app(self, workflow):
        """Test that compile creates the app."""
        checkpointer = InMemorySaver()
        workflow.compile(checkpointer)
        
        assert workflow.app is not None


# Made with Bob