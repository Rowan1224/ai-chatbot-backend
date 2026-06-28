"""Unit tests for simplified workflow module."""

from unittest.mock import MagicMock

import pytest
from langgraph.checkpoint.memory import InMemorySaver

from src.core.workflow import (
    ChatResponse,
    ConversationWorkflow,
    DuplicateDecision,
    DuplicateJudgement,
)


class TestConversationWorkflow:
    """Test ConversationWorkflow class."""

    @pytest.fixture
    def workflow(self):
        """Create workflow instance with mocked clients."""
        pg_client = MagicMock()
        return ConversationWorkflow(pg_client)

    def test_init(self, workflow):
        """Test workflow initialization."""
        assert workflow.pg_client is not None
        assert workflow.llm is not None
        assert workflow.embedding_model is not None
        assert workflow.llm_chat is not None
        assert workflow.llm_extract is not None
        assert workflow.llm_duplicate_judge is not None
        assert workflow.llm_duplicate_decision is not None


class TestRoutingLogic:
    """Test workflow routing logic."""

    @pytest.fixture
    def workflow(self):
        """Create workflow instance."""
        mongodb_client = MagicMock()
        return ConversationWorkflow(mongodb_client)

    def test_route_after_guardrail_to_chat(self, workflow):
        """Test routing to chat when no injection and no duplicate pending."""
        state = {
            "awaiting_duplicate_decision": False,
            "injection_blocked": False,
        }

        route = workflow.route_after_guardrail(state)
        assert route == "chat"

    def test_route_after_guardrail_blocked(self, workflow):
        """Test routing to END when injection was blocked."""
        state = {
            "injection_blocked": True,
        }

        route = workflow.route_after_guardrail(state)
        assert route == "__end__"

    def test_route_after_guardrail_to_duplicate_decision(self, workflow):
        """Test routing to handle_duplicate_decision when awaiting."""
        state = {
            "awaiting_duplicate_decision": True,
            "injection_blocked": False,
        }

        route = workflow.route_after_guardrail(state)
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
            "collected_data": {"request_type": "infra"},
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
            "collected_data": {"request_type": "infra"},
            "duplicate_warning": [{"id": "123"}],
            "awaiting_duplicate_decision": False,
        }

        route = workflow.should_save(state)
        assert route == "save"

    def test_after_duplicate_decision_modify(self, workflow):
        """Test after_duplicate_decision routes to chat on modify."""
        state = {"duplicate_decision": "modify"}
        assert workflow.after_duplicate_decision(state) == "chat"

    def test_after_duplicate_decision_proceed(self, workflow):
        """Test after_duplicate_decision routes to save on proceed."""
        state = {"duplicate_decision": "proceed"}
        assert workflow.after_duplicate_decision(state) == "save"

    def test_after_duplicate_decision_cancel(self, workflow):
        """Test after_duplicate_decision ends conversation on cancel."""
        state = {"duplicate_decision": "cancel"}
        assert workflow.after_duplicate_decision(state) == "__end__"

    def test_after_duplicate_decision_unrecognised(self, workflow):
        """Test after_duplicate_decision ends turn on unrecognised reply."""
        state = {"duplicate_decision": None}
        assert workflow.after_duplicate_decision(state) == "__end__"


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
        with pytest.raises(ValueError):  # Pydantic validation error
            ChatResponse(response="Test")  # Missing is_ready


class TestDuplicateJudgement:
    """Test DuplicateJudgement model."""

    def test_is_duplicate_true(self):
        """Test judgement when requests are duplicates."""
        judgement = DuplicateJudgement(
            is_duplicate=True,
            reasoning="Both requests are for infrastructure provisioning in production."
        )
        assert judgement.is_duplicate is True
        assert judgement.reasoning != ""

    def test_is_duplicate_false(self):
        """Test judgement when requests differ meaningfully."""
        judgement = DuplicateJudgement(
            is_duplicate=False,
            reasoning="New request targets production; existing targets development."
        )
        assert judgement.is_duplicate is False

    def test_duplicate_judgement_validation(self):
        """Test DuplicateJudgement requires both fields."""
        with pytest.raises(ValueError):
            DuplicateJudgement(is_duplicate=True)  # Missing reasoning


class TestDuplicateDecision:
    """Test DuplicateDecision model."""

    def test_duplicate_decision_modify(self):
        """Test creating DuplicateDecision with modify choice."""
        decision = DuplicateDecision(
            choice="modify",
            reasoning="User wants to change their request"
        )
        assert decision.choice == "modify"
        assert decision.reasoning == "User wants to change their request"

    def test_duplicate_decision_proceed(self):
        """Test creating DuplicateDecision with proceed choice."""
        decision = DuplicateDecision(
            choice="proceed",
            reasoning="Request is intentionally different"
        )
        assert decision.choice == "proceed"

    def test_duplicate_decision_cancel(self):
        """Test creating DuplicateDecision with cancel choice."""
        decision = DuplicateDecision(
            choice="cancel",
            reasoning="User wants to abandon the request"
        )
        assert decision.choice == "cancel"

    def test_duplicate_decision_validation(self):
        """Test DuplicateDecision validation."""
        with pytest.raises(ValueError):  # Pydantic validation error
            DuplicateDecision(choice="modify")  # Missing reasoning


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



