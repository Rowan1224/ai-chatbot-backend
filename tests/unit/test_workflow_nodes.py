"""
Unit tests for ConversationWorkflow async node methods.

All LLM / database / embedding calls are mocked so tests run
without any real external services.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_core.messages import (
    HumanMessage,
    SystemMessage,
)

from src.core.models import (
    ChatResponse,
    DuplicateDecision,
    DuplicateJudgement,
)
from src.core.workflow import (
    ConversationWorkflow,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def workflow():
    """
    ConversationWorkflow with all LLM/embedding attrs mocked.
    Patches get_llm and get_embedding_model at construction time.
    """
    mock_llm = MagicMock()
    mock_embedding = AsyncMock()
    mock_embedding.aembed_query = AsyncMock(return_value=[0.1] * 1536)

    with (
        patch("src.core.workflow.get_llm", return_value=mock_llm),
        patch("src.core.workflow.get_embedding_model", return_value=mock_embedding),
    ):
        pg = MagicMock()
        wf = ConversationWorkflow(pg_client=pg)

    # Replace structured LLMs with async-capable mocks
    wf.llm_chat = AsyncMock()
    wf.llm_extract = AsyncMock()
    wf.llm_duplicate_judge = AsyncMock()
    wf.llm_duplicate_decision = AsyncMock()
    wf.embedding_model = mock_embedding
    return wf


def _make_state(**overrides) -> dict:
    """Build a minimal ConversationState dict."""
    base = {
        "messages": [HumanMessage(content="Hello")],
        "collected_data": {},
        "is_ready": False,
        "is_complete": False,
        "duplicate_warning": [],
        "config_version": "v1",
        "awaiting_duplicate_decision": False,
        "duplicate_decision": None,
    }
    base.update(overrides)
    return base


# ---------------------------------------------------------------------------
# chat_node
# ---------------------------------------------------------------------------


class TestChatNode:
    """Tests for ConversationWorkflow.chat_node."""

    @pytest.mark.asyncio
    async def test_chat_node_returns_ai_message(self, workflow):
        """chat_node returns an AIMessage with the LLM response."""
        workflow.llm_chat.ainvoke = AsyncMock(
            return_value=ChatResponse(response="Hello user!", is_ready=False)
        )

        state = _make_state(messages=[HumanMessage(content="Hi")])
        result = await workflow.chat_node(state)

        assert "messages" in result
        assert result["messages"][0].content == "Hello user!"
        assert result["is_ready"] is False

    @pytest.mark.asyncio
    async def test_chat_node_sets_is_ready_true(self, workflow):
        """When LLM signals ready, is_ready=True propagates."""
        workflow.llm_chat.ainvoke = AsyncMock(
            return_value=ChatResponse(response="All done!", is_ready=True)
        )

        state = _make_state(messages=[HumanMessage(content="My name is Jane")])
        result = await workflow.chat_node(state)

        assert result["is_ready"] is True

    @pytest.mark.asyncio
    async def test_chat_node_injects_system_message_if_absent(self, workflow):
        """chat_node prepends SystemMessage when none is present."""
        captured_msgs = []

        async def capture(msgs):
            captured_msgs.extend(msgs)
            return ChatResponse(response="ok", is_ready=False)

        workflow.llm_chat.ainvoke = capture

        state = _make_state(messages=[HumanMessage(content="start")])
        await workflow.chat_node(state)

        assert any(isinstance(m, SystemMessage) for m in captured_msgs)

    @pytest.mark.asyncio
    async def test_chat_node_does_not_duplicate_system_message(self, workflow):
        """chat_node does NOT prepend a second SystemMessage."""
        captured_msgs = []

        async def capture(msgs):
            captured_msgs.extend(msgs)
            return ChatResponse(response="ok", is_ready=False)

        workflow.llm_chat.ainvoke = capture

        state = _make_state(
            messages=[
                SystemMessage(content="existing system"),
                HumanMessage(content="hi"),
            ]
        )
        await workflow.chat_node(state)

        system_count = sum(
            1 for m in captured_msgs if isinstance(m, SystemMessage)
        )
        assert system_count == 1


# ---------------------------------------------------------------------------
# extract_node
# ---------------------------------------------------------------------------


class TestExtractNode:
    """Tests for ConversationWorkflow.extract_node."""

    @pytest.mark.asyncio
    async def test_extract_node_success(self, workflow):
        """Successful extraction returns collected_data and is_complete=True."""
        from src.core.schema import DataField, ExtractedRequest

        workflow.llm_extract.ainvoke = AsyncMock(
            return_value=ExtractedRequest(
                request_type="infrastructure-provisioning",
                name="Jane",
                employee_id="EMP001",
                additional_data=[
                    DataField(key="target_environment", value="production"),
                ],
            )
        )

        state = _make_state(messages=[HumanMessage(content="deploy please")])
        result = await workflow.extract_node(state)

        assert result["is_complete"] is True
        assert result["collected_data"]["request_type"] == "infrastructure-provisioning"
        assert result["collected_data"]["name"] == "Jane"
        assert result["collected_data"]["target_environment"] == "production"

    @pytest.mark.asyncio
    async def test_extract_node_failure_returns_error_message(self, workflow):
        """When LLM extraction throws, node returns an error AIMessage."""
        workflow.llm_extract.ainvoke = AsyncMock(
            side_effect=RuntimeError("LLM timeout")
        )

        state = _make_state(messages=[HumanMessage(content="deploy please")])
        result = await workflow.extract_node(state)

        assert result["is_complete"] is False
        assert result["is_ready"] is True
        assert "trouble" in result["messages"][0].content.lower()


# ---------------------------------------------------------------------------
# save_node
# ---------------------------------------------------------------------------


class TestSaveNode:
    """Tests for ConversationWorkflow.save_node."""

    @pytest.mark.asyncio
    async def test_save_node_success(self, workflow):
        """save_node returns a success AIMessage containing the request ID."""
        workflow.embedding_model.aembed_query = AsyncMock(
            return_value=[0.1] * 1536
        )

        with patch("src.core.workflow.save_request", new_callable=AsyncMock) as mock_save:
            mock_save.return_value = "uuid-1234"

            state = _make_state(
                collected_data={
                    "request_type": "infrastructure-provisioning",
                    "target_environment": "production",
                    "business_justification": "Need servers",
                    "name": "Jane",
                    "employee_id": "EMP001",
                }
            )
            result = await workflow.save_node(state)

        assert result["is_complete"] is True
        assert "uuid-1234" in result["messages"][0].content
        mock_save.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_save_node_missing_collected_data_returns_error(self, workflow):
        """save_node returns error when collected_data is absent."""
        state = _make_state(collected_data={})

        result = await workflow.save_node(state)

        assert result["is_complete"] is False
        assert result["is_ready"] is True
        assert "trouble" in result["messages"][0].content.lower()

    @pytest.mark.asyncio
    async def test_save_node_calls_embedding(self, workflow):
        """save_node calls aembed_query with non-PII text."""
        embed_mock = AsyncMock(return_value=[0.2] * 1536)
        workflow.embedding_model.aembed_query = embed_mock

        with patch("src.core.workflow.save_request", new_callable=AsyncMock) as mock_save:
            mock_save.return_value = "abc"

            state = _make_state(
                collected_data={
                    "request_type": "access-grant",
                    "business_justification": "Need access",
                    "name": "Bob",
                    "employee_id": "EMP99",
                }
            )
            await workflow.save_node(state)

        embed_mock.assert_awaited_once()
        # Ensure PII fields are NOT in the text passed to embed
        call_text = embed_mock.call_args[0][0]
        assert "Bob" not in call_text
        assert "EMP99" not in call_text


# ---------------------------------------------------------------------------
# duplicate_check_node
# ---------------------------------------------------------------------------


class TestDuplicateCheckNode:
    """Tests for ConversationWorkflow.duplicate_check_node."""

    @pytest.mark.asyncio
    async def test_returns_empty_when_detection_disabled(self, workflow):
        with patch("src.core.workflow.app_config") as mock_cfg:
            mock_cfg.duplicate_detection_enabled = False

            state = _make_state(
                collected_data={"request_type": "infra", "name": "X", "employee_id": "EMP1"}
            )
            result = await workflow.duplicate_check_node(state)

        assert result == {}

    @pytest.mark.asyncio
    async def test_returns_empty_when_no_collected_data(self, workflow):
        with patch("src.core.workflow.app_config") as mock_cfg:
            mock_cfg.duplicate_detection_enabled = True

            state = _make_state(collected_data={})
            result = await workflow.duplicate_check_node(state)

        assert result == {}

    @pytest.mark.asyncio
    async def test_returns_empty_when_no_fuzzy_candidates(self, workflow):
        with (
            patch("src.core.workflow.app_config") as mock_cfg,
            patch(
                "src.core.workflow.find_fuzzy_candidates",
                new_callable=AsyncMock,
                return_value=[],
            ),
        ):
            mock_cfg.duplicate_detection_enabled = True
            mock_cfg.lookback_days = 90
            mock_cfg.fuzzy_threshold = 0.3
            mock_cfg.candidate_limit = 50
            mock_cfg.semantic_search_enabled = True
            mock_cfg.similarity_threshold = 0.85
            mock_cfg.pii_fields = []

            state = _make_state(
                collected_data={
                    "request_type": "infrastructure-provisioning",
                    "business_justification": "Need servers",
                    "name": "Jane",
                    "employee_id": "EMP001",
                }
            )
            result = await workflow.duplicate_check_node(state)

        assert result == {}

    @pytest.mark.asyncio
    async def test_returns_duplicate_warning_when_confirmed(self, workflow):
        """When LLM judge confirms a duplicate, state gets duplicate_warning."""
        import uuid
        from datetime import datetime

        candidate_id = str(uuid.uuid4())
        similar_row = {
            "_id": candidate_id,
            "similarity_score": 0.95,
            "created_at": datetime.utcnow(),
            "config_version": "v1",
            "data": {
                "request_type": "infrastructure-provisioning",
                "target_environment": "production",
                "business_justification": "Same justification",
            },
        }

        verdict = DuplicateJudgement(is_duplicate=True, reasoning="Same request")

        with (
            patch("src.core.workflow.app_config") as mock_cfg,
            patch(
                "src.core.workflow.find_fuzzy_candidates",
                new_callable=AsyncMock,
                return_value=[candidate_id],
            ),
            patch(
                "src.core.workflow.find_similar_requests",
                new_callable=AsyncMock,
                return_value=[similar_row],
            ),
        ):
            mock_cfg.duplicate_detection_enabled = True
            mock_cfg.lookback_days = 90
            mock_cfg.fuzzy_threshold = 0.3
            mock_cfg.candidate_limit = 50
            mock_cfg.semantic_search_enabled = True
            mock_cfg.similarity_threshold = 0.85
            mock_cfg.pii_fields = []

            workflow.llm_duplicate_judge.ainvoke = AsyncMock(return_value=verdict)
            workflow.embedding_model.aembed_query = AsyncMock(
                return_value=[0.1] * 1536
            )

            state = _make_state(
                collected_data={
                    "request_type": "infrastructure-provisioning",
                    "target_environment": "production",
                    "business_justification": "Need servers",
                    "name": "Jane",
                    "employee_id": "EMP001",
                }
            )
            result = await workflow.duplicate_check_node(state)

        assert result.get("awaiting_duplicate_decision") is True
        assert len(result.get("duplicate_warning", [])) == 1
        assert result["is_complete"] is False

    @pytest.mark.asyncio
    async def test_no_duplicate_when_judge_says_false(self, workflow):
        """LLM judge returning is_duplicate=False leads to empty result."""
        import uuid
        from datetime import datetime

        candidate_id = str(uuid.uuid4())
        similar_row = {
            "_id": candidate_id,
            "similarity_score": 0.92,
            "created_at": datetime.utcnow(),
            "config_version": "v1",
            "data": {
                "request_type": "infrastructure-provisioning",
                "target_environment": "development",
            },
        }
        verdict = DuplicateJudgement(
            is_duplicate=False, reasoning="Different environment"
        )

        with (
            patch("src.core.workflow.app_config") as mock_cfg,
            patch(
                "src.core.workflow.find_fuzzy_candidates",
                new_callable=AsyncMock,
                return_value=[candidate_id],
            ),
            patch(
                "src.core.workflow.find_similar_requests",
                new_callable=AsyncMock,
                return_value=[similar_row],
            ),
        ):
            mock_cfg.duplicate_detection_enabled = True
            mock_cfg.lookback_days = 90
            mock_cfg.fuzzy_threshold = 0.3
            mock_cfg.candidate_limit = 50
            mock_cfg.semantic_search_enabled = True
            mock_cfg.similarity_threshold = 0.85
            mock_cfg.pii_fields = []

            workflow.llm_duplicate_judge.ainvoke = AsyncMock(return_value=verdict)
            workflow.embedding_model.aembed_query = AsyncMock(
                return_value=[0.1] * 1536
            )

            state = _make_state(
                collected_data={
                    "request_type": "infrastructure-provisioning",
                    "target_environment": "production",
                    "business_justification": "Need servers",
                    "name": "Jane",
                    "employee_id": "EMP001",
                }
            )
            result = await workflow.duplicate_check_node(state)

        # Judge cleared the candidate — no warning
        assert result == {}

    @pytest.mark.asyncio
    async def test_semantic_search_disabled_skips_vector_step(self, workflow):
        """When semantic_search_enabled=False, find_similar_requests is not called."""
        import uuid

        candidate_id = str(uuid.uuid4())

        with (
            patch("src.core.workflow.app_config") as mock_cfg,
            patch(
                "src.core.workflow.find_fuzzy_candidates",
                new_callable=AsyncMock,
                return_value=[candidate_id],
            ),
            patch(
                "src.core.workflow.find_similar_requests",
                new_callable=AsyncMock,
            ) as mock_vector,
        ):
            mock_cfg.duplicate_detection_enabled = True
            mock_cfg.lookback_days = 90
            mock_cfg.fuzzy_threshold = 0.3
            mock_cfg.candidate_limit = 50
            mock_cfg.semantic_search_enabled = False
            mock_cfg.pii_fields = []

            state = _make_state(
                collected_data={
                    "request_type": "infrastructure-provisioning",
                    "business_justification": "Need servers",
                    "name": "Jane",
                    "employee_id": "EMP001",
                }
            )
            result = await workflow.duplicate_check_node(state)

        mock_vector.assert_not_awaited()
        assert result == {}


# ---------------------------------------------------------------------------
# handle_duplicate_decision_node
# ---------------------------------------------------------------------------


class TestHandleDuplicateDecisionNode:
    """Tests for ConversationWorkflow.handle_duplicate_decision_node."""

    @pytest.mark.asyncio
    async def test_choice_modify_clears_data_and_returns_to_chat(self, workflow):
        workflow.llm_duplicate_decision.ainvoke = AsyncMock(
            return_value=DuplicateDecision(
                choice="modify", reasoning="User wants to change"
            )
        )

        state = _make_state(
            messages=[HumanMessage(content="modify")],
            collected_data={"request_type": "infra"},
            awaiting_duplicate_decision=True,
        )
        result = await workflow.handle_duplicate_decision_node(state)

        assert result["duplicate_decision"] == "modify"
        assert result["collected_data"] == {}
        assert result["is_ready"] is False
        assert result["awaiting_duplicate_decision"] is False

    @pytest.mark.asyncio
    async def test_choice_proceed_keeps_data_and_signals_save(self, workflow):
        workflow.llm_duplicate_decision.ainvoke = AsyncMock(
            return_value=DuplicateDecision(
                choice="proceed", reasoning="Intentionally different"
            )
        )

        state = _make_state(
            messages=[HumanMessage(content="proceed")],
            awaiting_duplicate_decision=True,
        )
        result = await workflow.handle_duplicate_decision_node(state)

        assert result["duplicate_decision"] == "proceed"
        assert result["awaiting_duplicate_decision"] is False

    @pytest.mark.asyncio
    async def test_choice_cancel_ends_conversation(self, workflow):
        workflow.llm_duplicate_decision.ainvoke = AsyncMock(
            return_value=DuplicateDecision(
                choice="cancel", reasoning="User gives up"
            )
        )

        state = _make_state(
            messages=[HumanMessage(content="cancel")],
            awaiting_duplicate_decision=True,
        )
        result = await workflow.handle_duplicate_decision_node(state)

        assert result["duplicate_decision"] == "cancel"
        assert result["is_complete"] is False

    @pytest.mark.asyncio
    async def test_unrecognised_choice_sets_decision_none(self, workflow):
        workflow.llm_duplicate_decision.ainvoke = AsyncMock(
            return_value=DuplicateDecision(
                choice="gibberish", reasoning="Unknown"
            )
        )

        state = _make_state(
            messages=[HumanMessage(content="what?")],
            awaiting_duplicate_decision=True,
        )
        result = await workflow.handle_duplicate_decision_node(state)

        assert result["duplicate_decision"] is None
        assert "modify" in result["messages"][0].content or "catch" in result["messages"][0].content

    @pytest.mark.asyncio
    async def test_llm_exception_returns_safe_message(self, workflow):
        workflow.llm_duplicate_decision.ainvoke = AsyncMock(
            side_effect=RuntimeError("LLM error")
        )

        state = _make_state(
            messages=[HumanMessage(content="proceed")],
            awaiting_duplicate_decision=True,
        )
        result = await workflow.handle_duplicate_decision_node(state)

        assert result["duplicate_decision"] is None
        assert result["messages"]


# ---------------------------------------------------------------------------
# get_app
# ---------------------------------------------------------------------------


class TestGetApp:
    """Tests for ConversationWorkflow.get_app."""

    def test_get_app_raises_before_compile(self):
        with (
            patch("src.core.workflow.get_llm", return_value=MagicMock()),
            patch("src.core.workflow.get_embedding_model", return_value=AsyncMock()),
        ):
            wf = ConversationWorkflow(pg_client=MagicMock())

        with pytest.raises(RuntimeError, match="not compiled"):
            wf.get_app()

    def test_get_app_returns_app_after_compile(self):
        from langgraph.checkpoint.memory import InMemorySaver

        with (
            patch("src.core.workflow.get_llm", return_value=MagicMock()),
            patch("src.core.workflow.get_embedding_model", return_value=AsyncMock()),
        ):
            wf = ConversationWorkflow(pg_client=MagicMock())

        wf.compile(InMemorySaver())
        assert wf.get_app() is wf.app


# Made with Bob
