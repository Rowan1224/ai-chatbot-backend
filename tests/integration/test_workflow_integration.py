"""
Integration tests for the ConversationWorkflow node chain.

Wires a real ConversationWorkflow against a live PostgreSQL container
while mocking only the LLM and embedding model.  Verifies that the
node collaboration (chat → extract → duplicate_check → save) produces
the correct database side-effects and state transitions.
"""

import math
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import InMemorySaver

from src.core.database import save_request
from src.core.schema import DataField, ExtractedRequest
from src.core.workflow import (
    ChatResponse,
    ConversationWorkflow,
    DuplicateDecision,
    DuplicateJudgement,
)

pytestmark = pytest.mark.integration

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_DIM = 1536
_SCALE = 1.0 / math.sqrt(_DIM)
EMBED_MATCH = [_SCALE] * _DIM          # identical to itself — similarity = 1.0
EMBED_DIFFER = [-_SCALE] * _DIM        # opposite direction — similarity ≈ -1


def _make_workflow(pg_client) -> ConversationWorkflow:
    """
    Build a ConversationWorkflow with mocked LLM and embedding.
    The LLM attributes are replaced after construction so each test
    can control them independently.
    """
    mock_llm = MagicMock()
    mock_embed = AsyncMock()
    mock_embed.aembed_query = AsyncMock(return_value=EMBED_MATCH)

    with (
        patch("src.core.workflow.get_llm", return_value=mock_llm),
        patch("src.core.workflow.get_embedding_model", return_value=mock_embed),
    ):
        wf = ConversationWorkflow(pg_client=pg_client)

    # Replace structured LLMs with controllable async mocks
    wf.llm_chat = MagicMock()
    wf.llm_extract = MagicMock()
    wf.llm_duplicate_judge = AsyncMock()
    wf.llm_duplicate_decision = AsyncMock()
    wf.embedding_model = mock_embed
    return wf


async def _row_count(pg_client) -> int:
    """Return total number of rows in the requests table."""
    async with pg_client.pool.acquire() as conn:
        return await conn.fetchval("SELECT COUNT(*) FROM requests")


# ---------------------------------------------------------------------------
# Happy path — full save
# ---------------------------------------------------------------------------


class TestWorkflowHappyPath:

    @pytest.mark.asyncio
    async def test_happy_path_saves_row_to_database(self, pg_client, clean_db):
        """
        When the LLM signals is_ready=True and extraction succeeds,
        the workflow must persist exactly one row in PostgreSQL.
        """
        wf = _make_workflow(pg_client)

        # chat node: LLM says all info is collected
        wf.llm_chat.ainvoke = AsyncMock(
            return_value=ChatResponse(
                response="Got everything, submitting now.", is_ready=True
            )
        )
        # extract node: returns a valid structured request
        wf.llm_extract.ainvoke = AsyncMock(
            return_value=ExtractedRequest(
                request_type="infrastructure-provisioning",
                name="Jane",
                employee_id="EMP001",
                additional_data=[
                    DataField(key="target_environment", value="production"),
                    DataField(
                        key="business_justification", value="Need servers"
                    ),
                ],
            )
        )
        # duplicate judge: no duplicates (not called unless similar rows exist)
        wf.llm_duplicate_judge.ainvoke = AsyncMock(
            return_value=DuplicateJudgement(
                is_duplicate=False, reasoning="No match"
            )
        )

        app = wf.compile(InMemorySaver())

        await app.ainvoke(
            {"messages": [HumanMessage(content="I need infrastructure")]},
            config={"configurable": {"thread_id": "thread-happy"}},
        )

        count = await _row_count(pg_client)
        assert count == 1, f"Expected 1 row, found {count}"

    @pytest.mark.asyncio
    async def test_happy_path_response_is_complete(self, pg_client, clean_db):
        """result['is_complete'] must be True after a successful save."""
        wf = _make_workflow(pg_client)
        wf.llm_chat.ainvoke = AsyncMock(
            return_value=ChatResponse(response="Done.", is_ready=True)
        )
        wf.llm_extract.ainvoke = AsyncMock(
            return_value=ExtractedRequest(
                request_type="service-deployment",
                name="Bob",
                employee_id="EMP002",
                additional_data=[
                    DataField(key="target_environment", value="staging"),
                    DataField(key="business_justification", value="Deploy v2"),
                ],
            )
        )
        wf.llm_duplicate_judge.ainvoke = AsyncMock(
            return_value=DuplicateJudgement(
                is_duplicate=False, reasoning="No match"
            )
        )

        app = wf.compile(InMemorySaver())
        result = await app.ainvoke(
            {"messages": [HumanMessage(content="Deploy service")]},
            config={"configurable": {"thread_id": "thread-complete"}},
        )

        assert result.get("is_complete") is True


# ---------------------------------------------------------------------------
# Extraction failure
# ---------------------------------------------------------------------------


class TestExtractFailure:

    @pytest.mark.asyncio
    async def test_extract_failure_does_not_save(self, pg_client, clean_db):
        """If extraction throws, no row must be saved."""
        wf = _make_workflow(pg_client)
        wf.llm_chat.ainvoke = AsyncMock(
            return_value=ChatResponse(response="Ready.", is_ready=True)
        )
        wf.llm_extract.ainvoke = AsyncMock(
            side_effect=RuntimeError("LLM timed out")
        )

        app = wf.compile(InMemorySaver())
        result = await app.ainvoke(
            {"messages": [HumanMessage(content="deploy")]},
            config={"configurable": {"thread_id": "thread-extract-fail"}},
        )

        count = await _row_count(pg_client)
        assert count == 0, "No row should be saved when extraction fails"
        assert result.get("is_complete") is not True


# ---------------------------------------------------------------------------
# Duplicate detection interaction
# ---------------------------------------------------------------------------


class TestDuplicateDetectionInteraction:

    @pytest.mark.asyncio
    async def test_duplicate_detected_does_not_save_immediately(
        self, pg_client, clean_db
    ):
        """
        When a near-identical row already exists, the workflow must
        set awaiting_duplicate_decision=True and NOT save a new row.
        """
        # Seed an existing identical row
        existing_data = {
            "request_type": "infrastructure-provisioning",
            "target_environment": "production",
            "business_justification": "Need servers",
            "name": "Existing User",
            "employee_id": "EMP-EXISTING",
        }
        await save_request(pg_client, existing_data, EMBED_MATCH)

        wf = _make_workflow(pg_client)
        # Use the same embedding so cosine similarity = 1.0
        wf.embedding_model.aembed_query = AsyncMock(return_value=EMBED_MATCH)

        wf.llm_chat.ainvoke = AsyncMock(
            return_value=ChatResponse(response="Ready.", is_ready=True)
        )
        wf.llm_extract.ainvoke = AsyncMock(
            return_value=ExtractedRequest(
                request_type="infrastructure-provisioning",
                name="New User",
                employee_id="EMP-NEW",
                additional_data=[
                    DataField(key="target_environment", value="production"),
                    DataField(
                        key="business_justification", value="Need servers"
                    ),
                ],
            )
        )
        # LLM judge says it IS a duplicate
        wf.llm_duplicate_judge.ainvoke = AsyncMock(
            return_value=DuplicateJudgement(
                is_duplicate=True,
                reasoning="Same request type, environment, and justification",
            )
        )

        with patch("src.core.workflow.app_config") as mock_settings:
            mock_settings.duplicate_detection_enabled = True
            mock_settings.semantic_search_enabled = True
            mock_settings.similarity_threshold = 0.85
            mock_settings.lookback_days = 90
            mock_settings.fuzzy_threshold = 0.3
            mock_settings.candidate_limit = 50

            app = wf.compile(InMemorySaver())
            result = await app.ainvoke(
                {"messages": [HumanMessage(content="Need infrastructure")]},
                config={"configurable": {"thread_id": "thread-dup"}},
            )

        # Only the seed row must exist — no new row saved
        count = await _row_count(pg_client)
        assert count == 1, "No new row should be saved when duplicate is detected"
        assert result.get("awaiting_duplicate_decision") is True

    @pytest.mark.asyncio
    async def test_proceed_after_duplicate_saves_new_row(
        self, pg_client, clean_db
    ):
        """
        When user replies 'proceed' after a duplicate warning,
        a second row must be saved (total = 2).
        """
        # Seed existing row
        existing_data = {
            "request_type": "infrastructure-provisioning",
            "target_environment": "production",
            "business_justification": "Need servers",
            "name": "Existing User",
            "employee_id": "EMP-EXISTING",
        }
        await save_request(pg_client, existing_data, EMBED_MATCH)

        wf = _make_workflow(pg_client)
        wf.embedding_model.aembed_query = AsyncMock(return_value=EMBED_MATCH)

        wf.llm_chat.ainvoke = AsyncMock(
            return_value=ChatResponse(response="Ready.", is_ready=True)
        )
        wf.llm_extract.ainvoke = AsyncMock(
            return_value=ExtractedRequest(
                request_type="infrastructure-provisioning",
                name="New User",
                employee_id="EMP-NEW",
                additional_data=[
                    DataField(key="target_environment", value="production"),
                    DataField(key="business_justification", value="Need servers"),
                ],
            )
        )
        wf.llm_duplicate_judge.ainvoke = AsyncMock(
            return_value=DuplicateJudgement(
                is_duplicate=True, reasoning="Same request"
            )
        )
        # Decision: user proceeds
        wf.llm_duplicate_decision.ainvoke = AsyncMock(
            return_value=DuplicateDecision(
                choice="proceed",
                reasoning="Request is intentionally different",
            )
        )

        checkpointer = InMemorySaver()

        with patch("src.core.workflow.app_config") as mock_settings:
            mock_settings.duplicate_detection_enabled = True
            mock_settings.semantic_search_enabled = True
            mock_settings.similarity_threshold = 0.85
            mock_settings.lookback_days = 90
            mock_settings.fuzzy_threshold = 0.3
            mock_settings.candidate_limit = 50

            app = wf.compile(checkpointer)

            # Turn 1: triggers duplicate warning
            await app.ainvoke(
                {"messages": [HumanMessage(content="Need infrastructure")]},
                config={"configurable": {"thread_id": "thread-proceed"}},
            )

            # Turn 2: user sends "proceed"
            await app.ainvoke(
                {"messages": [HumanMessage(content="proceed")]},
                config={"configurable": {"thread_id": "thread-proceed"}},
            )

        count = await _row_count(pg_client)
        assert count == 2, (
            f"Expected 2 rows after 'proceed' decision, found {count}"
        )


# ---------------------------------------------------------------------------
# Session state persists across turns (InMemorySaver)
# ---------------------------------------------------------------------------


class TestSessionPersistenceInMemory:

    @pytest.mark.asyncio
    async def test_messages_accumulate_across_turns(self, pg_client, clean_db):
        """
        Two sequential invocations with the same thread_id must share state:
        the second turn must see messages from the first.
        """
        wf = _make_workflow(pg_client)
        # Not ready yet — keep the conversation going
        wf.llm_chat.ainvoke = AsyncMock(
            return_value=ChatResponse(
                response="What environment?", is_ready=False
            )
        )

        checkpointer = InMemorySaver()
        app = wf.compile(checkpointer)

        tid = "thread-multi-turn"

        result1 = await app.ainvoke(
            {"messages": [HumanMessage(content="I need infra")]},
            config={"configurable": {"thread_id": tid}},
        )
        msgs_after_t1 = len(result1["messages"])

        result2 = await app.ainvoke(
            {"messages": [HumanMessage(content="Production environment")]},
            config={"configurable": {"thread_id": tid}},
        )
        msgs_after_t2 = len(result2["messages"])

        # State must have grown — second turn adds to the existing messages
        assert msgs_after_t2 > msgs_after_t1, (
            "Message list did not grow across turns — session state not persisted"
        )

    @pytest.mark.asyncio
    async def test_different_thread_ids_are_isolated(self, pg_client, clean_db):
        """Two different thread_ids must produce independent state."""
        wf = _make_workflow(pg_client)
        wf.llm_chat.ainvoke = AsyncMock(
            return_value=ChatResponse(response="ok", is_ready=False)
        )

        app = wf.compile(InMemorySaver())

        result_a = await app.ainvoke(
            {"messages": [HumanMessage(content="Session A message")]},
            config={"configurable": {"thread_id": "thread-a"}},
        )
        result_b = await app.ainvoke(
            {"messages": [HumanMessage(content="Session B message")]},
            config={"configurable": {"thread_id": "thread-b"}},
        )

        msgs_a = [m.content for m in result_a["messages"]]
        msgs_b = [m.content for m in result_b["messages"]]

        # Session B must not contain the message from Session A
        assert "Session A message" not in msgs_b
        # Session A must not contain the message from Session B
        assert "Session B message" not in msgs_a


# Made with Bob
