"""
Integration tests for Redis session persistence via LangGraph checkpointer.

Verifies that AsyncRedisSaver correctly persists conversation state across
workflow invocations — the real Redis container is used (no mocks).
LLM and embeddings are mocked throughout.
"""

import math
import uuid
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.redis import AsyncRedisSaver

from src.core.workflow import (
    ChatResponse as WorkflowChatResponse,
)
from src.core.workflow import ConversationWorkflow

pytestmark = pytest.mark.integration

_DIM = 1536
_SCALE = 1.0 / math.sqrt(_DIM)
EMBED = [_SCALE] * _DIM


# ---------------------------------------------------------------------------
# Fixture — workflow compiled with real AsyncRedisSaver
# ---------------------------------------------------------------------------


def _make_workflow(pg_client) -> ConversationWorkflow:
    mock_llm = MagicMock()
    mock_embed = AsyncMock()
    mock_embed.aembed_query = AsyncMock(return_value=EMBED)

    with (
        patch("src.core.workflow.get_llm", return_value=mock_llm),
        patch("src.core.workflow.get_embedding_model", return_value=mock_embed),
    ):
        wf = ConversationWorkflow(pg_client=pg_client)

    wf.llm_chat = MagicMock()
    wf.llm_extract = MagicMock()
    wf.llm_duplicate_judge = AsyncMock()
    wf.llm_duplicate_decision = AsyncMock()
    wf.embedding_model = mock_embed
    return wf


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestRedisSessionPersistence:

    @pytest.mark.asyncio
    async def test_state_persists_across_turns(
        self, pg_client, redis_url, clean_db
    ):
        """
        Two sequential invocations with the same thread_id and a
        real AsyncRedisSaver must share state: the second turn must
        see messages written by the first.
        """
        from redis.asyncio import Redis as AsyncRedis

        redis_conn = AsyncRedis.from_url(redis_url, decode_responses=False)
        checkpointer = AsyncRedisSaver(redis_client=redis_conn)
        await checkpointer.asetup()

        wf = _make_workflow(pg_client)
        wf.llm_chat.ainvoke = AsyncMock(
            return_value=WorkflowChatResponse(
                response="What environment?", is_ready=False
            )
        )

        app = wf.compile(checkpointer)
        tid = f"redis-persist-{uuid.uuid4().hex[:8]}"

        result1 = await app.ainvoke(
            {"messages": [HumanMessage(content="I need infrastructure")]},
            config={"configurable": {"thread_id": tid}},
        )
        msgs_t1 = len(result1["messages"])

        result2 = await app.ainvoke(
            {"messages": [HumanMessage(content="Production environment")]},
            config={"configurable": {"thread_id": tid}},
        )
        msgs_t2 = len(result2["messages"])

        assert msgs_t2 > msgs_t1, (
            "Message count did not grow — Redis checkpointer did not persist state"
        )

        await redis_conn.aclose()

    @pytest.mark.asyncio
    async def test_different_sessions_are_isolated(
        self, pg_client, redis_url, clean_db
    ):
        """
        Two different thread_ids must produce independent state — messages
        from session A must not appear in session B.
        """
        from redis.asyncio import Redis as AsyncRedis

        redis_conn = AsyncRedis.from_url(redis_url, decode_responses=False)
        checkpointer = AsyncRedisSaver(redis_client=redis_conn)
        await checkpointer.asetup()

        wf = _make_workflow(pg_client)
        wf.llm_chat.ainvoke = AsyncMock(
            return_value=WorkflowChatResponse(response="ok", is_ready=False)
        )

        app = wf.compile(checkpointer)

        tid_a = f"redis-a-{uuid.uuid4().hex[:8]}"
        tid_b = f"redis-b-{uuid.uuid4().hex[:8]}"

        result_a = await app.ainvoke(
            {"messages": [HumanMessage(content="Message from A")]},
            config={"configurable": {"thread_id": tid_a}},
        )
        result_b = await app.ainvoke(
            {"messages": [HumanMessage(content="Message from B")]},
            config={"configurable": {"thread_id": tid_b}},
        )

        msgs_a = [m.content for m in result_a["messages"]]
        msgs_b = [m.content for m in result_b["messages"]]

        assert "Message from A" not in msgs_b
        assert "Message from B" not in msgs_a

        await redis_conn.aclose()

    @pytest.mark.asyncio
    async def test_state_survives_workflow_reinstantiation(
        self, pg_client, redis_url, clean_db
    ):
        """
        State saved by one workflow instance must be readable by a new
        instance connected to the same Redis — simulates an app restart.
        """
        from redis.asyncio import Redis as AsyncRedis

        tid = f"redis-restart-{uuid.uuid4().hex[:8]}"

        # --- First workflow instance ---
        redis_conn1 = AsyncRedis.from_url(redis_url, decode_responses=False)
        checkpointer1 = AsyncRedisSaver(redis_client=redis_conn1)
        await checkpointer1.asetup()

        wf1 = _make_workflow(pg_client)
        wf1.llm_chat.ainvoke = AsyncMock(
            return_value=WorkflowChatResponse(
                response="Turn 1 response", is_ready=False
            )
        )
        app1 = wf1.compile(checkpointer1)

        result1 = await app1.ainvoke(
            {"messages": [HumanMessage(content="Turn 1 message")]},
            config={"configurable": {"thread_id": tid}},
        )
        msgs_after_t1 = len(result1["messages"])
        await redis_conn1.aclose()

        # --- Second workflow instance (simulates restart) ---
        redis_conn2 = AsyncRedis.from_url(redis_url, decode_responses=False)
        checkpointer2 = AsyncRedisSaver(redis_client=redis_conn2)
        await checkpointer2.asetup()

        wf2 = _make_workflow(pg_client)
        wf2.llm_chat.ainvoke = AsyncMock(
            return_value=WorkflowChatResponse(
                response="Turn 2 response", is_ready=False
            )
        )
        app2 = wf2.compile(checkpointer2)

        result2 = await app2.ainvoke(
            {"messages": [HumanMessage(content="Turn 2 message")]},
            config={"configurable": {"thread_id": tid}},
        )
        msgs_after_t2 = len(result2["messages"])
        await redis_conn2.aclose()

        assert msgs_after_t2 > msgs_after_t1, (
            "State was not carried over to the new workflow instance — "
            "Redis did not persist across reinstantiation"
        )


# Made with Bob
