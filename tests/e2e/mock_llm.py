"""
Mock LLM and embedding classes for E2E testing.

Lives entirely in tests/e2e/ — zero coupling to src/.
Used by app_override.py which is the Docker entrypoint for E2E runs.

The mock plays back a deterministic 5-turn scripted conversation so
no real OpenAI / Azure / Anthropic API key is required.  The fixed
unit-vector embeddings guarantee that a second identical submission
always crosses the duplicate-detection similarity threshold.
"""

import math
from typing import Any, List, Optional

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import Field

# ---------------------------------------------------------------------------
# Scripted conversation — one reply per turn
# ---------------------------------------------------------------------------

SCRIPT: List[str] = [
    # Turn 1
    "I'm here to help you submit a request. "
    "What type of request do you have? "
    "(e.g. infrastructure-provisioning, access-request)",
    # Turn 2
    "Got it! What target environment is this for? "
    "(development, staging, or production)",
    # Turn 3
    "Thanks. Please provide a brief business justification.",
    # Turn 4
    "Almost done! Please provide your full name and employee ID "
    "in the format: NAME,EMPLOYEE_ID",
    # Turn 5 — triggers is_ready=True in MockStructuredLLM
    "Thank you! Confirmed: infrastructure-provisioning for "
    "production. Your name is John Doe (EMP001). Submitting now.",
]

# Fixed 1536-d unit-vector embedding — identical for every query so
# duplicate detection always fires on the second identical submission.
_DIM = 1536
_UNIT = 1.0 / math.sqrt(_DIM)
FIXED_EMBEDDING: List[float] = [_UNIT] * _DIM


# ---------------------------------------------------------------------------
# MockStructuredLLM
# ---------------------------------------------------------------------------

class MockStructuredLLM:
    """
    Returned by MockChatModel.with_structured_output().

    Dispatches on the Pydantic schema name and returns a hardcoded
    instance of the appropriate type so every workflow node gets a
    valid object without touching a real LLM.
    """

    def __init__(self, base: "MockChatModel", schema: Any) -> None:
        self._base = base
        self._schema = schema

    # ------------------------------------------------------------------
    # Internal dispatch
    # ------------------------------------------------------------------

    def _dispatch(self, messages: Any) -> Any:
        # Import workflow types here to avoid circular imports at
        # module load time (app_override imports this before src.api)
        from src.core.schema import DataField, ExtractedRequest
        from src.core.workflow import (
            ChatResponse,
            DuplicateDecision,
            DuplicateJudgement,
        )

        name = getattr(
            self._schema, "__name__", str(self._schema)
        )

        if name == "ChatResponse":
            idx = self._base.call_count % len(SCRIPT)
            is_last = idx == len(SCRIPT) - 1
            # Advance the counter on the base model
            object.__setattr__(
                self._base, "call_count", self._base.call_count + 1
            )
            return ChatResponse(
                response=SCRIPT[idx], is_ready=is_last
            )

        if name == "ExtractedRequest":
            return ExtractedRequest(
                request_type="infrastructure-provisioning",
                name="John Doe",
                employee_id="EMP001",
                additional_data=[
                    DataField(
                        key="target_environment",
                        value="production",
                    ),
                    DataField(
                        key="business_justification",
                        value="E2E test submission",
                    ),
                ],
            )

        if name == "DuplicateJudgement":
            return DuplicateJudgement(
                is_duplicate=True,
                reasoning="Same request type and environment",
            )

        if name == "DuplicateDecision":
            # Reflect the actual last user message so that
            # sending "cancel" cancels, "proceed" proceeds, etc.
            last_user_msg = ""
            if hasattr(messages, "__iter__"):
                for m in reversed(list(messages)):
                    content = getattr(m, "content", "")
                    # HumanMessage role
                    if getattr(m, "type", "") == "human" or \
                       m.__class__.__name__ == "HumanMessage":
                        last_user_msg = content.strip().lower()
                        break
            if "cancel" in last_user_msg:
                choice = "cancel"
            elif "modify" in last_user_msg:
                choice = "modify"
            else:
                choice = "proceed"
            return DuplicateDecision(
                choice=choice,
                reasoning=f"User said: {last_user_msg}",
            )

        # Fallback for any other schema
        return self._schema()

    def invoke(self, messages: Any, **kwargs: Any) -> Any:
        return self._dispatch(messages)

    async def ainvoke(self, messages: Any, **kwargs: Any) -> Any:
        return self._dispatch(messages)


# ---------------------------------------------------------------------------
# MockChatModel
# ---------------------------------------------------------------------------

class MockChatModel(BaseChatModel):
    """
    Scripted BaseChatModel for E2E tests.

    Keeps a call counter so each invocation returns the next line
    from SCRIPT in order.  with_structured_output() returns a
    MockStructuredLLM that drives the LangGraph workflow.
    """

    call_count: int = Field(default=0)

    @property
    def _llm_type(self) -> str:
        return "mock"

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> ChatResult:
        idx = self.call_count % len(SCRIPT)
        object.__setattr__(
            self, "call_count", self.call_count + 1
        )
        return ChatResult(
            generations=[
                ChatGeneration(
                    message=AIMessage(content=SCRIPT[idx])
                )
            ]
        )

    def with_structured_output(
        self, schema: Any, **kwargs: Any
    ) -> MockStructuredLLM:
        return MockStructuredLLM(base=self, schema=schema)


# ---------------------------------------------------------------------------
# MockEmbeddings
# ---------------------------------------------------------------------------

class MockEmbeddings:
    """Returns the fixed unit-vector for every query."""

    async def aembed_query(self, text: str) -> List[float]:
        return list(FIXED_EMBEDDING)

    def embed_query(self, text: str) -> List[float]:
        return list(FIXED_EMBEDDING)


# Made with Bob
