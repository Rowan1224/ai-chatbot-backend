"""Pydantic models and state definitions for the conversation workflow."""

import operator
from typing import Annotated, Any, Dict, List, Optional, TypedDict

from langchain_core.messages import BaseMessage
from pydantic import BaseModel, Field


class ChatResponse(BaseModel):
    """Structured response from chat node."""

    response: str = Field(description="Bot's response message to the user")
    is_ready: bool = Field(
        description="True when all required information has been collected and ready to extract"
    )


class DuplicateJudgement(BaseModel):
    """LLM verdict on whether a candidate record is a true duplicate."""

    is_duplicate: bool = Field(
        description=(
            "True if the candidate request represents the same intent "
            "as the new request — i.e. same purpose, same target, same "
            "scope.  False if any meaningful field differs (e.g. "
            "different environment, system, date, or access level)."
        )
    )
    reasoning: str = Field(
        description=(
            "One sentence explaining what is the same or different "
            "between the two requests."
        )
    )


class DuplicateDecision(BaseModel):
    """Structured response for duplicate decision."""

    choice: str = Field(
        description=(
            "User's choice — one of:\n"
            "  'modify'  — user wants to go back to chat and change "
            "something in their current request\n"
            "  'proceed' — the request is intentionally different, "
            "save it as a new request anyway\n"
            "  'cancel'  — abandon this request without saving"
        )
    )
    reasoning: str = Field(
        description="Brief explanation of why the user made this choice"
    )


class ConversationState(TypedDict, total=False):
    """State for the conversation workflow.

    ``total=False`` makes every key optional so LangGraph can build
    the state incrementally — keys are only present once a node sets
    them.  Nodes must use ``.get()`` with sensible defaults rather
    than direct key access.
    """

    messages: Annotated[List[BaseMessage], operator.add]
    collected_data: Dict[str, Any]
    is_ready: bool
    is_complete: bool
    duplicate_warning: List[Dict[str, Any]]
    config_version: str
    awaiting_duplicate_decision: bool
    # 'modify' | 'proceed' | 'cancel' | None — set by
    # handle_duplicate_decision_node so the router can branch cleanly
    duplicate_decision: Optional[str]
