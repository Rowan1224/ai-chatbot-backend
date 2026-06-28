"""API request and response models."""

from typing import Any

from pydantic import BaseModel, Field


# Request Models
class ChatRequest(BaseModel):
    """Request model for chat endpoint."""

    session_id: str = Field(
        ...,
        min_length=1,
        max_length=128,
        pattern=r"^[a-zA-Z0-9_\-]+$",
        description=(
            "Unique session identifier. "
            "1–128 alphanumeric, hyphen, or underscore characters."
        ),
    )
    message: str = Field(
        ...,
        min_length=1,
        max_length=2000,
        description=(
            "User's message. "
            "1–2000 characters."
        ),
    )


# Response Models
class ChatResponse(BaseModel):
    """Response model for chat endpoint."""

    session_id: str = Field(..., description="Session identifier")
    response: str = Field(..., description="Bot's response message")
    is_ready: bool = Field(default=False, description="True when ready to extract data")
    is_complete: bool = Field(default=False, description="True when request submission is complete")
    request_active: bool = Field(
        default=True,
        description="True while the current request flow is still active",
    )
    collected_data: dict[str, Any] | None = Field(
        default=None, description="Collected structured data (if available)"
    )
    duplicate_warning: list[dict[str, Any]] | None = Field(
        default=None, description="Similar requests found (if any)"
    )


class SessionResponse(BaseModel):
    """Response model for session inspection."""

    session_id: str
    state: dict[str, Any]


class HealthResponse(BaseModel):
    """Response model for health check."""

    status: str
    postgresql: str



