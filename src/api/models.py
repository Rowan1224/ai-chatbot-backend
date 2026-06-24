"""API request and response models."""

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


# Request Models
class ChatRequest(BaseModel):
    """Request model for chat endpoint."""

    session_id: str = Field(..., description="Unique session identifier")
    message: str = Field(..., description="User's message")


# Response Models
class ChatResponse(BaseModel):
    """Response model for chat endpoint."""

    session_id: str = Field(..., description="Session identifier")
    response: str = Field(..., description="Bot's response message")
    is_ready: bool = Field(default=False, description="True when ready to extract data")
    is_complete: bool = Field(default=False, description="True when data extraction complete")
    collected_data: Optional[Dict[str, Any]] = Field(
        default=None, description="Collected structured data (if complete)"
    )
    duplicate_warning: Optional[List[Dict[str, Any]]] = Field(
        default=None, description="Similar requests found (if any)"
    )


class SessionResponse(BaseModel):
    """Response model for session inspection."""

    session_id: str
    state: Dict[str, Any]


class HealthResponse(BaseModel):
    """Response model for health check."""

    status: str
    mongodb: str
    redis: str


# Made with Bob