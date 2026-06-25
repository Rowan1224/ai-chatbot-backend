"""FastAPI application for AI Chatbot Backend."""

import logging
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, HTTPException, Header, status
from fastapi.middleware.cors import CORSMiddleware
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.redis import AsyncRedisSaver

from src.api.models import (
    ChatRequest,
    ChatResponse,
    SessionResponse,
    HealthResponse,
)
from src.config.settings import settings
from src.core.database import PostgreSQLClient, RedisClient
from src.core.workflow import ConversationWorkflow

# Configure logging
logging.basicConfig(
    level=settings.log_level,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# Lifespan context manager for startup/shutdown
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Handle startup and shutdown events."""
    # Startup
    logger.info("Starting AI Chatbot Backend...")
    
    # Initialize database clients
    pg_client = PostgreSQLClient()
    redis_client = RedisClient()

    # Connect to PostgreSQL
    await pg_client.connect()
    logger.info("PostgreSQL connected")
    
    # Create LangGraph checkpointer based on environment
    if settings.use_redis_checkpointer:
        # Production/Test/Acc: Use Redis for persistence
        await redis_client.connect()
        logger.info("Redis connected")
        
        redis_conn = await redis_client.get_client()
        checkpointer = AsyncRedisSaver(redis_client=redis_conn)
        await checkpointer.asetup()
        
        logger.info(f"Using Redis checkpointer at {settings.redis_url}")
    else:
        # Development: Use in-memory (no persistence between restarts)
        checkpointer = InMemorySaver()
        logger.info("Using InMemory checkpointer (dev mode - no persistence)")
    
    # Initialize workflow with clients and compile with checkpointer
    workflow = ConversationWorkflow(pg_client=pg_client)
    conversation_app = workflow.compile(checkpointer=checkpointer)
    logger.info("LangGraph workflow compiled")

    # Store in app state for access in endpoints
    app.state.pg_client = pg_client
    app.state.redis_client = redis_client
    app.state.conversation_app = conversation_app
    
    yield
    
    # Shutdown
    logger.info("Shutting down AI Chatbot Backend...")
    await pg_client.close()
    logger.info("PostgreSQL connection closed")
    
    if settings.use_redis_checkpointer:
        await redis_client.close()
        logger.info("Redis connection closed")


# Create FastAPI app
app = FastAPI(
    title="AI Chatbot Backend",
    description="Conversational AI for structured data collection with duplicate detection",
    version="1.0.0",
    lifespan=lifespan,
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# API Key Authentication Dependency
async def verify_api_key(x_api_key: str = Header(..., alias="X-API-Key")) -> None:
    """
    Verify API key from header.
    
    This is a FastAPI dependency that can be injected into endpoints.
    Raises HTTPException if the API key is invalid.
    """
    if x_api_key != settings.api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key",
        )


# Endpoints
@app.post("/chat", response_model=ChatResponse, tags=["Chat"])
async def chat(
    request: ChatRequest,
    _: None = Depends(verify_api_key),
) -> ChatResponse:
    """
    Send a message to the chatbot and get a response.
    
    The chatbot maintains conversation state using the session_id and handles
    the complete conversation flow including:
    1. Collecting request details through natural conversation
    2. Extracting structured data when ready
    3. Collecting PII (name and employee ID) in format: NAME,EMPLOYEE_ID
    4. Confirming PII with the user
    5. Checking for duplicate requests
    6. Saving the request to MongoDB
    
    When the request is successfully saved, is_complete will be True.
    
    If duplicates are found, the bot will ask if you want to update the existing
    request or create a new one. Respond with 'update' or 'new'.
    """
    logger.info(f"Chat request from session {request.session_id}: {request.message}")
    
    try:
        # Invoke LangGraph workflow
        result = await app.state.conversation_app.ainvoke(
            {"messages": [HumanMessage(content=request.message)]},
            config={"configurable": {"thread_id": request.session_id}},
        )
        
        # Extract response from result
        last_message = result["messages"][-1] if result.get("messages") else None
        response_text = last_message.content if last_message else "No response generated"
        
        logger.info(f"Chat response for session {request.session_id}: is_complete={result.get('is_complete', False)}")
        
        return ChatResponse(
            session_id=request.session_id,
            response=response_text,
            is_ready=result.get("is_ready", False),
            is_complete=result.get("is_complete", False),
            collected_data=result.get("collected_data"),
            duplicate_warning=result.get("duplicate_warning"),
        )
        
    except Exception as e:
        logger.error(f"Error in chat endpoint: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to process chat message: {str(e)}",
        )


@app.get("/session/{session_id}", response_model=SessionResponse, tags=["Debug"])
async def get_session(
    session_id: str,
    _: None = Depends(verify_api_key),
) -> SessionResponse:
    """
    Get the current state of a session (for debugging).
    
    Returns the full conversation state including messages and collected data.
    """
    try:
        state = app.state.conversation_app.get_state(
            config={"configurable": {"thread_id": session_id}}
        )
        
        return SessionResponse(
            session_id=session_id,
            state=state.values,
        )
        
    except Exception as e:
        logger.error(f"Error getting session state: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get session state: {str(e)}",
        )


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check() -> HealthResponse:
    """
    Health check endpoint for Docker and monitoring.
    
    Checks connectivity to MongoDB and Redis.
    """
    mongodb_status = "disconnected"
    redis_status = "disconnected"
    
    try:
        # Check PostgreSQL
        await app.state.pg_client.ping()
        mongodb_status = "connected"
    except Exception as e:
        logger.error(f"PostgreSQL health check failed: {e}")

    try:
        # Check Redis (only if using Redis checkpointer)
        if settings.use_redis_checkpointer:
            client = await app.state.redis_client.get_client()
            await client.ping()
            redis_status = "connected"
        else:
            redis_status = "disabled (dev mode)"
    except Exception as e:
        logger.error(f"Redis health check failed: {e}")
    
    overall_status = (
        "healthy"
        if mongodb_status == "connected"
        and redis_status == "connected"
        else "unhealthy"
    )

    return HealthResponse(
        status=overall_status,
        mongodb=mongodb_status,
        redis=redis_status,
    )


@app.get("/", tags=["Root"])
async def root():
    """Root endpoint with API information."""
    return {
        "name": "AI Chatbot Backend",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health",
    }

# Made with Bob
