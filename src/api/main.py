"""FastAPI application for AI Chatbot Backend."""

import asyncio
import logging
import secrets
from collections import deque
from contextlib import asynccontextmanager
from time import monotonic

from fastapi import (
    Depends,
    FastAPI,
    Header,
    HTTPException,
    Request,
    status,
)
from fastapi.middleware.cors import CORSMiddleware
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from openai import BadRequestError

from src.api.models import (
    ChatRequest,
    ChatResponse,
    HealthResponse,
    SessionResponse,
)
from src.config.settings import app_config, settings
from src.core.database import PostgreSQLClient
from src.core.workflow import ConversationWorkflow

# Configure logging
logging.basicConfig(
    level=app_config.log_level,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# Lifespan context manager for startup/shutdown
# ---------------------------------------------------------------------------
# Per-session sliding-window rate limiter
# ---------------------------------------------------------------------------

# module-level store: session_id → deque of monotonic timestamps
_rate_limit_store: dict[str, deque[float]] = {}
_WINDOW = 60.0  # seconds


async def check_rate_limit(request: Request) -> None:
    """
    FastAPI dependency — enforces a per-session sliding-window rate limit.

    The limit (requests_per_minute) is read from app_config.yaml so it
    can be tuned without a code change.  Setting it to 0 disables the
    check entirely.
    """
    rpm = app_config.rate_limit_rpm
    if rpm <= 0:
        return  # rate limiting disabled

    # Extract session_id from the JSON body.
    # Body is already validated by Pydantic before this dep runs.
    body = await request.json()
    session_id = body.get("session_id", "")

    now = monotonic()
    window = _rate_limit_store.setdefault(session_id, deque())

    # Drop timestamps older than the sliding window
    while window:
        oldest = window[0]
        if now - oldest <= _WINDOW:
            break
        window.popleft()

    if len(window) >= rpm:
        retry_after = int(_WINDOW - (now - window[0])) + 1
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=(
                f"Rate limit exceeded: max {rpm} requests "
                f"per {int(_WINDOW)}s per session."
            ),
            headers={"Retry-After": str(retry_after)},
        )

    window.append(now)


# ---------------------------------------------------------------------------
# Lifespan
# ---------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Handle startup and shutdown events."""
    # Startup
    logger.info("Starting AI Chatbot Backend...")

    # Initialize PostgreSQL client
    pg_client = PostgreSQLClient()

    # Connect to PostgreSQL — retry up to 3 times to tolerate brief
    # unavailability during rolling deployments or container startup.
    for attempt in range(1, 4):
        try:
            await pg_client.connect()
            logger.info("PostgreSQL connected")
            break
        except Exception as exc:
            if attempt == 3:
                logger.error(
                    "PostgreSQL connection failed after 3 attempts"
                )
                raise
            wait = attempt * 2  # 2s, 4s
            logger.warning(
                f"PostgreSQL connection attempt {attempt} failed "
                f"({exc}); retrying in {wait}s…"
            )
            await asyncio.sleep(wait)

    # Create LangGraph checkpointer based on environment:
    # - True  (default): AsyncPostgresSaver — persistent sessions, production-ready
    # - False: InMemorySaver — no DB needed, sessions lost on restart (local dev)
    if settings.use_postgres_checkpointer:
        async with AsyncPostgresSaver.from_conn_string(
            settings.postgresql_url
        ) as checkpointer:
            await checkpointer.setup()
            logger.info("Using Postgres checkpointer")

            workflow = ConversationWorkflow(pg_client=pg_client)
            conversation_app = workflow.compile(
                checkpointer=checkpointer
            )
            logger.info("LangGraph workflow compiled")

            app.state.pg_client = pg_client
            app.state.conversation_app = conversation_app

            yield

            # Shutdown
            logger.info("Shutting down AI Chatbot Backend...")
            await pg_client.close()
            logger.info("PostgreSQL connection closed")
    else:
        # Development: InMemory — no DB dependency, sessions lost on restart
        checkpointer = InMemorySaver()
        logger.info(
            "Using InMemory checkpointer "
            "(dev mode - no persistence)"
        )

        workflow = ConversationWorkflow(pg_client=pg_client)
        conversation_app = workflow.compile(
            checkpointer=checkpointer
        )
        logger.info("LangGraph workflow compiled")

        app.state.pg_client = pg_client
        app.state.conversation_app = conversation_app

        yield

        # Shutdown
        logger.info("Shutting down AI Chatbot Backend...")
        await pg_client.close()
        logger.info("PostgreSQL connection closed")


# Create FastAPI app
app = FastAPI(
    title="AI Chatbot Backend",
    description="Conversational AI for structured data collection with duplicate detection",
    version="1.0.0",
    lifespan=lifespan,
)

# Add CORS middleware
# Origins are configured in app_config.yaml — set to your frontend's URL in production.
app.add_middleware(
    CORSMiddleware,
    allow_origins=app_config.cors_origins,
    allow_credentials=False,
    allow_methods=["POST", "GET"],
    allow_headers=["Content-Type", "X-API-Key"],
)


# API Key Authentication Dependency
async def verify_api_key(x_api_key: str = Header(..., alias="X-API-Key")) -> None:
    """
    Verify API key from header.

    This is a FastAPI dependency that can be injected into endpoints.
    Raises HTTPException if the API key is invalid.
    """
    if not secrets.compare_digest(x_api_key, settings.api_key):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key",
        )


# Endpoints
@app.post("/chat", response_model=ChatResponse, tags=["Chat"])
async def chat(
    request: ChatRequest,
    _auth: None = Depends(verify_api_key),
    _rate: None = Depends(check_rate_limit),
) -> ChatResponse:
    """
    Send a message to the chatbot and get a response.

    The chatbot maintains conversation state using the session_id and drives
    a multi-turn LangGraph workflow:
    1. Collecting request details through natural conversation
    2. Extracting structured data when all fields are ready
    3. Checking for duplicate requests (fuzzy → vector → LLM judge)
    4. Saving the request to PostgreSQL on confirmation

    When the request is successfully saved, is_complete will be True.

    If duplicates are found the bot pauses and asks the user to choose:
    'proceed', 'modify', or 'cancel'.
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
            request_active=result.get("request_active", True),
            collected_data=result.get("collected_data"),
            duplicate_warning=result.get("duplicate_warning"),
        )

    except BadRequestError as e:
        if e.status_code != 400:
            logger.error(f"Error in chat endpoint: {e}", exc_info=True)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to process chat message: {str(e)}",
            ) from e
        logger.warning(
            "LLM request blocked by provider content policy: %s",
            e,
        )
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                "⚠️ I can't help with that request. Please rephrase "
                "it as a legitimate engineering service desk request "
                "and try again."
            ),
        ) from e
    except Exception as e:
        logger.error(f"Error in chat endpoint: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to process chat message: {str(e)}",
        ) from e


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
        state = await app.state.conversation_app.aget_state(
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
        ) from e


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check() -> HealthResponse:
    """
    Health check endpoint for Docker and monitoring.

    Checks connectivity to PostgreSQL.
    """
    postgresql_status = "disconnected"

    try:
        await app.state.pg_client.ping()
        postgresql_status = "connected"
    except Exception as e:
        logger.error(f"PostgreSQL health check failed: {e}")

    overall_status = (
        "healthy" if postgresql_status == "connected" else "unhealthy"
    )

    return HealthResponse(
        status=overall_status,
        postgresql=postgresql_status,
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

