"""LangGraph workflow for conversational data collection."""

import logging
import operator
from typing import Annotated, Any, Dict, List, Optional, TypedDict

from langchain_core.messages import AIMessage, BaseMessage, SystemMessage
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, StateGraph
from pydantic import BaseModel, Field

from src.config.settings import prompt_config, settings
from src.core.database import (
    MongoDBClient,
    RedisClient,
    find_exact_match_duplicates,
    find_similar_requests,
    save_request,
    update_request,
)
from src.core.llm import get_embedding_model, get_llm
from src.core.schema import RequestSchema, get_text_fields, schema_to_dict

logger = logging.getLogger(__name__)


class ChatResponse(BaseModel):
    """Structured response from chat node."""

    response: str = Field(description="Bot's response message to the user")
    is_ready: bool = Field(
        description="True when all required information has been collected and ready to extract"
    )


class DuplicateDecision(BaseModel):
    """Structured response for duplicate decision."""

    choice: str = Field(
        description="User's choice: either 'update' to update existing request or 'new' to create new request"
    )
    reasoning: str = Field(
        description="Brief explanation of why the user made this choice"
    )


class ConversationState(TypedDict):
    """State for the conversation workflow."""

    messages: Annotated[List[BaseMessage], operator.add]
    collected_data: Dict[str, Any]
    is_ready: bool
    is_complete: bool
    duplicate_warning: List[Dict[str, Any]]
    config_version: str
    update_existing: bool  # True if user wants to update existing request
    existing_request_id: Optional[str]  # ID of request to update
    awaiting_duplicate_decision: bool  # True when waiting for user to choose update/new


class ConversationWorkflow:
    """LangGraph workflow for conversational data collection."""

    def __init__(self, mongodb_client: MongoDBClient) -> None:
        """Initialize the workflow with database clients."""
        self.mongodb_client = mongodb_client
        
        self.llm = get_llm()
        self.embedding_model = get_embedding_model()
        
        # Create structured LLMs once at initialization (performance optimization)
        self.llm_chat = self.llm.with_structured_output(ChatResponse)
        self.llm_extract = self.llm.with_structured_output(RequestSchema)
        self.llm_duplicate_decision = self.llm.with_structured_output(DuplicateDecision)
        
        self.app = None

    async def chat_node(self, state: ConversationState) -> Dict[str, Any]:
        """
        Continue conversation with structured output.
        
        LLM decides when all information is collected via is_ready flag.
        No extraction happens here - just conversation.
        """
        logger.info("Chat node: Processing user message")
        
        # Add system prompt if not present
        messages = state["messages"]
        if not any(isinstance(msg, SystemMessage) for msg in messages):
            messages = [SystemMessage(content=prompt_config.system_prompt)] + messages
        
        # Get structured chat response (using pre-created structured LLM)
        result = self.llm_chat.invoke(messages)
        
        logger.info(f"Chat response: is_ready={result.is_ready}")
        
        return {
            "messages": [AIMessage(content=result.response)],
            "is_ready": result.is_ready,
        }

    async def extract_node(self, state: ConversationState) -> Dict[str, Any]:
        """
        Extract structured data from conversation including ALL fields (business + PII).
        
        Only called when is_ready=True.
        Uses RequestSchema (dynamically generated from config).
        LLM extracts everything in one go - simple and straightforward.
        """
        logger.info("Extract node: Extracting structured data")
        
        messages = state["messages"]
        
        # Add system prompt if needed
        if not any(isinstance(msg, SystemMessage) for msg in messages):
            messages = [SystemMessage(content=prompt_config.system_prompt)] + messages
        
        try:
            # Extract structured data (using pre-created structured LLM)
            extracted = self.llm_extract.invoke(messages)
            data = schema_to_dict(extracted)
            
            logger.info(f"Successfully extracted data with {len(data)} fields")
            
            return {
                "collected_data": data,
                "is_complete": True,  # Complete after extraction
                "config_version": prompt_config.config_version,
            }
            
        except Exception as e:
            logger.error(f"Extraction failed: {e}")
            
            # Extraction failed - ask for clarification
            error_msg = AIMessage(
                content="I need some clarification on the information provided. "
                       "Could you please provide more details?"
            )
            
            return {
                "messages": [error_msg],
                "is_ready": False,
                "is_complete": False,
            }

    async def duplicate_check_node(self, state: ConversationState) -> Dict[str, Any]:
        """
        Check for duplicate requests using two-stage detection:
        1. Exact match on non-PII fields (fast, accurate)
        2. Semantic similarity if no exact match and enabled (slower, fuzzy)
        
        Only checks text fields to preserve privacy (no PII in embeddings).
        Returns duplicate warnings for human review.
        """
        if not settings.duplicate_detection_enabled:
            logger.info("Duplicate detection disabled")
            return {}
        
        logger.info("Duplicate check node: Checking for duplicate requests")
        
        collected_data = state["collected_data"]
        
        # STAGE 1: Try exact match first (fast, no embeddings needed)
        logger.info("Stage 1: Checking for exact match duplicates...")
        exact_matches = await find_exact_match_duplicates(
            mongodb_client=self.mongodb_client,
            data=collected_data,
            lookback_days=settings.lookback_days,
        )
        
        if exact_matches:
            logger.info(f"Found {len(exact_matches)} exact match duplicate(s)")
            similar = exact_matches
        else:
            # STAGE 2: No exact match, try semantic similarity if enabled
            if settings.semantic_search_enabled:
                logger.info("Stage 2: No exact matches, checking semantic similarity...")
                
                # Extract text fields for embedding
                text_fields = get_text_fields()
                text_content = " ".join([
                    str(collected_data.get(field, ""))
                    for field in text_fields
                ])
                
                if not text_content.strip():
                    logger.warning("No text content for embedding")
                    return {}
                
                # Generate embedding
                embedding_result = await self.embedding_model.aembed_query(text_content)
                
                # Find similar requests
                similar = await find_similar_requests(
                    mongodb_client=self.mongodb_client,
                    embedding=embedding_result,
                    threshold=settings.similarity_threshold,
                    lookback_days=settings.lookback_days,
                )
            else:
                logger.info("Stage 2: Semantic search disabled, skipping")
                similar = []
        
        if similar:
            match_type = "exact match" if exact_matches else "similar"
            logger.info(f"Found {len(similar)} {match_type} request(s)")
            
            # Format duplicate warning (without PII)
            warnings = []
            for req in similar:
                warnings.append({
                    "id": str(req.get("_id")),
                    "similarity_score": req.get("similarity_score", 0),
                    "is_exact_match": req.get("is_exact_match", False),
                    "created_at": str(req.get("created_at")),
                    "config_version": req.get("config_version"),
                    # Only include non-PII fields
                    "request_type": req.get("data", {}).get("request_type"),
                    "target_environment": req.get("data", {}).get("target_environment"),
                })
            
            # Add message about duplicates
            match_description = "exact duplicate(s)" if exact_matches else "similar request(s)"
            duplicate_msg = AIMessage(
                content=f"⚠️ I found {len(similar)} {match_description} that may be duplicates:\n\n"
                       f"Would you like to:\n"
                       f"1. Update the existing request\n"
                       f"2. Create a new request anyway\n\n"
                       f"Please respond with 'update' or 'new'."
            )
            
            return {
                "duplicate_warning": warnings,
                "messages": [duplicate_msg],
                "awaiting_duplicate_decision": True,  # Set flag to wait for decision
                "is_complete": False,  # Keep session open for user decision
            }
        
        logger.info("No duplicates found (exact or semantic)")
        return {}

    async def handle_duplicate_decision_node(self, state: ConversationState) -> Dict[str, Any]:
        """
        Handle user's decision on duplicate: update existing or create new.
        
        Uses LLM with structured output for robust parsing.
        """
        logger.info("Handle duplicate decision node: Processing user choice")
        
        # Create prompt for LLM to parse user's decision
        system_prompt = SystemMessage(
            content="You are parsing a user's decision about duplicate requests. "
                   "The user should respond with 'update' to update an existing request "
                   "or 'new' to create a new request. Extract their choice."
        )
        
        messages = [system_prompt] + state["messages"]
        
        try:
            # Use structured LLM to parse decision
            result = await self.llm_duplicate_decision.ainvoke(messages)
            
            choice = result.choice.lower().strip()
            logger.info(f"Parsed duplicate decision: {choice}")
            
            if choice == "update":
                # User wants to update existing request
                logger.info("User chose to update existing request")
                
                # Get the first duplicate's ID (we'll update the most similar one)
                duplicate_warning = state.get("duplicate_warning", [])
                if duplicate_warning:
                    existing_id = duplicate_warning[0]["id"]
                    logger.info(f"Will update request ID: {existing_id}")
                    
                    return {
                        "update_existing": True,
                        "existing_request_id": existing_id,
                        "awaiting_duplicate_decision": False,
                        "messages": [AIMessage(
                            content=f"✅ I'll update your existing request with the new information.\n"
                                   f"Reason: {result.reasoning}"
                        )]
                    }
                else:
                    logger.error("No duplicate warning found in state")
                    return {
                        "update_existing": False,
                        "awaiting_duplicate_decision": False,
                        "messages": [AIMessage(
                            content="❌ Error: Could not find the duplicate request. Creating a new request instead."
                        )]
                    }
            
            elif choice == "new":
                # User wants to create new request
                logger.info("User chose to create new request")
                
                return {
                    "update_existing": False,
                    "awaiting_duplicate_decision": False,
                    "messages": [AIMessage(
                        content=f"✅ I'll create a new request for you.\n"
                               f"Reason: {result.reasoning}"
                    )]
                }
            
            else:
                # Invalid choice from LLM
                logger.warning(f"LLM returned invalid choice: {choice}")
                return {
                    "messages": [AIMessage(
                        content="❌ I couldn't understand your choice. Please respond with 'update' or 'new'."
                    )]
                }
        
        except Exception as e:
            logger.error(f"Error parsing duplicate decision: {e}")
            return {
                "messages": [AIMessage(
                    content="❌ Error processing your choice. Please respond with 'update' or 'new'."
                )]
            }

    async def save_node(self, state: ConversationState) -> Dict[str, Any]:
        """
        Save or update the collected data in MongoDB.
        
        All data (including PII) is already in collected_data from extract_node.
        If update_existing=True, updates the existing request.
        Otherwise, creates a new request.
        """
        collected_data = state["collected_data"].copy()  # Make a copy to avoid mutating state
        update_existing = state.get("update_existing", False)
        existing_request_id = state.get("existing_request_id")
        
        # Generate embedding for future duplicate detection (excluding PII)
        text_fields = get_text_fields()
        text_content = " ".join([
            str(collected_data.get(field, ""))
            for field in text_fields
        ])
        
        embedding = await self.embedding_model.aembed_query(text_content)
        
        if update_existing and existing_request_id:
            # Update existing request
            logger.info(f"Save node: Updating existing request {existing_request_id}")
            
            success = await update_request(
                self.mongodb_client, existing_request_id, collected_data, embedding
            )
            
            if success:
                success_msg = AIMessage(
                    content=f"✅ Your existing request has been successfully updated!\n"
                           f"Request ID: {existing_request_id}\n\n"
                           f"Thank you!"
                )
            else:
                success_msg = AIMessage(
                    content="❌ Failed to update the existing request. "
                           "Creating a new request instead..."
                )
                # Fallback to creating new request
                request_id = await save_request(self.mongodb_client, collected_data, embedding)
                success_msg = AIMessage(
                    content=f"✅ New request created successfully!\n"
                           f"Request ID: {request_id}\n\n"
                           f"Thank you!"
                )
        else:
            # Create new request
            logger.info("Save node: Creating new request")
            
            request_id = await save_request(self.mongodb_client, collected_data, embedding)
            
            logger.info(f"Request saved with ID: {request_id}")
            
            success_msg = AIMessage(
                content=f"✅ Your request has been successfully submitted!\n"
                       f"Request ID: {request_id}\n\n"
                       f"Thank you!"
            )
        
        return {
            "messages": [success_msg],
            "is_complete": True  # Mark conversation as complete after successful save
        }

    def should_extract(self, state: ConversationState) -> str:
        """Routing: Extract if ready, otherwise end and wait for next message."""
        if state.get("is_ready", False):
            return "extract"
        return END

    def should_save(self, state: ConversationState) -> str:
        """
        Routing: Determine if we should save immediately or wait for user decision.
        
        If duplicates found and awaiting decision, END and wait.
        If decision made (awaiting_duplicate_decision=False), proceed to save.
        If no duplicates, proceed to save.
        """
        if state.get("duplicate_warning") and state.get("awaiting_duplicate_decision"):
            # Duplicates found and waiting for user decision
            return END
        # No duplicates or decision already made - proceed to save
        return "save"
    
    def after_duplicate_decision(self, state: ConversationState) -> str:
        """
        Routing after duplicate decision is made.
        
        If decision was invalid (still awaiting), END and wait for retry.
        Otherwise, proceed to save.
        """
        if state.get("awaiting_duplicate_decision"):
            # Invalid decision, still waiting
            return END
        # Valid decision made, proceed to save
        return "save"

    def route_entry(self, state: ConversationState) -> str:
        """
        Entry point routing.
        
        Routes to appropriate node based on state.
        """
        # If we're waiting for duplicate decision, route to handler
        if state.get("awaiting_duplicate_decision"):
            return "handle_duplicate_decision"
        
        # Otherwise, normal chat flow
        return "chat"

    def build_graph(self) -> StateGraph:
        """
        Build the LangGraph workflow.
        
        Simplified flow:
        1. chat -> Continue conversation (structured output with is_ready flag)
        2. extract -> Extract ALL data including PII when ready
        3. duplicate_check -> Find similar requests
        4. handle_duplicate_decision -> Parse user's choice (update/new)
        5. save -> Persist to MongoDB (create new or update existing)
        
        Human intervention points:
        - After each chat message (API returns, waits for next user input)
        - After duplicate_check (if duplicates found, user must choose update/new)
        - After handle_duplicate_decision (if invalid choice, wait for retry)
        """
        workflow = StateGraph(ConversationState)
        
        # Add nodes
        workflow.add_node("chat", self.chat_node)
        workflow.add_node("extract", self.extract_node)
        workflow.add_node("duplicate_check", self.duplicate_check_node)
        workflow.add_node("handle_duplicate_decision", self.handle_duplicate_decision_node)
        workflow.add_node("save", self.save_node)
        
        # Set entry point with conditional routing
        workflow.set_conditional_entry_point(
            self.route_entry,
            {
                "chat": "chat",
                "handle_duplicate_decision": "handle_duplicate_decision",
            }
        )
        
        # Routing from chat
        workflow.add_conditional_edges(
            "chat",
            self.should_extract,
            {
                "extract": "extract",
                END: END,  # Wait for next user message
            }
        )
        
        # Routing from extract -> duplicate_check
        workflow.add_edge("extract", "duplicate_check")
        
        # Routing from duplicate_check
        workflow.add_conditional_edges(
            "duplicate_check",
            self.should_save,
            {
                "save": "save",  # No duplicates, proceed to save
                END: END,  # Duplicates found, wait for user decision
            }
        )
        
        # Routing from handle_duplicate_decision
        workflow.add_conditional_edges(
            "handle_duplicate_decision",
            self.after_duplicate_decision,
            {
                "save": "save",  # Valid decision made, proceed to save
                END: END,  # Invalid decision, wait for retry
            }
        )
        
        # After save, end
        workflow.add_edge("save", END)
        
        return workflow

    def compile(self, checkpointer: BaseCheckpointSaver):
        """
        Compile the workflow with the provided checkpointer.
        
        Args:
            checkpointer: LangGraph checkpointer (e.g., RedisSaver, InMemorySaver)
        
        Returns:
            Compiled LangGraph application
        """
        # Build and compile workflow
        graph = self.build_graph()
        self.app = graph.compile(checkpointer=checkpointer)
        
        logger.info(f"LangGraph workflow compiled with {type(checkpointer).__name__}")
        
        return self.app

    def get_app(self):
        """
        Get the compiled application.
        
        Raises:
            RuntimeError: If app hasn't been compiled yet
        """
        if not self.app:
            raise RuntimeError(
                "Workflow not compiled. Call compile(checkpointer) first."
            )
        return self.app


# Made with Bob
