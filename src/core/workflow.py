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
    PostgreSQLClient,
    RedisClient,
    find_fuzzy_candidates,
    find_similar_requests,
    save_request,
)
from src.core.llm import get_embedding_model, get_llm
from src.core.schema import (
    ExtractedRequest,
    extracted_to_dict,
    get_text_fields,
)

logger = logging.getLogger(__name__)


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


class ConversationWorkflow:
    """LangGraph workflow for conversational data collection."""

    def __init__(self, pg_client: PostgreSQLClient) -> None:
        """Initialize the workflow with database clients."""
        self.pg_client = pg_client
        
        self.llm = get_llm()
        self.embedding_model = get_embedding_model()
        
        # Create structured LLMs once at initialization (performance optimization)
        self.llm_chat = self.llm.with_structured_output(ChatResponse)
        self.llm_extract = self.llm.with_structured_output(ExtractedRequest)
        self.llm_duplicate_judge = self.llm.with_structured_output(DuplicateJudgement)
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
        Extract structured data from the conversation.

        Uses ``ExtractedRequest`` — a schema-agnostic model with
        only ``request_type`` fixed and everything else captured
        in ``additional_data``.  This means prompt changes never
        require a code or database migration; historical records
        remain intact with whatever structure they were saved with.
        """
        logger.info("Extract node: Extracting structured data")

        messages = state["messages"]

        if not any(isinstance(msg, SystemMessage) for msg in messages):
            messages = [SystemMessage(content=prompt_config.system_prompt)] + messages

        try:
            extracted = self.llm_extract.invoke(messages)
            # Flatten into a single dict: request_type + everything else
            data = extracted_to_dict(extracted)

            logger.info(
                f"Extracted data with {len(data)} field(s): "
                f"{list(data.keys())}"
            )

            return {
                "collected_data": data,
                "is_complete": True,
                "config_version": prompt_config.config_version,
            }

        except Exception as e:
            logger.error(f"Extraction failed: {e}")
            # Keep is_ready=True so the next user message re-triggers
            # extract_node — the conversation history still has all
            # the data, no need to collect it again.
            return {
                "messages": [AIMessage(
                    content="⚠️ I had trouble reading your details. "
                            "Please send any message and I'll try again."
                )],
                "is_ready": True,
                "is_complete": False,
            }

    async def duplicate_check_node(self, state: ConversationState) -> Dict[str, Any]:
        """
        Two-stage duplicate detection: fuzzy pre-filter → vector search.

        Stage 1 — Fuzzy pre-filter (pg_trgm)
            Match request_type and target_environment with trigram
            similarity to build a candidate set. Catches typos /
            minor variations without scanning the full table.
            Returns only row IDs (no PII).

        Stage 2 — Vector search on candidates only
            Run cosine similarity exclusively against the candidate
            IDs from Stage 1, so unrelated request types are never
            compared. If the candidate set is empty, Stage 2 is
            skipped entirely — no duplicates are possible.

        Stage 3 — LLM semantic judge
            For each candidate that passed the vector threshold, ask
            the LLM whether it is a *true* duplicate of the new request
            by comparing non-PII fields side-by-side.  This eliminates
            false positives where vector similarity is high but a
            meaningful field differs (e.g. development vs production).
            Only records the LLM confirms as duplicates reach the user.
        """
        if not settings.duplicate_detection_enabled:
            logger.info("Duplicate detection disabled")
            return {}

        collected_data = state.get("collected_data")
        if not collected_data:
            logger.warning(
                "duplicate_check_node: no collected_data in state"
            )
            return {}

        logger.info(
            "Duplicate check: starting fuzzy → vector pipeline"
        )

        request_type = collected_data.get("request_type", "")

        # STAGE 1: fuzzy pre-filter — build candidate set
        logger.info(
            "Stage 1: fuzzy pre-filter on "
            "request_type + target_environment..."
        )
        candidate_ids = await find_fuzzy_candidates(
            pg_client=self.pg_client,
            request_type=request_type,
            lookback_days=settings.lookback_days,
            fuzzy_threshold=settings.fuzzy_threshold,
            limit=settings.candidate_limit,
        )

        if not candidate_ids:
            logger.info(
                "Stage 1: no fuzzy candidates — "
                "skipping vector search"
            )
            return {}

        similar: List[Dict[str, Any]] = []

        # STAGE 2: vector search scoped to candidates
        if settings.semantic_search_enabled:
            logger.info(
                f"Stage 2: vector search over "
                f"{len(candidate_ids)} candidate(s)..."
            )

            text_content = " ".join(
                str(collected_data[f])
                for f in get_text_fields(collected_data)
            )

            if not text_content.strip():
                logger.warning("No text content for embedding")
                return {}

            embedding_result = (
                await self.embedding_model.aembed_query(
                    text_content
                )
            )

            similar = await find_similar_requests(
                pg_client=self.pg_client,
                embedding=embedding_result,
                candidate_ids=candidate_ids,
                threshold=settings.similarity_threshold,
            )
        else:
            logger.info(
                "Stage 2: semantic search disabled, skipping"
            )

        if not similar:
            logger.info("No duplicates found")
            return {}

        # STAGE 3: LLM semantic judge — filter out false positives
        # Only non-PII fields are sent; no privacy concern.
        logger.info(
            f"Stage 3: LLM judge evaluating "
            f"{len(similar)} vector candidate(s)..."
        )
        pii_fields = set(prompt_config.privacy.get("pii_fields", []))
        new_non_pii = {
            k: v for k, v in collected_data.items()
            if k not in pii_fields
        }

        confirmed: List[Dict[str, Any]] = []
        for req in similar:
            candidate_non_pii = {
                k: v
                for k, v in req.get("data", {}).items()
                if k not in pii_fields
            }
            judge_prompt = [
                SystemMessage(content=(
                    "You are a duplicate-request detector. "
                    "Compare the two requests below and decide if they "
                    "represent the same intent — same purpose, same "
                    "target, same scope. "
                    "If any meaningful field differs (e.g. different "
                    "environment, system, date, or access level) they "
                    "are NOT duplicates."
                )),
                AIMessage(content=(
                    f"NEW REQUEST:\n{new_non_pii}\n\n"
                    f"EXISTING REQUEST:\n{candidate_non_pii}"
                )),
            ]
            try:
                verdict = await self.llm_duplicate_judge.ainvoke(
                    judge_prompt
                )
                logger.info(
                    f"Judge verdict for {req.get('_id')}: "
                    f"is_duplicate={verdict.is_duplicate} — "
                    f"{verdict.reasoning}"
                )
                if verdict.is_duplicate:
                    confirmed.append(req)
            except Exception as e:
                logger.warning(
                    f"Judge failed for {req.get('_id')}: {e} — "
                    "treating as non-duplicate to avoid false positive"
                )

        similar = confirmed

        if similar:
            logger.info(
                f"Stage 3: {len(similar)} confirmed duplicate(s) "
                f"after LLM judgement"
            )

            warnings = []
            for req in similar:
                non_pii_data = {
                    k: v
                    for k, v in req.get("data", {}).items()
                    if k not in pii_fields
                }
                warnings.append({
                    "id": str(req.get("_id")),
                    "similarity_score": req.get(
                        "similarity_score", 0
                    ),
                    "created_at": str(req.get("created_at")),
                    "config_version": req.get("config_version"),
                    "data": non_pii_data,
                })

            duplicate_msg = AIMessage(
                content=(
                    f"⚠️ I found {len(similar)} request(s) that "
                    f"look similar to yours.\n\n"
                    f"This is not an exact match — it just means "
                    f"something close was submitted before. "
                    f"If your request is genuinely different "
                    f"(e.g. a different environment, system, or "
                    f"date), that's fine — just choose **proceed**.\n\n"
                    f"What would you like to do?\n"
                    f"• **proceed** — yes, my request is different, "
                    f"save it\n"
                    f"• **modify** — let me change something first\n"
                    f"• **cancel** — abandon this request\n\n"
                    f"Reply with 'proceed', 'modify', or 'cancel'."
                )
            )

            return {
                "duplicate_warning": warnings,
                "messages": [duplicate_msg],
                "awaiting_duplicate_decision": True,
                "is_complete": False,
            }

        logger.info("Stage 3: all candidates cleared by LLM judge — no duplicates")
        return {}

    async def handle_duplicate_decision_node(self, state: ConversationState) -> Dict[str, Any]:
        """
        Handle user's decision when a duplicate is detected.

        Three valid choices:
        - 'modify'  — reset collected data, clear duplicate state, and
                      return to the chat node so the user can tell the
                      bot what they want to change.  The conversation
                      history is preserved so context is not lost.
        - 'proceed' — the request is intentionally different; save it
                      as a brand-new record.
        - 'cancel'  — abandon without saving; end the conversation.

        The old/existing duplicate record is never modified.
        ``duplicate_decision`` is written to state so the router
        ``after_duplicate_decision`` can branch without re-reading the
        LLM output.
        """
        logger.info(
            "Handle duplicate decision: parsing user choice"
        )

        system_prompt = SystemMessage(
            content=(
                "You are parsing a user's response to a duplicate "
                "request warning.  The user should choose one of:\n"
                "  'modify'  — they want to go back to the "
                "conversation and change something in their current "
                "request\n"
                "  'proceed' — their request is intentionally "
                "different and should be saved as a new record\n"
                "  'cancel'  — they want to abandon the request "
                "without saving\n"
                "Extract their choice."
            )
        )

        messages = [system_prompt] + state.get("messages", [])

        try:
            result = await self.llm_duplicate_decision.ainvoke(
                messages
            )
            choice = result.choice.lower().strip()
            logger.info(f"Parsed duplicate decision: {choice}")

            if choice == "modify":
                # Clear extraction state so the user can refine their
                # request from scratch in the chat node.  Conversation
                # history is kept so the bot has context.
                logger.info(
                    "User chose to modify — returning to chat"
                )
                return {
                    "collected_data": {},
                    "is_ready": False,
                    "is_complete": False,
                    "duplicate_warning": [],
                    "awaiting_duplicate_decision": False,
                    "duplicate_decision": "modify",
                    "messages": [AIMessage(
                        content=(
                            "Sure! What would you like to change "
                            "about your request? Let me know and "
                            "I'll update it for you."
                        )
                    )],
                }

            elif choice == "proceed":
                # Save the current request as a new record
                logger.info(
                    "User chose to proceed — saving as new request"
                )
                return {
                    "awaiting_duplicate_decision": False,
                    "duplicate_warning": [],
                    "duplicate_decision": "proceed",
                    "messages": [AIMessage(
                        content=(
                            "✅ Got it — saving your request now."
                        )
                    )],
                }

            elif choice == "cancel":
                # End the conversation without saving anything
                logger.info(
                    "User chose to cancel — ending conversation"
                )
                return {
                    "awaiting_duplicate_decision": False,
                    "is_complete": False,
                    "duplicate_decision": "cancel",
                    "messages": [AIMessage(
                        content=(
                            "Your request has been cancelled. "
                            "Feel free to start a new request "
                            "whenever you're ready."
                        )
                    )],
                }

            else:
                logger.warning(
                    f"LLM returned unrecognised choice: {choice}"
                )
                return {
                    "duplicate_decision": None,
                    "messages": [AIMessage(
                        content=(
                            "❌ I didn't quite catch that. "
                            "Please reply with 'modify', "
                            "'proceed', or 'cancel'."
                        )
                    )],
                }

        except Exception as e:
            logger.error(f"Error parsing duplicate decision: {e}")
            return {
                "duplicate_decision": None,
                "messages": [AIMessage(
                    content=(
                        "❌ Something went wrong. Please reply "
                        "with 'modify', 'proceed', or 'cancel'."
                    )
                )],
            }

    async def save_node(self, state: ConversationState) -> Dict[str, Any]:
        """
        Save the collected data as a new request in PostgreSQL.

        All data (including PII) is already in collected_data from
        extract_node.  Always creates a new record — updating an
        existing duplicate is not supported; that path was removed
        in favour of the modify → chat loop.
        """
        collected_data = state.get("collected_data")
        if not collected_data:
            logger.error("save_node: collected_data missing from state")
            return {
                "messages": [AIMessage(
                    content="⚠️ I had trouble processing your "
                            "request details. Let me try again — "
                            "please send any message to continue."
                )],
                "is_complete": False,
                "is_ready": True,
            }
        collected_data = collected_data.copy()

        # Generate embedding for semantic duplicate detection (non-PII only)
        text_content = " ".join(
            str(collected_data[f])
            for f in get_text_fields(collected_data)
        )

        embedding = await self.embedding_model.aembed_query(text_content)

        logger.info("Save node: Creating new request")
        request_id = await save_request(self.pg_client, collected_data, embedding)
        logger.info(f"Request saved with ID: {request_id}")

        return {
            "messages": [AIMessage(
                content=f"✅ Your request has been successfully submitted!\n"
                        f"Request ID: {request_id}\n\n"
                        f"Thank you!"
            )],
            "is_complete": True,
        }

    def should_extract(self, state: ConversationState) -> str:
        """Routing: Extract if ready, otherwise end and wait for next message."""
        if state.get("is_ready", False):
            return "extract"
        return END

    def should_save(self, state: ConversationState) -> str:
        """
        Routing: Determine if we should save or end the turn.

        Guards:
        - No collected_data → extraction failed; END so the chat
          node can ask the user to clarify (is_ready is already
          False from extract_node's error path).
        - Duplicates found and awaiting decision → END and wait
          for the user to reply with 'update' or 'new'.
        - Otherwise → proceed to save.
        """
        if not state.get("collected_data"):
            # Extraction failed — user already got an error message
            # from extract_node; just end this turn.
            logger.warning(
                "should_save: no collected_data — "
                "skipping save, waiting for user clarification"
            )
            return END
        if state.get("duplicate_warning") and state.get("awaiting_duplicate_decision"):
            return END
        return "save"
    
    def after_duplicate_decision(self, state: ConversationState) -> str:
        """
        Routing after duplicate decision is made.

        - ``modify``  → route back to ``chat`` so the user can refine
                        their current request through conversation.
        - ``proceed`` → route to ``save`` to persist as a new record.
        - ``cancel``  → END the conversation without saving.
        - ``None``    → unrecognised reply; END and wait for a retry.
        """
        decision = state.get("duplicate_decision")
        if decision == "modify":
            return "chat"
        if decision == "proceed":
            return "save"
        # 'cancel' or unrecognised (None) — end the turn
        return END

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

        Flow:
        1. chat             — collect details through conversation
        2. extract          — structure the conversation into data fields
        3. duplicate_check  — two-stage fuzzy → vector similarity search
        4. handle_duplicate_decision — parse user's choice
        5. save / chat / END — branch on decision

        Human intervention points:
        - After every chat turn (API returns; waits for next user input)
        - After duplicate_check finds matches (user must reply)
        - After handle_duplicate_decision if reply is unrecognised or
          'cancel' (END without saving)
        - After 'modify': returns to chat so user can refine their request
        """
        workflow = StateGraph(ConversationState)

        # Add nodes
        workflow.add_node("chat", self.chat_node)
        workflow.add_node("extract", self.extract_node)
        workflow.add_node("duplicate_check", self.duplicate_check_node)
        workflow.add_node("handle_duplicate_decision", self.handle_duplicate_decision_node)
        workflow.add_node("save", self.save_node)

        # Entry point: route to duplicate-decision handler if we are
        # waiting for a reply, otherwise start fresh in chat.
        workflow.set_conditional_entry_point(
            self.route_entry,
            {
                "chat": "chat",
                "handle_duplicate_decision": "handle_duplicate_decision",
            }
        )

        # chat → extract (when ready) or END (wait for next message)
        workflow.add_conditional_edges(
            "chat",
            self.should_extract,
            {
                "extract": "extract",
                END: END,
            }
        )

        # extract always feeds into duplicate_check
        workflow.add_edge("extract", "duplicate_check")

        # duplicate_check → save (no duplicates) or END (await decision)
        workflow.add_conditional_edges(
            "duplicate_check",
            self.should_save,
            {
                "save": "save",
                END: END,
            }
        )

        # handle_duplicate_decision → chat (modify) / save (proceed) / END
        workflow.add_conditional_edges(
            "handle_duplicate_decision",
            self.after_duplicate_decision,
            {
                "chat": "chat",   # modify: back to conversation
                "save": "save",   # proceed: persist as new record
                END: END,         # cancel or unrecognised: end turn
            }
        )

        # save always ends the turn
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
