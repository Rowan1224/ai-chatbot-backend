"""Input guardrails for the conversation workflow.

Runs before every call to the chat node. Two concerns are handled:

1. PII leakage — emails and credit-card numbers in user messages
   are redacted using the same detector functions that back
   ``langchain.agents.middleware.PIIMiddleware``.  We consume the
   detectors directly because our graph is a hand-built StateGraph,
   not a ``create_agent`` runtime, so the middleware hook points
   (``before_model`` / ``after_model``) have no attachment surface.

2. Prompt injection — common jailbreak/override patterns are detected
   and the message is blocked before it reaches the LLM.

Both checks are togglable via ``app_config.yaml`` under
``guardrails:``.  When disabled the node is a no-op so the graph
topology stays the same in all configurations.
"""

import logging
import re
from typing import Any

from langchain.agents.middleware._redaction import (
    apply_strategy,
    detect_credit_card,
    detect_email,
)
from langchain_core.messages import AIMessage, HumanMessage

from src.config.settings import app_config
from src.core.models import ConversationState

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Prompt-injection keyword patterns
# ---------------------------------------------------------------------------

# Covers the most common jailbreak / override surface:
#   - role-override phrasing
#   - instruction-ignore phrasing
#   - "DAN" / "developer mode" jailbreaks
#   - system-prompt exfiltration requests
_INJECTION_PATTERNS: list[re.Pattern[str]] = [
    re.compile(p, re.IGNORECASE)
    for p in [
        r"\bignore\s+(all\s+)?(previous|prior|above)\s+(instructions?|prompts?|rules?)\b",
        r"\byou\s+are\s+now\s+(?!an?\s+AI\s+assistant)",  # "you are now X" (not normal)
        r"\bact\s+as\s+(if\s+you\s+(are|were)\s+)?(?!an?\s+AI\s+assistant)",
        r"\bforget\s+(all\s+)?(your\s+)?(previous\s+)?(instructions?|rules?|context)\b",
        r"\bnew\s+instructions?\s*:",
        r"\bsystem\s*prompt\b",
        r"\brepeat\s+(your\s+)?(system\s+)?(instructions?|prompt|rules?)\b",
        r"\bwhat\s+(are\s+your|is\s+your)\s+(instructions?|system\s+prompt|rules?)\b",
        r"\bDAN\b",          # "Do Anything Now" jailbreak
        r"\bdeveloper\s+mode\b",
        r"\bjailbreak\b",
        r"\bpretend\s+(you\s+)?(are|have\s+no)\s+(restrictions?|rules?|limits?|guidelines?)\b",
    ]
]


def _detect_injection(text: str) -> list[str]:
    """Return a list of matched injection pattern descriptions."""
    return [p.pattern for p in _INJECTION_PATTERNS if p.search(text)]


def _redact_pii(text: str) -> tuple[str, list[str]]:
    """
    Redact emails and credit-card numbers from ``text``.

    Returns ``(redacted_text, list_of_detected_pii_types)``.
    Uses the same detector callables that back PIIMiddleware so
    behaviour is consistent with the rest of the LangChain stack.
    """
    detected: list[str] = []

    email_matches = detect_email(text)
    if email_matches:
        text = apply_strategy(text, email_matches, "redact")
        detected.append("email")

    card_matches = detect_credit_card(text)
    if card_matches:
        text = apply_strategy(text, card_matches, "mask")
        detected.append("credit_card")

    return text, detected


# ---------------------------------------------------------------------------
# Public node
# ---------------------------------------------------------------------------


async def guardrail_node(
    state: ConversationState,
) -> dict[str, Any]:
    """
    LangGraph node — runs input guardrails before the chat node.

    Two checks (each independently togglable via app_config.yaml):

    **PII redaction** — emails are replaced with
    ``[REDACTED_EMAIL]``; credit-card numbers are masked to
    ``****-****-****-XXXX``.  The redacted message is written back
    into state so the LLM never sees the raw PII value.

    **Prompt injection** — if the message matches a known jailbreak
    or instruction-override pattern the node short-circuits:
    it returns an ``AIMessage`` refusing the input and sets
    ``is_ready=False`` so the workflow ends the turn without
    reaching the chat node.

    When both checks pass, the node returns an empty dict
    (or the PII-cleaned messages list) and the graph proceeds
    normally to ``chat``.
    """
    if not app_config.guardrails_enabled:
        return {}

    messages = state.get("messages", [])
    if not messages:
        return {}

    # Only inspect the most-recent HumanMessage — earlier turns have
    # already been checked in previous graph invocations.
    last_human_idx: int | None = None
    for i in range(len(messages) - 1, -1, -1):
        if isinstance(messages[i], HumanMessage):
            last_human_idx = i
            break

    if last_human_idx is None:
        return {}

    user_text: str = str(messages[last_human_idx].content)

    # ------------------------------------------------------------------
    # Check 1 — prompt injection
    # ------------------------------------------------------------------
    if app_config.injection_detection_enabled:
        matched = _detect_injection(user_text)
        if matched:
            logger.warning(
                "Prompt injection attempt detected — "
                "blocking turn. Patterns matched: %s",
                matched,
            )
            return {
                "messages": [AIMessage(
                    content=(
                        "⚠️ I can only help with engineering "
                        "service desk requests. I'm not able to "
                        "change my instructions or reveal internal "
                        "configuration. Please describe your "
                        "request and I'll be happy to help."
                    )
                )],
                "is_ready": False,
                "injection_blocked": True,
            }

    # ------------------------------------------------------------------
    # Check 2 — PII redaction
    # ------------------------------------------------------------------
    if app_config.pii_redaction_enabled:
        cleaned, detected = _redact_pii(user_text)
        if detected:
            logger.info(
                "PII detected and redacted from user message: %s",
                detected,
            )
            # Return only the redacted message; the add_messages
            # reducer will update the original in-place by matching id.
            original = messages[last_human_idx]
            redacted_msg = HumanMessage(
                content=cleaned,
                id=getattr(original, "id", None),
                name=getattr(original, "name", None),
            )
            return {"messages": [redacted_msg]}

    return {}
