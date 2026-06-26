"""Internal LLM prompt constants for workflow sub-tasks.

These are implementation-level prompts used by specific nodes in the
workflow — distinct from the user-facing system_prompt in prompt_config.yaml.
"""


DUPLICATE_JUDGE_PROMPT = (
    "You are a duplicate-request detector. "
    "Compare the two requests below and decide if they "
    "represent the same intent — same purpose, same "
    "target, same scope. "
    "If any meaningful field differs (e.g. different "
    "environment, system, date, or access level) they "
    "are NOT duplicates."
)

DUPLICATE_DECISION_PARSER_PROMPT = (
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
