"""Schema for LLM extraction — schema-agnostic by design.

Three fields are always guaranteed regardless of the active prompt:
- ``request_type`` — stable anchor for fuzzy matching and routing
- ``name``         — requester's full name (PII, never embedded)
- ``employee_id``  — requester's employee ID (PII, never embedded)

Everything else the LLM extracts lands in ``additional_data`` as
a free-form dict, so changing the prompt YAML never requires a
code or database migration.

Historical records are preserved as-is: their ``data`` JSONB blob
reflects whatever schema was active when they were submitted.
"""

from typing import Any

from pydantic import BaseModel, Field

from src.config.settings import app_config


class DataField(BaseModel):
    """A single key-value pair extracted from the conversation.

    OpenAI structured output requires ``additionalProperties: false``
    on every object — a free-form dict is not allowed.  Encoding
    variable fields as a typed list of key/value pairs satisfies the
    schema constraint while still letting the LLM return any fields
    the active prompt defines.
    """

    key: str = Field(description="Field name (e.g. 'target_environment')")
    value: str = Field(description="Field value as a string")


class ExtractedRequest(BaseModel):
    """
    Structured output from the LLM extraction step.

    Fixed platform fields (always present):
    - ``request_type``: stable anchor for fuzzy search & routing
    - ``name``: requester full name — collected by every prompt
    - ``employee_id``: requester employee ID — collected by every prompt

    Variable prompt fields:
    - ``additional_data``: list of key/value pairs for everything
      else the active prompt asks for.  Keys and values are driven
      by the system prompt, not by this class definition.
    """

    request_type: str = Field(
        description=(
            "The type/category of the request. "
            "Normalise to the closest known value as guided "
            "by the system prompt."
        )
    )
    name: str = Field(
        description="Requester's full name."
    )
    employee_id: str = Field(
        description="Requester's employee ID (e.g. EMP12345)."
    )
    additional_data: list[DataField] = Field(
        default_factory=list,
        description=(
            "All other information collected during the "
            "conversation (e.g. justification, environment, "
            "priority), encoded as key/value pairs. "
            "Include one entry per additional field."
        ),
    )


def extracted_to_dict(extracted: ExtractedRequest) -> dict[str, Any]:
    """
    Flatten ``ExtractedRequest`` into a single dict for storage.

    Fixed fields (``request_type``, ``name``, ``employee_id``) are
    always present at the top level.  Fields from ``additional_data``
    are merged in at the same level so the stored structure matches
    what the active prompt defined, regardless of schema version.

    Example:
        ExtractedRequest(
            request_type="infrastructure-provisioning",
            name="Jane",
            employee_id="EMP99",
            additional_data=[
                DataField(key="target_environment", value="production"),
                DataField(key="business_justification", value="..."),
            ]
        )
        → {
            "request_type": "infrastructure-provisioning",
            "name": "Jane",
            "employee_id": "EMP99",
            "target_environment": "production",
            "business_justification": "...",
          }
    """
    data: dict[str, Any] = {
        "request_type": extracted.request_type,
        "name": extracted.name,
        "employee_id": extracted.employee_id,
    }
    for field in extracted.additional_data:
        data[field.key] = field.value
    return data


def get_text_fields(data: dict[str, Any]) -> list[str]:
    """
    Return the non-PII text field names present in ``data``.

    Works from the *actual extracted data keys* rather than a
    static YAML field list, so it stays correct across schema
    versions without any code change.

    Args:
        data: The flat request dict (output of extracted_to_dict)

    Returns:
        List of key names whose values are non-empty strings
        and are not in the configured PII field list.
    """
    pii_fields = set(app_config.pii_fields)
    return [
        k for k, v in data.items()
        if k not in pii_fields and isinstance(v, str) and v.strip()
    ]


# Made with Bob
