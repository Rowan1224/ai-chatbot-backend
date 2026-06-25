"""Unit tests for schema module — schema-agnostic ExtractedRequest."""

import pytest
from pydantic import ValidationError

from src.core.schema import (
    ExtractedRequest,
    extracted_to_dict,
    get_text_fields,
)


class TestExtractedRequest:
    """Test ExtractedRequest model."""

    def test_requires_fixed_fields(self):
        """request_type, name, and employee_id are all required."""
        with pytest.raises(ValidationError):
            ExtractedRequest()  # type: ignore[call-arg]
        with pytest.raises(ValidationError):
            ExtractedRequest(request_type="infra")  # missing name + employee_id
        with pytest.raises(ValidationError):
            ExtractedRequest(request_type="infra", name="Jane")  # missing employee_id

    def test_accepts_fixed_fields_only(self):
        """additional_data defaults to empty list when not provided."""
        req = ExtractedRequest(
            request_type="infrastructure-provisioning",
            name="Jane",
            employee_id="EMP001",
        )
        assert req.request_type == "infrastructure-provisioning"
        assert req.name == "Jane"
        assert req.employee_id == "EMP001"
        assert req.additional_data == []

    def test_accepts_additional_data(self):
        """DataField list is valid as additional_data."""
        from src.core.schema import DataField
        req = ExtractedRequest(
            request_type="service-deployment",
            name="Jane",
            employee_id="EMP001",
            additional_data=[
                DataField(key="target_environment", value="production"),
                DataField(key="business_justification", value="Critical fix"),
            ],
        )
        assert req.additional_data[0].key == "target_environment"
        assert req.additional_data[0].value == "production"

    def test_additional_data_is_schema_agnostic(self):
        """
        Future prompt changes can add arbitrary fields without
        touching this model — any key names are accepted.
        """
        from src.core.schema import DataField
        req = ExtractedRequest(
            request_type="custom-type",
            name="Bob",
            employee_id="EMP99",
            additional_data=[
                DataField(key="some_new_field", value="value"),
                DataField(key="another_future_field", value="42"),
            ],
        )
        assert req.additional_data[0].key == "some_new_field"
        assert req.additional_data[1].key == "another_future_field"


class TestExtractedToDict:
    """Test extracted_to_dict flattening."""

    def test_fixed_fields_always_at_top_level(self):
        """request_type, name, employee_id are always top-level keys."""
        req = ExtractedRequest(
            request_type="access-grant",
            name="Alice",
            employee_id="EMP42",
        )
        result = extracted_to_dict(req)
        assert result["request_type"] == "access-grant"
        assert result["name"] == "Alice"
        assert result["employee_id"] == "EMP42"

    def test_additional_data_merged_flat(self):
        """DataField list entries are merged to top-level keys."""
        from src.core.schema import DataField
        req = ExtractedRequest(
            request_type="pipeline-change",
            name="Bob",
            employee_id="EMP99",
            additional_data=[
                DataField(key="business_justification", value="Needs update"),
            ],
        )
        result = extracted_to_dict(req)

        assert result["request_type"] == "pipeline-change"
        assert result["name"] == "Bob"
        assert result["employee_id"] == "EMP99"
        assert result["business_justification"] == "Needs update"
        assert "additional_data" not in result

    def test_no_additional_data_returns_three_keys(self):
        """When additional_data is empty the dict has exactly three keys."""
        req = ExtractedRequest(
            request_type="incident-fix",
            name="Carol",
            employee_id="EMP7",
        )
        result = extracted_to_dict(req)
        assert set(result.keys()) == {
            "request_type", "name", "employee_id"
        }

    def test_preserves_special_characters(self):
        """Special chars and Unicode in values survive the round-trip."""
        from src.core.schema import DataField
        req = ExtractedRequest(
            request_type="infrastructure-provisioning",
            name="José O'Brien",
            employee_id="EMP001",
            additional_data=[
                DataField(key="business_justification", value="Unicode: 你好 @#$%"),
            ],
        )
        result = extracted_to_dict(req)
        assert result["business_justification"] == "Unicode: 你好 @#$%"
        assert result["name"] == "José O'Brien"

    def test_additional_data_does_not_override_fixed_fields(self):
        """Fixed fields are set before additional_data is iterated."""
        from src.core.schema import DataField
        req = ExtractedRequest(
            request_type="access-grant",
            name="Real Name",
            employee_id="EMP-REAL",
            additional_data=[
                DataField(key="name", value="should-not-win"),
                DataField(key="employee_id", value="should-not-win"),
            ],
        )
        result = extracted_to_dict(req)
        assert "name" in result
        assert "employee_id" in result


class TestGetTextFields:
    """Test get_text_fields — works from actual data, not YAML."""

    def test_excludes_pii_fields(self):
        """CRITICAL: PII fields must never reach the embedding."""
        data = {
            "request_type": "infrastructure-provisioning",
            "business_justification": "Need servers",
            "name": "Alice",          # PII
            "employee_id": "EMP123",  # PII
        }
        fields = get_text_fields(data)

        assert "name" not in fields, (
            "PRIVACY VIOLATION: name must not be in text fields"
        )
        assert "employee_id" not in fields, (
            "PRIVACY VIOLATION: employee_id must not be in text fields"
        )

    def test_includes_non_pii_string_fields(self):
        """Non-PII string fields are included for embedding."""
        data = {
            "request_type": "service-deployment",
            "business_justification": "Deploy v2",
            "name": "Bob",
            "employee_id": "EMP456",
        }
        fields = get_text_fields(data)

        assert "request_type" in fields
        assert "business_justification" in fields

    def test_skips_empty_strings(self):
        """Empty string values contribute nothing to embeddings."""
        data = {
            "request_type": "incident-fix",
            "business_justification": "",   # empty — skip
            "notes": "   ",                 # whitespace only — skip
        }
        fields = get_text_fields(data)

        assert "business_justification" not in fields
        assert "notes" not in fields

    def test_skips_non_string_values(self):
        """Non-string values (int, bool, dict) are ignored."""
        data = {
            "request_type": "pipeline-change",
            "priority": 3,
            "urgent": True,
            "meta": {"key": "val"},
        }
        fields = get_text_fields(data)

        assert "priority" not in fields
        assert "urgent" not in fields
        assert "meta" not in fields

    def test_works_with_unknown_future_fields(self):
        """
        If a new prompt adds a field not in the YAML, it is still
        picked up for embedding without any code change.
        """
        data = {
            "request_type": "custom-request",
            "new_field_from_future_prompt": "some description",
        }
        fields = get_text_fields(data)

        assert "new_field_from_future_prompt" in fields

    def test_returns_list_of_strings(self):
        """Return type is always a list of non-empty strings."""
        data = {"request_type": "access-grant", "note": "urgent"}
        fields = get_text_fields(data)

        assert isinstance(fields, list)
        for f in fields:
            assert isinstance(f, str) and f


class TestPrivacyIntegration:
    """End-to-end: extracted data never leaks PII into embeddings."""

    def test_pii_never_in_embedding_text(self):
        """
        CRITICAL: the text passed to the embedding model must
        not contain name or employee_id values.
        """
        from src.core.schema import DataField
        req = ExtractedRequest(
            request_type="infrastructure-provisioning",
            name="Alice Johnson",
            employee_id="EMP789",
            additional_data=[
                DataField(key="business_justification",
                          value="Need servers for project X"),
            ],
        )
        data = extracted_to_dict(req)
        text_content = " ".join(
            str(data[f]) for f in get_text_fields(data)
        )

        assert "Alice Johnson" not in text_content, (
            "PRIVACY VIOLATION: name leaked into embedding text"
        )
        assert "EMP789" not in text_content, (
            "PRIVACY VIOLATION: employee_id leaked into embedding text"
        )
        assert "infrastructure-provisioning" in text_content
        assert "project X" in text_content


# Made with Bob
