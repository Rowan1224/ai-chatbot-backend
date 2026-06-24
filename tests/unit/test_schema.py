"""Unit tests for schema module - Fixed to match actual implementation."""

import pytest
from pydantic import ValidationError

from src.core.schema import (
    RequestSchema,
    get_text_fields,
    schema_to_dict,
)


class TestRequestSchema:
    """Test RequestSchema dynamic model generation."""

    def test_schema_creates_valid_instance(self):
        """Test that schema accepts valid data."""
        valid_data = {
            "request_type": "infrastructure-provisioning",
            "target_environment": "development",
            "business_justification": "Need to test new feature",
            "name": "John Doe",
            "employee_id": "EMP12345"
        }
        
        instance = RequestSchema(**valid_data)
        
        # Verify instance was created
        assert instance is not None
        
        # Verify values are accessible (may be Enums)
        # Use .value to get the actual enum value, not the string representation
        assert instance.request_type.value == "infrastructure-provisioning"
        assert instance.target_environment.value == "development"
        assert instance.business_justification == "Need to test new feature"
        assert instance.name == "John Doe"
        assert instance.employee_id == "EMP12345"
    
    def test_schema_rejects_invalid_request_type(self):
        """Test that invalid request_type is rejected - catches enum violations."""
        invalid_data = {
            "request_type": "invalid-type-that-does-not-exist",
            "target_environment": "development",
            "business_justification": "Test",
            "name": "John Doe",
            "employee_id": "EMP123"
        }
        
        with pytest.raises(ValidationError) as exc_info:
            RequestSchema(**invalid_data)
        
        # Verify error mentions the invalid field
        error_str = str(exc_info.value)
        assert "request_type" in error_str.lower()
    
    def test_schema_rejects_invalid_environment(self):
        """Test that invalid target_environment is rejected."""
        invalid_data = {
            "request_type": "infrastructure-provisioning",
            "target_environment": "invalid-environment",
            "business_justification": "Test",
            "name": "John Doe",
            "employee_id": "EMP123"
        }
        
        with pytest.raises(ValidationError) as exc_info:
            RequestSchema(**invalid_data)
        
        error_str = str(exc_info.value)
        assert "target_environment" in error_str.lower()
    
    def test_schema_requires_all_fields(self):
        """Test that missing required fields cause validation error."""
        incomplete_data = {
            "request_type": "infrastructure-provisioning",
            # Missing other required fields
        }
        
        with pytest.raises(ValidationError) as exc_info:
            RequestSchema(**incomplete_data)
        
        # Should mention missing fields
        error_str = str(exc_info.value)
        assert "field required" in error_str.lower() or "missing" in error_str.lower()
    
    def test_schema_accepts_all_valid_request_types(self):
        """Test all valid request types are accepted - catches if enum changes."""
        valid_types = [
            "infrastructure-provisioning",
            "service-deployment",
            "access-grant",
            "pipeline-change",
            "incident-fix"
        ]
        
        for req_type in valid_types:
            data = {
                "request_type": req_type,
                "target_environment": "development",
                "business_justification": "Test",
                "name": "Test User",
                "employee_id": "EMP999"
            }
            
            # Should not raise
            instance = RequestSchema(**data)
            assert instance.request_type.value == req_type
    
    def test_schema_accepts_all_valid_environments(self):
        """Test all valid environments are accepted - catches if enum changes."""
        valid_envs = ["development", "staging", "production"]
        
        for env in valid_envs:
            data = {
                "request_type": "infrastructure-provisioning",
                "target_environment": env,
                "business_justification": "Test",
                "name": "Test User",
                "employee_id": "EMP999"
            }
            
            # Should not raise
            instance = RequestSchema(**data)
            assert instance.target_environment.value == env


class TestGetTextFields:
    """Test get_text_fields function - Critical for privacy."""
    
    def test_excludes_pii_fields(self):
        """CRITICAL: Test that PII fields are never in text fields - privacy violation if fails."""
        text_fields = get_text_fields()
        
        # These MUST NOT be in text fields (privacy requirement)
        assert "name" not in text_fields, "PRIVACY VIOLATION: name should not be in text fields"
        assert "employee_id" not in text_fields, "PRIVACY VIOLATION: employee_id should not be in text fields"
    
    def test_includes_business_fields(self):
        """Test that business fields ARE included - catches if logic breaks."""
        text_fields = get_text_fields()
        
        # These SHOULD be in text fields
        assert "request_type" in text_fields, "request_type should be in text fields for duplicate detection"
        assert "target_environment" in text_fields, "target_environment should be in text fields"
        assert "business_justification" in text_fields, "business_justification should be in text fields"
    
    def test_returns_non_empty_list(self):
        """Test that function returns actual fields - catches if completely broken."""
        text_fields = get_text_fields()
        
        assert isinstance(text_fields, list), "Should return a list"
        assert len(text_fields) > 0, "Should return at least some fields"
    
    def test_all_returned_fields_are_strings(self):
        """Test field names are strings - catches type errors."""
        text_fields = get_text_fields()
        
        for field in text_fields:
            assert isinstance(field, str), f"Field {field} should be a string"
            assert len(field) > 0, f"Field name should not be empty"


class TestSchemaToDict:
    """Test schema_to_dict function."""
    
    def test_converts_to_dict(self):
        """Test basic conversion works."""
        data = {
            "request_type": "infrastructure-provisioning",
            "target_environment": "development",
            "business_justification": "Test",
            "name": "John Doe",
            "employee_id": "EMP123"
        }
        
        instance = RequestSchema(**data)
        result = schema_to_dict(instance)
        
        assert isinstance(result, dict), "Should return a dictionary"
    
    def test_preserves_all_fields(self):
        """Test that no data is lost in conversion - catches data loss bugs."""
        data = {
            "request_type": "service-deployment",
            "target_environment": "production",
            "business_justification": "Critical deployment needed",
            "name": "Jane Smith",
            "employee_id": "EMP456"
        }
        
        instance = RequestSchema(**data)
        result = schema_to_dict(instance)
        
        # All original fields should be present
        assert "request_type" in result
        assert "target_environment" in result
        assert "business_justification" in result
        assert "name" in result
        assert "employee_id" in result
        
        # Values should match (convert enums to strings for comparison)
        assert str(result["request_type"]) == data["request_type"]
        assert str(result["target_environment"]) == data["target_environment"]
        assert result["business_justification"] == data["business_justification"]
        assert result["name"] == data["name"]
        assert result["employee_id"] == data["employee_id"]
    
    def test_handles_special_characters(self):
        """Test that special characters in strings are preserved - catches encoding issues."""
        data = {
            "request_type": "infrastructure-provisioning",
            "target_environment": "development",
            "business_justification": "Test with special chars: @#$%^&*()_+-=[]{}|;:',.<>?/~`",
            "name": "O'Brien-Smith",
            "employee_id": "EMP123"
        }
        
        instance = RequestSchema(**data)
        result = schema_to_dict(instance)
        
        # Special characters should be preserved
        assert result["business_justification"] == data["business_justification"]
        assert result["name"] == data["name"]
    
    def test_handles_unicode(self):
        """Test Unicode characters are preserved - catches encoding issues."""
        data = {
            "request_type": "infrastructure-provisioning",
            "target_environment": "development",
            "business_justification": "Test with Unicode: 你好 مرحبا שלום",
            "name": "José García",
            "employee_id": "EMP123"
        }
        
        instance = RequestSchema(**data)
        result = schema_to_dict(instance)
        
        # Unicode should be preserved
        assert result["business_justification"] == data["business_justification"]
        assert result["name"] == data["name"]


class TestSchemaIntegration:
    """Integration tests for schema functionality."""
    
    def test_privacy_workflow(self):
        """CRITICAL: Test complete privacy-preserving workflow - catches privacy violations."""
        # 1. Create request with PII
        data = {
            "request_type": "infrastructure-provisioning",
            "target_environment": "development",
            "business_justification": "Need infrastructure for new project",
            "name": "Alice Johnson",
            "employee_id": "EMP789"
        }
        instance = RequestSchema(**data)
        
        # 2. Convert to dict
        result_dict = schema_to_dict(instance)
        
        # 3. Get text fields for embedding
        text_fields = get_text_fields()
        
        # 4. Extract only text fields (for duplicate detection)
        text_content = " ".join([str(result_dict.get(field, "")) for field in text_fields])
        
        # 5. CRITICAL: Verify PII is NOT in text content
        assert "Alice Johnson" not in text_content, "PRIVACY VIOLATION: Name leaked into text content"
        assert "EMP789" not in text_content, "PRIVACY VIOLATION: Employee ID leaked into text content"
        
        # 6. Verify business data IS in text content
        assert "infrastructure-provisioning" in text_content or "infrastructure" in text_content.lower()
        assert "development" in text_content.lower()
        assert "infrastructure" in text_content.lower() or "project" in text_content.lower()
    
    def test_schema_consistency_across_instances(self):
        """Test that schema behaves consistently - catches state pollution bugs."""
        data1 = {
            "request_type": "infrastructure-provisioning",
            "target_environment": "development",
            "business_justification": "First request",
            "name": "User One",
            "employee_id": "EMP001"
        }
        
        data2 = {
            "request_type": "service-deployment",
            "target_environment": "production",
            "business_justification": "Second request",
            "name": "User Two",
            "employee_id": "EMP002"
        }
        
        instance1 = RequestSchema(**data1)
        instance2 = RequestSchema(**data2)
        
        dict1 = schema_to_dict(instance1)
        dict2 = schema_to_dict(instance2)
        
        # Instances should be independent
        assert dict1["name"] == "User One"
        assert dict2["name"] == "User Two"
        assert dict1["business_justification"] != dict2["business_justification"]

# Made with Bob
