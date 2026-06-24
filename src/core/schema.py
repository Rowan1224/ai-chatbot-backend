"""Dynamic Pydantic schema generation from YAML configuration."""

from enum import Enum
from typing import Any, Dict, List, Type

from pydantic import BaseModel, Field, create_model

from src.config.settings import prompt_config


def generate_request_schema() -> Type[BaseModel]:
    """
    Generate Pydantic model dynamically from prompt configuration.
    
    This allows the schema to change based on the YAML config without code changes.
    The generated model is used for OpenAI structured output.
    
    Returns:
        Dynamically created Pydantic model class
    """
    fields: Dict[str, Any] = {}
    
    for field_config in prompt_config.fields:
        field_name = field_config["name"]
        field_type = field_config["type"]
        field_desc = field_config.get("description", "")
        is_required = field_config.get("required", True)
        
        # Determine Python type based on config
        if field_type == "enum":
            # Create enum dynamically
            options = field_config.get("options", [])
            enum_name = f"{field_name.title().replace('_', '')}Enum"
            enum_class = Enum(enum_name, {opt: opt for opt in options})
            python_type = enum_class
        
        elif field_type == "string" or field_type == "text":
            python_type = str
        
        elif field_type == "integer" or field_type == "int":
            python_type = int
        
        elif field_type == "number" or field_type == "float":
            python_type = float
        
        elif field_type == "boolean" or field_type == "bool":
            python_type = bool
        
        else:
            # Default to string for unknown types
            python_type = str
        
        # Create field with description
        if is_required:
            fields[field_name] = (python_type, Field(..., description=field_desc))
        else:
            fields[field_name] = (python_type, Field(default=None, description=field_desc))
    
    # Create the model dynamically
    RequestModel = create_model(
        "DynamicRequest",
        __doc__=f"Request schema for config version {prompt_config.config_version}",
        **fields
    )
    
    return RequestModel


def get_text_fields() -> List[str]:
    """
    Get list of text fields for embedding generation (excluding PII).
    
    Only text/string fields are used for semantic similarity.
    PII fields (name, employee_id) are excluded for privacy.
    
    Returns:
        List of field names that contain text (non-PII only)
    """
    # Get PII fields from config
    pii_fields = prompt_config.privacy.get("pii_fields", ["name", "employee_id"])
    
    text_fields = []
    for field_config in prompt_config.fields:
        field_name = field_config["name"]
        field_type = field_config["type"]
        
        # Include only text fields that are not PII
        if field_type in ["string", "text", "enum"] and field_name not in pii_fields:
            text_fields.append(field_name)
    
    return text_fields


def schema_to_dict(model_instance: BaseModel) -> Dict[str, Any]:
    """
    Convert Pydantic model instance to dictionary.
    
    Handles enum values properly (converts to string).
    
    Args:
        model_instance: Pydantic model instance
        
    Returns:
        Dictionary representation
    """
    data = model_instance.model_dump()
    
    # Convert enum values to strings
    for key, value in data.items():
        if isinstance(value, Enum):
            data[key] = value.value
    
    return data


# Generate the schema at module load time
RequestSchema = generate_request_schema()

# Made with Bob
