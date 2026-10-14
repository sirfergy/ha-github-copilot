"""Tests for schema normalization."""

from __future__ import annotations

from homeassistant.exceptions import HomeAssistantError
import pytest

from custom_components.github_copilot.schema import (
    strict_output_schema,
    tool_parameters_schema,
)


def test_strict_schema_nested() -> None:
    """Nested objects and arrays are closed and optional fields nullable."""
    schema = {
        "type": "object",
        "properties": {
            "entity": {"type": "string", "format": "entity_id", "default": "x"},
            "when": {"type": "string", "format": "date-time"},
            "items": {
                "type": "array",
                "uniqueItems": True,
                "items": {
                    "type": "object",
                    "properties": {"name": {"type": "string"}},
                },
            },
            "color": {"type": "string", "enum": ["red", "blue"]},
            "ref": {"$ref": "#/$defs/thing"},
        },
        "required": ["entity", "when"],
        "$defs": {"thing": {"type": "object", "properties": {}}},
    }
    result = strict_output_schema(schema)
    assert result["additionalProperties"] is False
    assert result["required"] == ["entity", "when", "items", "color", "ref"]
    assert result["properties"]["entity"] == {"type": "string"}
    assert result["properties"]["when"] == {"type": "string", "format": "date-time"}
    items = result["properties"]["items"]
    assert items["type"] == ["array", "null"]
    assert "uniqueItems" not in items
    assert items["items"]["additionalProperties"] is False
    assert items["items"]["properties"]["name"]["type"] == ["string", "null"]
    assert result["properties"]["color"]["enum"] == ["red", "blue", None]
    assert result["properties"]["ref"] == {
        "anyOf": [{"$ref": "#/$defs/thing"}, {"type": "null"}]
    }
    # The input is not mutated.
    assert "default" in schema["properties"]["entity"]


def test_strict_schema_rejects_free_form_objects() -> None:
    """Objects without declared fields can't be made strict."""
    with pytest.raises(HomeAssistantError, match="explicit fields"):
        strict_output_schema(
            {
                "type": "object",
                "properties": {
                    "data": {"type": "object", "additionalProperties": True}
                },
            }
        )


def test_strict_schema_merges_single_all_of() -> None:
    """A single allOf branch is inlined."""
    result = strict_output_schema(
        {"allOf": [{"type": "object", "properties": {"a": {"type": "integer"}}}]}
    )
    assert result["properties"]["a"]["type"] == ["integer", "null"]


def test_tool_parameters_schema() -> None:
    """Unsupported root keywords are dropped and an object type is ensured."""
    assert tool_parameters_schema({"anyOf": [{}], "properties": {"a": {}}}) == {
        "type": "object",
        "properties": {"a": {}},
    }
    assert tool_parameters_schema({}) == {"type": "object", "properties": {}}
