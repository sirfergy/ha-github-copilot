"""JSON schema helpers for GitHub Copilot structured output and tools."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from homeassistant.exceptions import HomeAssistantError

_DROPPED_KEYWORDS = frozenset(
    {
        "default",
        "examples",
        "$comment",
        "deprecated",
        "readOnly",
        "writeOnly",
        "uniqueItems",
    }
)
_UNSUPPORTED_KEYWORDS = frozenset(
    {"not", "dependentRequired", "dependentSchemas", "if", "then", "else"}
)
# Formats every provider behind Copilot understands; selector-specific formats
# such as "entity_id" are dropped so strict providers don't reject the schema.
_ALLOWED_FORMATS = frozenset(
    {
        "date-time",
        "date",
        "time",
        "duration",
        "email",
        "hostname",
        "ipv4",
        "ipv6",
        "uuid",
    }
)
_UNSUPPORTED_TOOL_ROOT_KEYS = frozenset({"oneOf", "anyOf", "allOf", "enum", "not"})


def strict_output_schema(schema: dict[str, Any]) -> dict[str, Any]:
    """Return a copy of an OpenAPI schema normalized for strict structured output.

    Copilot forwards the schema to the selected provider's strict JSON mode, which
    requires closed objects (``additionalProperties: false``) and, for OpenAI
    models, every property listed as required. Optional properties are therefore
    made nullable instead.
    """
    result = deepcopy(schema)
    _adjust(result, "$", nullable=False)
    if result.get("type") != "object":
        raise HomeAssistantError("Structured output must be an object")
    return result


def tool_parameters_schema(schema: dict[str, Any]) -> dict[str, Any]:
    """Normalize a tool's parameter schema to a plain object schema."""
    result = {k: v for k, v in schema.items() if k not in _UNSUPPORTED_TOOL_ROOT_KEYS}
    result.setdefault("type", "object")
    if result["type"] == "object":
        result.setdefault("properties", {})
    return result


def _adjust(node: Any, path: str, *, nullable: bool) -> None:
    if not isinstance(node, dict):
        raise HomeAssistantError(f"Unsupported structured output schema at {path}")

    _merge_single_all_of(node, path)
    for keyword in _DROPPED_KEYWORDS & node.keys():
        del node[keyword]
    if unsupported := _UNSUPPORTED_KEYWORDS & node.keys():
        raise HomeAssistantError(
            f"Unsupported structured output keywords at {path}: "
            f"{', '.join(sorted(unsupported))}"
        )
    if "format" in node and node["format"] not in _ALLOWED_FORMATS:
        del node["format"]

    for name, definition in node.get("$defs", {}).items():
        _adjust(definition, f"{path}.$defs.{name}", nullable=False)
    for keyword in ("anyOf", "oneOf"):
        for index, variant in enumerate(node.get(keyword, [])):
            _adjust(variant, f"{path}.{keyword}[{index}]", nullable=False)

    types = node.get("type", [])
    types = [types] if isinstance(types, str) else list(types)
    if "object" in types:
        if node.get("additionalProperties", False) is not False and not node.get(
            "properties"
        ):
            raise HomeAssistantError(
                f"Structured output objects need explicit fields (at {path})"
            )
        node["additionalProperties"] = False
        properties = node.setdefault("properties", {})
        required = node.setdefault("required", [])
        for name, prop in properties.items():
            _adjust(prop, f"{path}.{name}", nullable=name not in required)
            if name not in required:
                required.append(name)
    if "array" in types:
        node.setdefault("items", {"type": "string"})
        _adjust(node["items"], f"{path}[]", nullable=False)

    if nullable:
        _make_nullable(node)


def _merge_single_all_of(node: dict[str, Any], path: str) -> None:
    while "allOf" in node:
        branches = node.pop("allOf")
        if len(branches) != 1 or not isinstance(branches[0], dict):
            raise HomeAssistantError(f"Unsupported allOf in schema at {path}")
        for key, value in branches[0].items():
            node.setdefault(key, value)


def _make_nullable(node: dict[str, Any]) -> None:
    if "type" not in node or node.keys() & {"$ref", "const", "anyOf", "oneOf"}:
        original = dict(node)
        node.clear()
        node["anyOf"] = [original, {"type": "null"}]
        return
    types = node["type"]
    types = [types] if isinstance(types, str) else list(types)
    if "null" not in types:
        types.append("null")
    node["type"] = types
    if "enum" in node and None not in node["enum"]:
        node["enum"] = [*node["enum"], None]
