"""Tests for the GitHub Copilot AI Task entity."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

from homeassistant.components import ai_task, media_source
from homeassistant.core import HomeAssistant
from homeassistant.exceptions import HomeAssistantError
from homeassistant.helpers import entity_registry as er, selector
import probatio
import pytest
from pytest_homeassistant_custom_component.common import MockConfigEntry

from custom_components.github_copilot.const import CONF_CHAT_MODEL, SUBENTRY_AI_TASK

from .conftest import FakeCopilotClient, FakeSession

ENTITY_ID = "ai_task.github_copilot_ai_task"


def _client() -> FakeCopilotClient:
    return FakeCopilotClient.instances[-1]


def _reply(text: str):
    async def script(session: FakeSession) -> None:
        session.say(text)
        session.idle()

    return script


async def test_entity_linked_to_subentry(
    hass: HomeAssistant,
    mock_config_entry: MockConfigEntry,
    setup_integration: FakeCopilotClient,
    entity_registry: er.EntityRegistry,
) -> None:
    """The AI task entity belongs to the AI task subentry."""
    subentry = next(
        s
        for s in mock_config_entry.subentries.values()
        if s.subentry_type == SUBENTRY_AI_TASK
    )
    entity = entity_registry.async_get(ENTITY_ID)
    assert entity is not None
    assert entity.config_entry_id == mock_config_entry.entry_id
    assert entity.config_subentry_id == subentry.subentry_id


async def test_generate_text(
    hass: HomeAssistant, setup_integration: FakeCopilotClient
) -> None:
    """Free-text generation returns the assistant's reply."""
    _client().script = _reply("The garage door is open!")
    result = await ai_task.async_generate_data(
        hass,
        task_name="garage",
        entity_id=ENTITY_ID,
        instructions="Write a funny alert",
    )
    assert result.data == "The garage door is open!"
    session = _client().sessions[-1]
    assert session.sent[0]["prompt"] == "Write a funny alert"
    assert session.options["tools"] == []
    assert "Home Assistant expert" in session.options["system_message"]["content"]


@pytest.mark.parametrize(
    "reply",
    [
        '{"birds": 3, "note": null}',
        '```json\n{"birds": 3, "note": null}\n```',
        'Here you go: {"birds": 3, "note": null}',
    ],
)
async def test_generate_structured(
    hass: HomeAssistant, setup_integration: FakeCopilotClient, reply: str
) -> None:
    """Structured output sends a strict schema and parses the JSON reply."""
    _client().script = _reply(reply)
    result = await ai_task.async_generate_data(
        hass,
        task_name="birds",
        entity_id=ENTITY_ID,
        instructions="Count the birds",
        structure=probatio.Schema(
            {
                probatio.Required("birds", description="Bird count"): selector.selector(
                    {"number": {}}
                ),
                probatio.Optional("note"): selector.selector({"text": {}}),
            }
        ),
    )
    assert result.data == {"birds": 3, "note": None}

    sent = _client().sessions[-1].sent[0]
    schema = sent["response_schema"]
    assert schema["type"] == "object"
    assert schema["additionalProperties"] is False
    assert schema["required"] == ["birds", "note"]
    assert schema["properties"]["birds"]["type"] == "number"
    assert schema["properties"]["birds"]["description"] == "Bird count"
    assert schema["properties"]["note"]["type"] == ["string", "null"]
    assert "Respond with only a JSON object" in sent["prompt"]


async def test_generate_structured_invalid_json(
    hass: HomeAssistant, setup_integration: FakeCopilotClient
) -> None:
    """Unparseable structured replies raise."""
    _client().script = _reply("I can't count birds.")
    with pytest.raises(HomeAssistantError, match="invalid JSON"):
        await ai_task.async_generate_data(
            hass,
            task_name="birds",
            entity_id=ENTITY_ID,
            instructions="Count the birds",
            structure=probatio.Schema(
                {probatio.Required("birds"): selector.selector({"number": {}})}
            ),
        )


async def _generate_with_image(hass: HomeAssistant, path: Path, mime: str):
    with patch(
        "homeassistant.components.media_source.async_resolve_media",
        return_value=media_source.PlayMedia(url="/x", mime_type=mime, path=path),
    ):
        return await ai_task.async_generate_data(
            hass,
            task_name="camera",
            entity_id=ENTITY_ID,
            instructions="Describe the image",
            attachments=[
                {
                    "media_content_id": "media-source://media_source/local/a.png",
                    "media_content_type": mime,
                }
            ],
        )


async def test_attachments(
    hass: HomeAssistant, setup_integration: FakeCopilotClient, tmp_path: Path
) -> None:
    """Image attachments are sent inline as base64 blobs."""
    image = tmp_path / "a.png"
    image.write_bytes(b"\x89PNGdata")
    _client().script = _reply("A red square")

    result = await _generate_with_image(hass, image, "image/png")

    assert result.data == "A red square"
    (blob,) = _client().sessions[-1].sent[0]["attachments"]
    assert blob == {
        "type": "blob",
        "data": "iVBOR2RhdGE=",
        "mimeType": "image/png",
        "displayName": "a.png",
    }


async def test_attachment_too_large(
    hass: HomeAssistant, setup_integration: FakeCopilotClient, tmp_path: Path
) -> None:
    """Attachments over the model's size limit are rejected before sending."""
    image = tmp_path / "big.png"
    image.write_bytes(b"x" * 2048)
    with pytest.raises(HomeAssistantError, match="larger than"):
        await _generate_with_image(hass, image, "image/png")
    assert _client().sessions == []


async def test_attachment_unsupported_type(
    hass: HomeAssistant, setup_integration: FakeCopilotClient, tmp_path: Path
) -> None:
    """Non-image attachments are rejected."""
    doc = tmp_path / "a.txt"
    doc.write_text("hello")
    with pytest.raises(HomeAssistantError, match="Unsupported attachment type"):
        await _generate_with_image(hass, doc, "text/plain")


async def test_attachment_needs_vision_model(
    hass: HomeAssistant,
    mock_config_entry: MockConfigEntry,
    setup_integration: FakeCopilotClient,
    tmp_path: Path,
) -> None:
    """Models without vision support refuse attachments."""
    subentry = next(
        s
        for s in mock_config_entry.subentries.values()
        if s.subentry_type == SUBENTRY_AI_TASK
    )
    hass.config_entries.async_update_subentry(
        mock_config_entry, subentry, data={CONF_CHAT_MODEL: "text-only"}
    )
    await hass.async_block_till_done()
    image = tmp_path / "a.png"
    image.write_bytes(b"png")
    with pytest.raises(HomeAssistantError, match="does not support image"):
        await _generate_with_image(hass, image, "image/png")


async def test_error_event(
    hass: HomeAssistant, setup_integration: FakeCopilotClient
) -> None:
    """Session errors raise HomeAssistantError."""

    async def script(session: FakeSession) -> None:
        session.error("model unavailable")
        session.idle()

    _client().script = script
    with pytest.raises(HomeAssistantError, match="model unavailable"):
        await ai_task.async_generate_data(
            hass, task_name="x", entity_id=ENTITY_ID, instructions="hi"
        )
