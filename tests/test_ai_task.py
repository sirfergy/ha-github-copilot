"""Tests for the GitHub Copilot AI Task entity."""

from __future__ import annotations

from pathlib import Path
import struct
from unittest.mock import patch

from copilot.session_events import UserMessageData
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


def _client_runtime(entry: MockConfigEntry):
    return entry.runtime_data


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


async def test_attachment_hard_cap_for_unknown_model(
    hass: HomeAssistant,
    mock_config_entry: MockConfigEntry,
    setup_integration: FakeCopilotClient,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Models that don't advertise limits still get the integration's caps."""
    from .conftest import _model

    monkeypatch.setattr(
        "custom_components.github_copilot.entity.MAX_ATTACHMENT_BYTES", 10
    )
    subentry = next(
        s
        for s in mock_config_entry.subentries.values()
        if s.subentry_type == SUBENTRY_AI_TASK
    )
    hass.config_entries.async_update_subentry(
        mock_config_entry, subentry, data={CONF_CHAT_MODEL: "my-custom-model"}
    )
    await hass.async_block_till_done()
    _client().models.append(_model("my-custom-model", vision_limits=False))
    await _client_runtime(mock_config_entry).async_refresh_models()
    image = tmp_path / "a.png"
    image.write_bytes(b"x" * 11)
    with (
        patch.object(Path, "read_bytes", side_effect=AssertionError("read")),
        pytest.raises(HomeAssistantError, match="larger than 10 bytes"),
    ):
        await _generate_with_image(hass, image, "image/png")


async def test_unknown_model_is_refused(
    hass: HomeAssistant,
    mock_config_entry: MockConfigEntry,
    setup_integration: FakeCopilotClient,
) -> None:
    """Copilot would silently substitute a model, so unknown ids fail loudly."""
    subentry = next(
        s
        for s in mock_config_entry.subentries.values()
        if s.subentry_type == SUBENTRY_AI_TASK
    )
    hass.config_entries.async_update_subentry(
        mock_config_entry, subentry, data={CONF_CHAT_MODEL: "not-a-real-model"}
    )
    await hass.async_block_till_done()
    with pytest.raises(HomeAssistantError, match="isn't available"):
        await ai_task.async_generate_data(
            hass, task_name="x", entity_id=ENTITY_ID, instructions="hi"
        )
    assert _client().sessions == []

    # A model added to the account later is picked up without a reload.
    from .conftest import _model

    _client().models.append(_model("not-a-real-model"))
    _client().script = _reply("ok")
    result = await ai_task.async_generate_data(
        hass, task_name="x", entity_id=ENTITY_ID, instructions="hi"
    )
    assert result.data == "ok"
    assert _client().sessions[-1].options["model"] == "not-a-real-model"


def _png(width: int, height: int) -> bytes:
    header = struct.pack(">II", width, height) + b"\x08\x02\x00\x00\x00"
    return b"\x89PNG\r\n\x1a\n" + b"\x00\x00\x00\x0dIHDR" + header + b"rest"


@pytest.fixture
def image_tool(mock_copilot: type[FakeCopilotClient]) -> None:
    """Offer Copilot's image tool to the account."""

    def configure(client: FakeCopilotClient) -> None:
        client.image_tool_available = True

    mock_copilot.configure = configure


async def _generate_image(hass: HomeAssistant, instructions: str = "a cabin"):
    with patch.object(
        media_source.local_source.LocalSource,
        "async_upload_media",
        return_value="media-source://ai_task/image/test.png",
    ) as upload:
        result = await ai_task.async_generate_image(
            hass, task_name="cabin", entity_id=ENTITY_ID, instructions=instructions
        )
    return result, upload


async def test_image_feature_requires_account_access(
    hass: HomeAssistant, setup_integration: FakeCopilotClient
) -> None:
    """Without the image tool the entity doesn't advertise image generation."""
    features = hass.states.get(ENTITY_ID).attributes["supported_features"]
    assert not features & ai_task.AITaskEntityFeature.GENERATE_IMAGE
    with pytest.raises(HomeAssistantError, match="does not support generating"):
        await _generate_image(hass)


async def test_generate_image(
    hass: HomeAssistant, image_tool: None, setup_integration: FakeCopilotClient
) -> None:
    """Images from Copilot's image tool are returned and stored by HA."""
    features = hass.states.get(ENTITY_ID).attributes["supported_features"]
    assert features & ai_task.AITaskEntityFeature.GENERATE_IMAGE
    png = _png(1536, 1024)

    async def script(session: FakeSession) -> None:
        session.emit(UserMessageData(content="x"))
        await session.generate_image(png, prompt="A cozy cabin at dusk, snow")
        # Copilot would continue with a chat turn; we abort instead.

    _client().script = script
    result, upload = await _generate_image(hass)

    assert result["mime_type"] == "image/png"
    assert (result["width"], result["height"]) == (1536, 1024)
    assert result["revised_prompt"] == "A cozy cabin at dusk, snow"
    assert upload.call_args[0][1].file.getvalue() == png

    session = _client().sessions[-1]
    assert session.options["available_tools"] == ["builtin:image_generation"]
    assert session.options["image_generation"] == {"enabled": True}
    assert session.options["feature_flags"] == {"IMAGE_GENERATION_TOOL": True}
    assert session.options["tools"] == []
    assert session.permission_decisions == ["PermissionDecisionApproveOnce"]
    assert "image_generation tool" in session.options["system_message"]["content"]
    await hass.async_block_till_done()
    # The follow-up chat turn is skipped.
    assert session.aborted


async def test_image_session_denies_other_permissions(
    hass: HomeAssistant, image_tool: None, setup_integration: FakeCopilotClient
) -> None:
    """Only the image tool is approved in image sessions."""
    from copilot.session_events import PermissionRequestCustomTool

    async def script(session: FakeSession) -> None:
        decision = session.options["on_permission_request"](
            PermissionRequestCustomTool(tool_description="x", tool_name="bash"), {}
        )
        session.permission_decisions.append(type(decision).__name__)
        session.say("no")
        session.idle()

    _client().script = script
    with pytest.raises(HomeAssistantError):
        await _generate_image(hass)
    assert _client().sessions[-1].permission_decisions == ["PermissionDecisionReject"]


async def test_generate_image_refused(
    hass: HomeAssistant, image_tool: None, setup_integration: FakeCopilotClient
) -> None:
    """If the model declines, its explanation is surfaced."""
    _client().script = _reply("I can't create that image.")
    with pytest.raises(HomeAssistantError, match="I can't create that image"):
        await _generate_image(hass)


async def test_generate_image_tool_failure(
    hass: HomeAssistant, image_tool: None, setup_integration: FakeCopilotClient
) -> None:
    """A failed image tool call is reported."""
    from copilot.session_events import (
        ToolExecutionCompleteData,
        ToolExecutionCompleteError,
        ToolExecutionStartData,
    )

    async def script(session: FakeSession) -> None:
        session.emit(UserMessageData(content="x"))
        session.emit(
            ToolExecutionStartData(
                tool_call_id="c1", tool_name="image_generation", arguments={}
            )
        )
        session.emit(
            ToolExecutionCompleteData(
                success=False,
                tool_call_id="c1",
                error=ToolExecutionCompleteError(message="content policy"),
            )
        )

    _client().script = script
    with pytest.raises(HomeAssistantError, match="content policy"):
        await _generate_image(hass)
