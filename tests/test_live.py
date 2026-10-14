"""Live end-to-end tests against the real Copilot runtime.

Skipped unless GITHUB_COPILOT_LIVE_TOKEN is set. Each test consumes a few
Copilot requests.
"""

from __future__ import annotations

import os
from pathlib import Path
import struct
from unittest.mock import patch
import zlib

from homeassistant.components import ai_task, conversation, media_source
from homeassistant.components.homeassistant.exposed_entities import (
    async_expose_entity,
)
from homeassistant.config_entries import ConfigEntryState
from homeassistant.const import CONF_ACCESS_TOKEN
from homeassistant.core import Context, HomeAssistant
from homeassistant.helpers import selector
from homeassistant.setup import async_setup_component
import probatio
import pytest
from pytest_homeassistant_custom_component.common import MockConfigEntry

from custom_components.github_copilot.const import (
    CONF_CHAT_MODEL,
    CONF_REASONING_EFFORT,
    DEFAULT_AI_TASK_OPTIONS,
    DEFAULT_CONVERSATION_OPTIONS,
    DOMAIN,
    SUBENTRY_AI_TASK,
    SUBENTRY_CONVERSATION,
)

TOKEN = os.environ.get("GITHUB_COPILOT_LIVE_TOKEN")
MODEL = os.environ.get("GITHUB_COPILOT_LIVE_MODEL", "gpt-5-mini")

pytestmark = [
    pytest.mark.skipif(not TOKEN, reason="GITHUB_COPILOT_LIVE_TOKEN not set"),
    pytest.mark.timeout(600),
]


@pytest.fixture
async def live_entry(hass: HomeAssistant):
    """Set up the integration with the real runtime."""
    options = {CONF_CHAT_MODEL: MODEL, CONF_REASONING_EFFORT: "low"}
    entry = MockConfigEntry(
        domain=DOMAIN,
        unique_id="live",
        data={CONF_ACCESS_TOKEN: TOKEN},
        subentries_data=[
            {
                "subentry_type": SUBENTRY_CONVERSATION,
                "data": {**DEFAULT_CONVERSATION_OPTIONS, **options},
                "title": "Copilot",
                "unique_id": None,
            },
            {
                "subentry_type": SUBENTRY_AI_TASK,
                "data": {**DEFAULT_AI_TASK_OPTIONS, **options},
                "title": "Copilot task",
                "unique_id": None,
            },
        ],
    )
    entry.add_to_hass(hass)
    assert await hass.config_entries.async_setup(entry.entry_id)
    await hass.async_block_till_done()
    assert entry.state is ConfigEntryState.LOADED
    yield entry
    assert await hass.config_entries.async_unload(entry.entry_id)
    await hass.async_block_till_done()


def _png(rgb: tuple[int, int, int], size: int = 64) -> bytes:
    row = b"\x00" + bytes(rgb) * size
    raw = row * size

    def chunk(kind: bytes, data: bytes) -> bytes:
        return (
            struct.pack(">I", len(data))
            + kind
            + data
            + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF)
        )

    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", size, size, 8, 2, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(raw))
        + chunk(b"IEND", b"")
    )


async def test_live_controls_exposed_entity(
    hass: HomeAssistant, live_entry: MockConfigEntry
) -> None:
    """The agent turns on an exposed entity through the Assist API."""
    assert await async_setup_component(
        hass,
        "input_boolean",
        {"input_boolean": {"kitchen_light": {"name": "Kitchen light"}}},
    )
    async_expose_entity(hass, "conversation", "input_boolean.kitchen_light", True)
    assert hass.states.get("input_boolean.kitchen_light").state == "off"

    result = await conversation.async_converse(
        hass,
        "Turn on the kitchen light",
        None,
        Context(),
        agent_id="conversation.copilot",
    )
    print("REPLY:", result.response.speech)
    assert result.response.response_type.value == "action_done"
    assert hass.states.get("input_boolean.kitchen_light").state == "on"


async def test_live_multi_turn(
    hass: HomeAssistant, live_entry: MockConfigEntry
) -> None:
    """Follow-up questions see earlier turns."""
    first = await conversation.async_converse(
        hass,
        "Remember this: my favorite color is teal.",
        None,
        Context(),
        agent_id="conversation.copilot",
    )
    second = await conversation.async_converse(
        hass,
        "What is my favorite color? Answer in one word.",
        first.conversation_id,
        Context(),
        agent_id="conversation.copilot",
    )
    speech = second.response.speech["plain"]["speech"]
    print("REPLY:", speech)
    assert "teal" in speech.lower()


async def test_live_ai_task_structured_with_image(
    hass: HomeAssistant, live_entry: MockConfigEntry, tmp_path: Path
) -> None:
    """Structured output works with an image attachment."""
    image = tmp_path / "snapshot.png"
    image.write_bytes(_png((220, 20, 20)))
    with patch(
        "homeassistant.components.media_source.async_resolve_media",
        return_value=media_source.PlayMedia(
            url="/x", mime_type="image/png", path=image
        ),
    ):
        result = await ai_task.async_generate_data(
            hass,
            task_name="color check",
            entity_id="ai_task.copilot_task",
            instructions="What is the dominant color of this image?",
            structure=probatio.Schema(
                {
                    probatio.Required(
                        "color", description="Lowercase color name"
                    ): selector.selector({"text": {}}),
                    probatio.Optional("confidence"): selector.selector(
                        {"number": {"min": 0, "max": 1}}
                    ),
                }
            ),
            attachments=[
                {
                    "media_content_id": "media-source://media_source/local/x.png",
                    "media_content_type": "image/png",
                }
            ],
        )
    print("DATA:", result.data)
    assert isinstance(result.data, dict)
    assert result.data["color"] == "red"


async def test_live_ai_task_text(
    hass: HomeAssistant, live_entry: MockConfigEntry
) -> None:
    """Plain text generation works."""
    result = await ai_task.async_generate_data(
        hass,
        task_name="haiku",
        entity_id="ai_task.copilot_task",
        instructions="Reply with exactly the word: pong",
    )
    print("DATA:", result.data)
    assert "pong" in result.data.lower()
