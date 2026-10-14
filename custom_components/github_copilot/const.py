"""Constants for the GitHub Copilot integration."""

from __future__ import annotations

import logging
from typing import Final

from homeassistant.const import CONF_LLM_HASS_API
from homeassistant.helpers import llm

DOMAIN: Final = "github_copilot"
LOGGER = logging.getLogger(__package__)

CONF_CHAT_MODEL: Final = "chat_model"
CONF_PROMPT: Final = "prompt"
CONF_REASONING_EFFORT: Final = "reasoning_effort"

SUBENTRY_CONVERSATION: Final = "conversation"
SUBENTRY_AI_TASK: Final = "ai_task_data"

DEFAULT_CONVERSATION_NAME: Final = "GitHub Copilot conversation"
DEFAULT_AI_TASK_NAME: Final = "GitHub Copilot AI Task"

# Preferred defaults, in order. The first one the account can use is picked.
PREFERRED_CHAT_MODELS: Final = (
    "gpt-5-mini",
    "gpt-4.1",
    "claude-haiku-4.5",
    "auto",
)
FALLBACK_CHAT_MODEL: Final = "gpt-5-mini"

REASONING_EFFORT_DEFAULT: Final = "default"
REASONING_EFFORTS: Final = (REASONING_EFFORT_DEFAULT, "low", "medium", "high")
RECOMMENDED_REASONING_EFFORT: Final = "low"

DEFAULT_CONVERSATION_OPTIONS: Final = {
    CONF_LLM_HASS_API: [llm.LLM_API_ASSIST],
    CONF_PROMPT: llm.DEFAULT_INSTRUCTIONS_PROMPT,
    CONF_REASONING_EFFORT: RECOMMENDED_REASONING_EFFORT,
}
DEFAULT_AI_TASK_OPTIONS: Final = {
    CONF_REASONING_EFFORT: RECOMMENDED_REASONING_EFFORT,
}

# Timeouts (seconds) for a single request, including all tool round trips.
CONVERSATION_TIMEOUT: Final = 120
AI_TASK_TIMEOUT: Final = 300

# Upper bound on how much of a single prior tool result is replayed as history.
MAX_HISTORY_TOOL_RESULT_CHARS: Final = 2000
