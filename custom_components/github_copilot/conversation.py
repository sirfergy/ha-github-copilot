"""Conversation agent backed by GitHub Copilot."""

from __future__ import annotations

from typing import Literal

from homeassistant.components import conversation
from homeassistant.config_entries import ConfigSubentry
from homeassistant.const import CONF_LLM_HASS_API, MATCH_ALL
from homeassistant.core import HomeAssistant
from homeassistant.helpers.entity_platform import AddConfigEntryEntitiesCallback

from . import GitHubCopilotConfigEntry
from .const import CONF_PROMPT, CONVERSATION_TIMEOUT, DOMAIN, SUBENTRY_CONVERSATION
from .entity import GitHubCopilotBaseLLMEntity


async def async_setup_entry(
    hass: HomeAssistant,
    config_entry: GitHubCopilotConfigEntry,
    async_add_entities: AddConfigEntryEntitiesCallback,
) -> None:
    """Set up one conversation entity per conversation subentry."""
    for subentry in config_entry.subentries.values():
        if subentry.subentry_type != SUBENTRY_CONVERSATION:
            continue
        async_add_entities(
            [GitHubCopilotConversationEntity(config_entry, subentry)],
            config_subentry_id=subentry.subentry_id,
        )


class GitHubCopilotConversationEntity(
    conversation.ConversationEntity,
    conversation.AbstractConversationAgent,
    GitHubCopilotBaseLLMEntity,
):
    """GitHub Copilot conversation agent."""

    _attr_supports_streaming = True

    def __init__(
        self, entry: GitHubCopilotConfigEntry, subentry: ConfigSubentry
    ) -> None:
        """Initialize the agent."""
        super().__init__(entry, subentry)
        if self.subentry.data.get(CONF_LLM_HASS_API):
            self._attr_supported_features = (
                conversation.ConversationEntityFeature.CONTROL
            )

    @property
    def supported_languages(self) -> list[str] | Literal["*"]:
        """Return a list of supported languages."""
        return MATCH_ALL

    async def _async_handle_message(
        self,
        user_input: conversation.ConversationInput,
        chat_log: conversation.ChatLog,
    ) -> conversation.ConversationResult:
        """Process a sentence."""
        options = self.subentry.data
        try:
            await chat_log.async_provide_llm_data(
                user_input.as_llm_context(DOMAIN),
                options.get(CONF_LLM_HASS_API),
                options.get(CONF_PROMPT),
                user_input.extra_system_prompt,
            )
        except conversation.ConverseError as err:
            return err.as_conversation_result()

        await self._async_handle_chat_log(
            chat_log, request_timeout=CONVERSATION_TIMEOUT
        )

        return conversation.async_get_result_from_chat_log(user_input, chat_log)
