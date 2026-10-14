# GitHub Copilot for Home Assistant

Use your **GitHub Copilot** subscription as an LLM provider in Home Assistant:

- **Conversation agent** for Assist: chat or talk to Copilot and let it control the entities you expose to Assist.
- **AI Task** entity for automations: `ai_task.generate_data` with free text or structured output, including camera snapshots and other image attachments.

It's built on the official [GitHub Copilot SDK](https://github.com/github/copilot-sdk). You can pick any model your Copilot plan offers (GPT, Claude, Gemini, …) separately for each agent or task.

> [!NOTE]
> This is a community integration. It is not affiliated with or supported by GitHub or the Home Assistant project.

## Requirements

- Home Assistant **2026.10** or newer (HA OS, Container or Core) on Linux x86-64 or arm64. Alpine/musl is supported.
- A GitHub account with an active **Copilot** plan.
- About 150 MB of free space for the Copilot runtime. It is downloaded automatically the first time you set up the integration.
- Outbound HTTPS access to `github.com`, `api.github.com` and the Copilot API.

## Installation

### HACS

1. In HACS, open the menu and choose **Custom repositories**.
2. Add `https://github.com/obvioussean/ha-github-copilot` as an **Integration**.
3. Install **GitHub Copilot** and restart Home Assistant.

### Manual

Copy `custom_components/github_copilot` into your Home Assistant `config/custom_components/` folder and restart.

## Create a token

1. Go to **GitHub → Settings → Developer settings → Fine-grained tokens → [Generate new token](https://github.com/settings/personal-access-tokens/new)**.
2. Give it a name and an expiration. For **Repository access**, the minimum ("Public repositories") is enough.
3. Under **Account permissions**, set **Copilot Requests** to **Read-only**. No other permissions are needed.
4. Generate the token and copy it (`github_pat_…`).

Classic tokens (`ghp_…`) don't work with Copilot. When the token expires, Home Assistant asks you to re-authenticate.

## Setup

1. Go to **Settings → Devices & services → Add integration → GitHub Copilot** and paste the token.
2. The first setup downloads and starts the Copilot runtime, which can take a minute or two.
3. Two services are created: **GitHub Copilot conversation** and **GitHub Copilot AI Task**.

Then, optionally:

- **Assist:** go to **Settings → Voice assistants**, open (or create) an assistant and pick **GitHub Copilot conversation** as the conversation agent. Copilot can only see and control entities that are [exposed to Assist](https://my.home-assistant.io/redirect/voice_assistants/).
- **AI Task default:** go to **Settings → System → General → AI suggestions** and pick the GitHub Copilot AI Task entity as the preferred AI task entity.

### Options

Open the integration and use **⋮ → Reconfigure** on a service, or **Add conversation agent** / **Add AI task** to create more with different models.

| Option | Applies to | Description |
| --- | --- | --- |
| Model | both | Any model your plan allows. The list comes from your account, and you can also type a model ID. |
| Reasoning effort | both | `low` (default) keeps voice responses snappy. Higher effort is slower but can be smarter. Models without reasoning support ignore it. |
| Instructions | conversation | System prompt, which supports templates. |
| Control Home Assistant | conversation | Which LLM APIs (usually **Assist**) Copilot may use as tools. |

## AI Task examples

Count birds in a camera snapshot every 5 minutes:

```yaml
template:
  - triggers:
      - trigger: time_pattern
        minutes: "/5"
    actions:
      - action: ai_task.generate_data
        data:
          task_name: chicken counter
          entity_id: ai_task.github_copilot_ai_task
          instructions: How many chickens are in this picture of the coop?
          structure:
            chickens:
              selector:
                number:
          attachments:
            media_content_id: media-source://camera/camera.chicken_coop
            media_content_type: image/jpeg
        response_variable: result
    sensor:
      - name: Chickens
        state: "{{ result.data.chickens }}"
```

A plain-text notification:

```yaml
- action: ai_task.generate_data
  data:
    task_name: garage
    instructions: Write a short, funny reminder that the garage door has been open for 10 minutes.
  response_variable: reply
- action: notify.mobile_app_phone
  data:
    message: "{{ reply.data }}"
```

## How it works

- Each config entry runs one Copilot runtime process, authenticated with your token. Each request (one conversation turn or AI task) runs in a fresh, short-lived Copilot session.
- The session runs in the SDK's **empty mode** in a private temporary directory:
  - All of Copilot's built-in tools (shell, files, web, MCP, skills, custom instructions) are disabled.
  - The only tools Copilot can call are the Home Assistant LLM tools you enabled.
  - Home Assistant runs those tools itself, so exposure settings and permissions apply. Any other permission request is denied.
- Earlier turns of a conversation are sent as context with each new turn. Tool calls and results appear in the conversation's debug view like any other agent.
- Structured output uses Copilot's native JSON-schema support, with a strict schema derived from your `structure`. For models that ignore it, the reply is parsed leniently.

## Billing and limits

Each conversation turn or AI task counts as a Copilot request against your plan. Premium models consume premium requests according to their multiplier. See [GitHub's documentation](https://docs.github.com/copilot/managing-copilot/monitoring-usage-and-entitlements/about-premium-requests) for details.

Known limitations:

- `ai_task.generate_image` is not supported yet.
- Image attachments must fit the model's limits (usually 1–5 images of up to about 3 MB each).
- Gemini models don't enforce structured output natively. They usually still return valid JSON, but stricter models are more reliable for structured tasks.

## Development

```bash
uv venv --python 3.14 .venv
uv pip install --python .venv/bin/python pytest-homeassistant-custom-component==0.13.372 github-copilot-sdk==1.0.22 ruff
.venv/bin/python -m pytest            # unit tests with a fake SDK
.venv/bin/ruff check . && .venv/bin/ruff format --check .

# Live tests against real Copilot (uses a few requests)
GITHUB_COPILOT_LIVE_TOKEN=github_pat_... .venv/bin/python -m pytest tests/test_live.py -s
```

## License

MIT
