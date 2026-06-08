---
title: "Road B: The Other Screen"
emoji: "🪞"
colorFrom: purple
colorTo: indigo
sdk: gradio
sdk_version: "6.16.0"
python_version: "3.12"
app_file: app.py
fullWidth: true
header: mini
short_description: "Talk to the self who chose differently."
models:
  - unsloth/Qwen3.5-9B-GGUF
tags:
  - gradio
  - gradio-server
  - modal
  - llama-cpp
  - gguf
  - qwen
  - interactive-fiction
  - custom-frontend
  - small-models
  - build-small-hackathon
thumbnail: thumbnail.png
license: mit
---

# Road B: The Other Screen

**Talk to a fictional version of yourself who chose differently.**

Road B is a small-model interactive fiction experience for the Build Small Hackathon, Chapter Two: **An Adventure in Thousand Token Wood**.

You name a fork in your life. The app opens a cinematic **Other Screen** and lets you speak with a fictional alternate self: the version of you who took Road B.

It is not prediction, therapy, or advice. It is a strange mirror.

## What the app does

Road B turns a life fork into a limited-signal ritual.

The user can:

- describe a decision point
- invoke a fictional Road B self
- chat with that alternate self
- collect Echo Artifacts
- unlock a Final Transmission
- download a souvenir card from the road not taken
- export a synthetic-style trace of the interaction

The core loop is:

```text
Name the fork
→ Tune the Other Screen
→ Meet Road B
→ Collect Echo Artifacts
→ Unlock Final Transmission
→ Save the Souvenir Card
```

## Echo Artifacts

Road B is not just a chat app. After the first transmission, the user can collect artifacts from the alternate life:

- **Cost Ledger**
- **Beauty Ledger**
- **A Typical Tuesday**
- **The Unsent Letter**
- **The Moment It Split**

After three Echo Artifacts, the **Final Transmission** unlocks.

## Runtime

Road B is hosted as a Hugging Face Gradio Space.

The Hugging Face Space handles:

- custom cinematic UI
- Gradio app shell
- Road B session state
- Echo Artifacts game loop
- souvenir card rendering
- trace export
- public Space hosting

Model inference runs on Modal GPU using:

- `unsloth/Qwen3.5-9B-GGUF`
- `Qwen3.5-9B-Q4_K_M.gguf`
- `llama-cpp-python`
- llama.cpp runtime

The Hugging Face Space calls our Modal endpoint for model inference. The Modal endpoint runs the GGUF model through llama.cpp.

## Model

```text
Model repo: unsloth/Qwen3.5-9B-GGUF
Model file: Qwen3.5-9B-Q4_K_M.gguf
Runtime: Modal GPU + llama.cpp via llama-cpp-python
Mock mode: false
```

The model is 9B parameters, within the hackathon’s small-model limit.

## Why Road B fits the judging criteria

### Genuinely delightful

Road B feels like a small strange machine: portal hero, signal chamber, alternate-self chat, Echo Artifacts, Final Transmission, and a downloadable souvenir card.

### AI is load-bearing

Without the model, there is no alternate self, no transmission, no artifact, no final message, and no meaningful souvenir card.

The AI is the experience.

### Originality of concept

Road B is not a generic chatbot. It is a fictional machine for talking to the life beside yours.

### Polish of the Gradio app

The app uses a custom cinematic frontend, active navigation, Echo Artifacts, visible souvenir card, trace export, and Modal-powered Qwen inference for smoother judging.

## Bonus badge proof

### Off-Brand

Road B uses a custom cinematic frontend instead of the default Gradio look.

It includes:

- portal hero
- animated signal atmosphere
- alternate-self chat chamber
- Echo Artifacts
- Final Transmission unlock
- downloadable souvenir card
- active navigation and menu actions

### Llama Champion

The model runs through llama.cpp using `llama-cpp-python`.

The llama.cpp runtime runs on Modal GPU.

### Sharing is Caring

A synthetic public trace is included here:

`samples/public_trace_sample.json`

The live app also supports trace export.

### Field Notes

A short build report is included here:

`docs/FIELD_NOTES.md`

### Off the Grid note

Road B does **not** claim the Off the Grid bonus in the final Modal version.

The app uses an open GGUF model through llama.cpp, but inference runs on Modal GPU compute rather than fully inside the Hugging Face Space.

## Files

```text
app.py
index.html
README.md
requirements.txt
assets/favicon.svg
assets/hero-reference.png
docs/FIELD_NOTES.md
samples/public_trace_sample.json
thumbnail.png
```

## Environment variables

The Hugging Face Space expects these secrets:

```text
MODAL_QWEN_URL
MODAL_QWEN_TOKEN
MODAL_TIMEOUT
```

Recommended values:

```text
MODAL_TIMEOUT=900
MAX_TOKENS=850
MODEL_FILENAME=Qwen3.5-9B-Q4_K_M.gguf
```

## Health check

The running app exposes:

```text
/health
```

A healthy Modal configuration should show:

```json
{
  "runtime": "Modal GPU + llama.cpp",
  "modal_qwen_enabled": true,
  "modal_qwen_url_set": true,
  "mock_mode": false
}
```

## Safety and privacy note

Road B is speculative interactive fiction. It is not a medical, legal, financial, psychological, or crisis-support tool.

User text is sent from the browser to the Hugging Face Space backend, then to the project’s Modal inference endpoint so the Qwen GGUF model can generate a response.

The public trace in `samples/public_trace_sample.json` is synthetic and does not contain real user data.