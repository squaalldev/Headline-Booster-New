---
license: mit
title: ' Headline-booster'
sdk: gradio
emoji: 💻
colorFrom: yellow
colorTo: yellow
pinned: true
sdk_version: 6.17.3
---
# Headline Booster

Headline Booster is a small-model-ready Gradio chatbot that helps entrepreneurs create clearer, stronger headlines in seconds.

## What it does

The user gives four simple pieces of information:
1. What they sell
2. Who it is for
3. The result the audience wants
4. How many headlines they want

The app then generates persuasive headline options.

## How to run locally

```bash
pip install -r requirements.txt
python app.py
```

## Hackathon alignment

- Built with Gradio
- Designed for Hugging Face Spaces
- Prepared for a small model under 32B parameters
- Tiny Titan target model: Qwen/Qwen2.5-1.5B-Instruct
- Quality fallback model: Qwen/Qwen2.5-3B-Instruct
- Mock mode by default for local development
- Tiny model mode by default on Hugging Face Spaces
- Real model mode via USE_REAL_MODEL=true or UI runtime selector
- ZeroGPU integration prepared

## Architecture

```text
.
├── app.py
├── requirements.txt
├── README.md
├── frontend/
│   ├── README.md
│   └── styles.css
└── docs/
    ├── CODEX_NOTES.md
    ├── COMMIT_LOG.md
    ├── FIELD_NOTES.md
    ├── FRONTEND_REFERENCE_NOTES.md
    └── TINY_TITAN_PLAN.md
```

## Chatbot behavior

Headline Booster now behaves like a focused chatbot instead of a static generator:

- Short greetings like `hola` start the conversation and ask only for the four required data points.
- Responses stream into the chat area so the user sees the assistant writing instead of waiting for a static block.
- Sidebar examples are clickable chat turns that fill the conversation with real headline requests.
- A lightweight JSON history is saved locally for the current namespace and cleared with `+ Nueva conversación`; no login, database, payments, or external API is required.

## Performance defaults

The app uses faster defaults for the first public Space version:

- `MAX_NEW_TOKENS=280` keeps the tiny model focused on headline output instead of long generations.
- `STREAM_BATCH_SIZE=48` reduces the number of Gradio UI updates while keeping a lightweight streaming feel.
- Use `USE_REAL_MODEL=false` when iterating on visual design and `USE_REAL_MODEL=true` only when testing model quality.

## Tiny Titan model mode

The app defaults to mock mode locally, but defaults to the tiny model on Hugging Face Spaces when Space environment variables are present. To force the Tiny Titan model path, configure:

```bash
USE_REAL_MODEL=true
MODEL_ID=Qwen/Qwen2.5-1.5B-Instruct
```

If the 1.5B model needs stronger copywriting quality, use the under-4B fallback:

```bash
MODEL_ID=Qwen/Qwen2.5-3B-Instruct
```

## Source code

GitHub repo: TODO

## Built with Codex

This app was built with help from OpenAI Codex as a coding agent.

Codex migration evidence is documented in `docs/CODEX_NOTES.md` and `docs/COMMIT_LOG.md`. Visual adaptation notes are documented in `docs/FRONTEND_REFERENCE_NOTES.md`.
