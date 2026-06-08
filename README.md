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
- Tiny model mode by default
- Local visual-development fallback via USE_REAL_MODEL=false
- No model selector in the public interface
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
- The chat shows `La IA está trabajando...` while the model prepares the answer.
- The frontend no longer shows example buttons or a model selector, keeping the screen focused on one chat input.
- A lightweight JSON history is saved locally per Gradio session/user namespace and cleared with `+ Nuevo chat`; no login, database, payments, or external API is required.

## Performance defaults

The app uses faster defaults for the first public Space version:

- `MAX_NEW_TOKENS=280` keeps the tiny model focused on headline output instead of long generations.
- The old character-by-character streaming helper was removed; the app now swaps a working message for the final answer.
- Use `USE_REAL_MODEL=false` only when iterating on visual design locally.

## Tiny Titan model mode

The app uses the tiny model path by default. For explicit Tiny Titan configuration, set:

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
