---
license: mit
title: Headline Booster
sdk: gradio
emoji: 🏆
colorFrom: yellow
colorTo: red
pinned: true
short_description: Make titles with AI
sdk_version: 6.16.0
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
- Planned model: Qwen2.5-3B-Instruct
- No external AI APIs in the first version
- ZeroGPU integration planned

## Architecture

```text
.
├── app.py
├── requirements.txt
├── README.md
└── docs/
    ├── CODEX_NOTES.md
    ├── COMMIT_LOG.md
    └── FIELD_NOTES.md
```

## Source code

GitHub repo: TODO

## Built with Codex

This app was built with help from OpenAI Codex as a coding agent.

Codex migration evidence is documented in `docs/CODEX_NOTES.md` and `docs/COMMIT_LOG.md`.
