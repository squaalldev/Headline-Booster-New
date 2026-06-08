# Codex Notes

## How Codex helped

Codex reviewed the existing repository, identified the current React/FastAPI/Supabase/Groq architecture, and converted the project into a focused Gradio app for Hugging Face Spaces.

## Repository analysis

The original repository was an English-learning chatbot named COMPARTIR AI. It used a React/Vite frontend, Tailwind custom styling, FastAPI backend routes, Supabase authentication/session storage, Groq/OpenAI-related AI dependencies, and multiple chat components for a full learning product.

The useful visual ideas were the left sidebar, central chat area, soft brown palette, rounded message bubbles, and clean chatbot layout. The old business logic was not aligned with Headline Booster.

## What was reused

- The product-level idea of a sidebar plus main chat workspace.
- The warm brown visual direction from the previous Tailwind chat styles.
- The simple user/right and assistant/left chat experience.

## What was removed

- React/Vite application files. A lightweight `frontend/` folder was added back only for Gradio CSS assets.
- FastAPI backend files.
- Supabase authentication and protected routes.
- Groq, OpenAI, Google/Gemini-style external AI dependencies.
- English-learning services: translation, grammar correction, vocabulary, roadmap, personalities, speech, login, usage counters, and advanced session flows.
- Screenshots from the previous product.

## Files created or refactored

- `app.py`: new Gradio Blocks application with sidebar, chat UI, mock headline generation, style adjustment detection, and reset behavior.
- `frontend/styles.css`: custom visual layer for the Gradio interface, kept separate from Python so the frontend can be adjusted without reintroducing React/Vite.
- `requirements.txt`: includes Gradio plus optional Tiny Titan runtime dependencies for Transformers and ZeroGPU.
- `README.md`: rewritten for Headline Booster and hackathon alignment.
- `docs/CODEX_NOTES.md`: documents repository analysis and migration decisions.
- `docs/FIELD_NOTES.md`: documents product problem, small-model angle, and future work.
- `docs/COMMIT_LOG.md`: records the relevant Git commits for hackathon/Codex review.
- `docs/FRONTEND_REFERENCE_NOTES.md`: explains how the Road B frontend reference was used as visual inspiration without copying product logic.
- `docs/TINY_TITAN_PLAN.md`: documents the Qwen2.5-1.5B real-model path and fallback plan.

## Chatbot interaction update

Codex adapted the useful behavior from the Streamlit reference without reintroducing Streamlit or external APIs: short greetings now start the three-question flow, the UI shows a working message while the model runs, example buttons and the model selector were removed, and a small JSON file stores each user/session history until that user starts a new chat.

## Current status

Headline Booster is now a simple Gradio app that can run with `python app.py`. It uses the tiny model path by default, keeps `USE_REAL_MODEL=false` as a local visual-development fallback, and no longer exposes a runtime selector in the frontend. It asks for only three data points when input is incomplete and returns Spanish headline options when the request is complete.

## Next steps

- Deploy the repository as a Hugging Face Space.
- Test the `USE_REAL_MODEL=true` path on Hugging Face ZeroGPU.
- Add light manual tests for Spanish headline quality and tone adjustment behavior.

## Planned model integration

The Tiny Titan target is `Qwen/Qwen2.5-1.5B-Instruct`, with `Qwen/Qwen2.5-3B-Instruct` as the quality fallback. Both keep Headline Booster under the hackathon tiny-model direction while preserving a mock fallback for local development.
