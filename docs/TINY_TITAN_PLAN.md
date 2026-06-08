# Tiny Titan Plan

## Target model

Headline Booster now has a real-model path prepared for:

```text
Qwen/Qwen2.5-1.5B-Instruct
```

This is the first Tiny Titan target because it is small enough for the challenge while still being an instruction-tuned multilingual model suitable for Spanish headline generation.

## Quality fallback

If the 1.5B model is not strong enough for Spanish copywriting quality, use:

```text
Qwen/Qwen2.5-3B-Instruct
```

This keeps the app under the 4B Tiny Titan target while improving output quality.

## Runtime switch

The app defaults to mock mode for safe local development, and defaults to the tiny model automatically on Hugging Face Spaces when Space environment variables are present:

```bash
python app.py
```

To force the real model runtime in any environment, set:

```bash
USE_REAL_MODEL=true
MODEL_ID=Qwen/Qwen2.5-1.5B-Instruct
```

Optional quality fallback:

```bash
MODEL_ID=Qwen/Qwen2.5-3B-Instruct
```

## Why this fits the app

Headline generation is a narrow task. The app constrains the input to three details and constrains the output to a fixed Markdown format, which makes it a good fit for a tiny model.

## Step 1 completed

- Added model runtime configuration in `app.py`.
- Added `build_headline_prompt` for constrained Spanish generation.
- Added cached Hugging Face model loading.
- Removed the visible runtime selector so the public interface always behaves like an AI-model chat.
- Kept a `generate_headlines` wrapper so local development can still use `USE_REAL_MODEL=false` without exposing that choice in the UI.
- Added Tiny Titan dependencies in `requirements.txt`.

## Performance defaults

For the first public Tiny Titan path, the app generates 5 headlines by default, limits generation to `MAX_NEW_TOKENS=280`, and removes character-by-character response streaming. The UI now shows a working message while the model runs, then swaps in the final answer.
