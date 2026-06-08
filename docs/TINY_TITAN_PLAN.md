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

Headline generation is a narrow task. The app constrains the input to four details and constrains the output to a fixed Markdown format, which makes it a good fit for a tiny model.

## Step 1 completed

- Added model runtime configuration in `app.py`.
- Added `build_headline_prompt` for constrained Spanish generation.
- Added cached Hugging Face model loading.
- Added a visible Gradio runtime selector so users can choose the tiny model instead of silently staying in mock mode.
- Added a `generate_headlines` wrapper that chooses real model or mock fallback from the UI runtime selector.
- Added Tiny Titan dependencies in `requirements.txt`.

## Performance defaults

For the first public Tiny Titan path, the app limits generation to `MAX_NEW_TOKENS=280` by default and streams in `STREAM_BATCH_SIZE=48` character batches. These values reduce cold-start perceived latency and avoid over-generating for a narrow headline task.
