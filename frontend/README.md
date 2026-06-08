# Frontend assets

This folder contains the lightweight frontend assets used by the Gradio Space.

The app is still launched from the repository root with:

```bash
python app.py
```

`styles.css` holds the custom visual layer for Headline Booster: sidebar, topbar, welcome avatar, chat bubbles, responsive layout, and composer styling.

## Visual direction

The current styling is inspired by the public `frontend-reference/` app, but only as a visual reference. Headline Booster keeps its own Gradio runtime and headline-generation flow; it does not copy the reference app's product logic, model runtime, Modal endpoint, artifacts, or trace export features.
