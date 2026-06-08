# Frontend Reference Notes

## Source reviewed

The visual reference is the `frontend-reference/` folder from the public GitHub repository. It is a separate Gradio/custom-frontend app called **Road B: The Other Screen**.

## What was borrowed

Headline Booster borrows visual ideas only:

- a dark two-column layout;
- a wider previous-chats sidebar;
- a centered hero/logo area;
- a minimal lower composer;
- a cleaner app surface with the old top navigation removed.

## What was not copied

The Road B product logic was not copied. Headline Booster does not use:

- alternate-self game state;
- Echo Artifacts;
- souvenir cards;
- trace export;
- Modal endpoints;
- llama.cpp runtime yet;
- Road B prompts or narrative mechanics.

## How it is used in this app

The runtime app is still `app.py`. The visual layer lives in `frontend/styles.css` and is passed into Gradio at launch time. The app now follows the dark reference layout while keeping Gradio as the only frontend runtime.

## Future adaptation path

If more of the reference app is needed, copy only reusable static assets into `frontend/assets/`, then translate the look into Gradio HTML/CSS. Do not reintroduce a separate React/Vite build or external API dependency for the first Headline Booster version.
## CSS reference pass

The Hugging Face `chatbot_mail` CSS reference was reviewed for its practical frontend justification: explicit component targeting, full-width sidebar buttons, compact button heights, clear hover states, constrained main content width, and stronger `!important` overrides where the framework injects styles. Headline Booster applies the same idea in `frontend/styles.css` while translating it from Streamlit selectors to Gradio selectors.
