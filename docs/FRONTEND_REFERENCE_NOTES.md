# Frontend Reference Notes

## Source reviewed

The visual reference is the `frontend-reference/` folder from the public GitHub repository. It is a separate Gradio/custom-frontend app called **Road B: The Other Screen**.

## What was borrowed

Headline Booster borrows visual ideas only:

- cinematic "signal" language;
- a darker premium sidebar;
- pill navigation in the topbar;
- a portal-style hero/avatar;
- compact proof/metric cards;
- a more polished chat chamber and composer.

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

The runtime app is still `app.py`. The visual layer lives in `frontend/styles.css` and is passed into Gradio at launch time. The `frontend-reference/` folder should remain a reference source only unless specific assets are intentionally copied into `frontend/assets/`.

## Future adaptation path

If more of the reference app is needed, copy only reusable static assets into `frontend/assets/`, then translate the look into Gradio HTML/CSS. Do not reintroduce a separate React/Vite build or external API dependency for the first Headline Booster version.
