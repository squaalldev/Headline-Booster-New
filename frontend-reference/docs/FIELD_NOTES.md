# Field Notes — Road B: The Other Screen

## What I built

Road B is a small-model interactive fiction app about alternate life paths.

The user describes a decision point: the city they did not move to, the job they did not take, the person they did not tell, the path they still wonder about.

The app then opens **The Other Screen** and generates a fictional version of the user who chose differently.

The goal is not prediction. It is not therapy. It is not advice. It is a poetic speculative mirror.

## Final experience

The final app is structured as a limited-signal ritual:

```text
Name the fork
→ Tune the Other Screen
→ Meet Road B
→ Ask questions
→ Collect Echo Artifacts
→ Unlock Final Transmission
→ Download a Souvenir Card
```

The Echo Artifacts are:

- Cost Ledger
- Beauty Ledger
- A Typical Tuesday
- The Unsent Letter
- The Moment It Split

These make Road B feel less like a normal chatbot and more like a strange little story game.

## Why AI is load-bearing

The AI is not decorative.

Qwen generates:

- the alternate self
- the opening transmission
- the daily scene
- Road B’s replies
- Echo Artifacts
- reflective insights
- the Final Transmission
- content for the Souvenir Card

Without the model, Road B has no experience.

## Runtime architecture

The prototype first attempted to run Qwen GGUF directly inside the Hugging Face Space.

That worked slowly on CPU, but it was not smooth enough for a polished hackathon demo. ZeroGPU was also not a good fit for the llama.cpp/GGUF runtime because Road B is not a PyTorch/Transformers app.

The final architecture is:

```text
Hugging Face Space:
- Gradio app
- custom cinematic frontend
- session state
- Echo Artifacts game loop
- souvenir card rendering
- trace export

Modal GPU:
- Qwen3.5-9B GGUF
- llama-cpp-python
- llama.cpp inference endpoint
```

This keeps Road B as a Hugging Face Gradio Space while using Modal sponsor GPU credits for reliable model inference.

## Model choice

Road B uses:

```text
unsloth/Qwen3.5-9B-GGUF
Qwen3.5-9B-Q4_K_M.gguf
llama-cpp-python
```

I chose this model/runtime because:

- it is within the hackathon small-model limit
- it works with GGUF
- it supports llama.cpp
- it can produce vivid fictional prose
- it keeps the model role narrow and focused

## Prompt design

The system prompt frames Qwen as the narrative engine for Road B.

The model is instructed to:

- write fictional alternate-life transmissions
- avoid prediction
- avoid ranking Road A and Road B
- avoid giving life advice
- keep every gain tied to a cost
- use concrete sensory details
- return structured JSON internally

The user should not see raw JSON. The backend parses model output and the frontend renders clean story text, artifacts, and cards.

## UI design

The interface is designed as a fictional machine rather than a standard chatbot.

The UI includes:

- dark cinematic portal hero
- dystopic Road B icon
- signal/tuning language
- alternate-self chat chamber
- Echo Artifact buttons
- Final Transmission unlock
- visible Souvenir Card
- download card action
- public trace support

The goal is to make the user feel like they are tuning into another life, not filling out a form.

## What changed during development

Early versions were closer to a normal Gradio app.

The project then moved toward a custom cinematic frontend. The key lesson was that the UI must make the AI feel load-bearing. A generic chat box makes the model feel optional. A fictional machine makes the model feel necessary.

Later, Echo Artifacts were added to make the experience more game-like and less like plain chat.

Finally, inference moved to Modal GPU to make the app fast enough for judging.

## Badge targets

Road B targets:

- Off-Brand: custom cinematic UI
- Llama Champion: llama.cpp runtime
- Sharing is Caring: public synthetic trace
- Field Notes: this report

Road B does not claim Off the Grid in the final Modal version because inference runs on Modal GPU compute.

## Limitations

Road B is fiction.

It should not be used for:

- mental health support
- crisis support
- medical advice
- legal advice
- financial advice
- real-life decision prediction

The app is a reflective story machine, not a tool for deciding what someone should do.

## What I learned

Small models become more compelling when they are given:

- a narrow role
- a strong voice
- a world to inhabit
- a visible interaction loop
- artifacts the user can keep

Road B works best when it feels like a signal from another path.