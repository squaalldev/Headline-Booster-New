"""Headline Booster: a small-model-ready Gradio chatbot for Spanish headlines.

The app can run in mock mode by default or use a local Hugging Face model
when USE_REAL_MODEL=true. The first Tiny Titan target is
Qwen/Qwen2.5-1.5B-Instruct, with Qwen/Qwen2.5-3B-Instruct as the quality fallback.
"""

from __future__ import annotations

import os
import re
from functools import lru_cache
from pathlib import Path
from typing import Any

import gradio as gr


MISSING_INFO_MESSAGE = """Claro. Para crear mejores encabezados necesito solo 4 datos:

1. ¿Qué vendes?
2. ¿Para quién es?
3. ¿Qué resultado quiere conseguir esa persona?
4. ¿Cuántos encabezados quieres?"""

STYLE_LABELS = {
    "default": "claros y persuasivos",
    "emotional": "más emocionales",
    "direct": "más directos",
    "elegant": "más elegantes",
    "curious": "más curiosos",
}

STYLE_HEADLINES = {
    "default": [
        "Deja de decidir desde la duda y empieza a elegir con claridad",
        "Tu cuerpo ya sabe qué decisión tomar. Aprende a escucharlo",
        "Si cada decisión te agota, quizá no estás decidiendo desde ti",
        "Toma decisiones con más confianza sin exigirte funcionar como los demás",
        "Descubre una forma más clara de entender tu energía antes de elegir",
        "Elige con más calma, claridad y confianza desde tu propio diseño",
        "Si tomar decisiones te pesa, quizá necesitas una forma distinta de escucharte",
        "Aprende a reconocer cuándo una decisión sí es para ti",
        "Deja de compararte y empieza a tomar decisiones desde tu energía",
        "Tu claridad no está afuera: empieza a escucharla desde ti",
    ],
    "emotional": [
        "Vuelve a confiar en esa voz interna que sabe lo que necesitas elegir",
        "Deja de sentirte perdida cada vez que una decisión importante aparece",
        "Tu claridad merece más espacio que tus dudas",
        "Elige desde tu energía y recupera la calma que tanto necesitas",
        "No estás confundida: solo necesitas una forma más tuya de decidir",
        "Convierte la presión de decidir en una sensación real de confianza",
        "Escúchate con más amor antes de exigirte respuestas perfectas",
        "Cuando decides desde ti, la duda deja de mandar",
        "Encuentra paz en tus decisiones sin compararte con nadie más",
        "La claridad que buscas puede empezar en tu propio cuerpo",
    ],
    "direct": [
        "Aprende a tomar mejores decisiones usando tu Diseño Humano",
        "Decide con más claridad sin depender de consejos externos",
        "Un taller práctico para emprendedoras que quieren elegir con confianza",
        "Usa tu energía para saber qué decisión tomar",
        "Deja de dudar y empieza a decidir con un método claro",
        "Toma decisiones de negocio alineadas con tu forma de funcionar",
        "Descubre cómo decidir con más calma en menos tiempo",
        "Convierte tu Diseño Humano en una guía práctica para elegir",
        "Mejora tus decisiones entendiendo tu energía personal",
        "Decide con seguridad sin copiar la estrategia de otras personas",
    ],
    "elegant": [
        "Una forma más consciente de decidir desde tu propia naturaleza",
        "Claridad interior para emprendedoras que desean elegir con confianza",
        "Descubre el arte de tomar decisiones alineadas con tu energía",
        "Decisiones más serenas para una vida y un negocio más coherentes",
        "Aprende a escuchar tu diseño antes de elegir tu siguiente paso",
        "Una guía íntima para decidir con calma, presencia y certeza",
        "Eleva tu manera de decidir comprendiendo cómo funciona tu energía",
        "Elige con sobriedad, claridad y respeto por tu propio ritmo",
        "Tu diseño puede convertirse en brújula para decisiones más conscientes",
        "Menos ruido externo. Más claridad desde tu propia energía",
    ],
    "curious": [
        "¿Y si tu cuerpo ya supiera qué decisión tomar?",
        "La razón por la que decidir te agota quizá no es la que crees",
        "¿Estás decidiendo desde tu energía o desde la presión?",
        "El mapa interno que puede cambiar cómo eliges en tu negocio",
        "Antes de pedir otro consejo, escucha esta señal de tu diseño",
        "¿Por qué algunas decisiones se sienten tan pesadas para ti?",
        "La forma de decidir que quizá nadie te enseñó",
        "¿Y si la claridad no estuviera en pensar más?",
        "Lo que tu Diseño Humano puede revelar antes de tu próxima decisión",
        "¿Tus decisiones son tuyas o aprendiste a copiarlas?",
    ],
}

CUSTOM_CSS = (Path(__file__).parent / "frontend" / "styles.css").read_text(encoding="utf-8")

MODEL_ID = os.getenv("MODEL_ID", "Qwen/Qwen2.5-1.5B-Instruct")
USE_REAL_MODEL = os.getenv("USE_REAL_MODEL", "false").lower() == "true"
REAL_MODEL_MAX_NEW_TOKENS = int(os.getenv("MAX_NEW_TOKENS", "700"))
RUNTIME_LABEL = "real model" if USE_REAL_MODEL else "mock mode"
MODEL_BADGE = MODEL_ID.split("/")[-1] if USE_REAL_MODEL else "small-model ready"
TINY_TITAN_TARGET = "Qwen/Qwen2.5-1.5B-Instruct"


def _normalize(text: str) -> str:
    return (text or "").strip().lower()


def _extract_count(user_message: str) -> int:
    text = _normalize(user_message)
    match = re.search(r"\b(\d{1,2})\b", text)
    if match:
        return max(1, min(int(match.group(1)), 20))

    number_words = {
        "uno": 1,
        "una": 1,
        "dos": 2,
        "tres": 3,
        "cuatro": 4,
        "cinco": 5,
        "seis": 6,
        "siete": 7,
        "ocho": 8,
        "nueve": 9,
        "diez": 10,
        "once": 11,
        "doce": 12,
    }
    for word, value in number_words.items():
        if re.search(rf"\b{word}\b", text):
            return value
    return 10


def is_request_complete(user_message: str) -> bool:
    """
    Detecta si el usuario ya dio suficiente información.
    Debe usar reglas simples.
    """
    text = _normalize(user_message)
    if len(text.split()) < 10:
        return False

    has_offer = any(
        token in text
        for token in [
            "vendo",
            "ofrezco",
            "tengo",
            "lanzaré",
            "lanzo",
            "curso",
            "taller",
            "mentoría",
            "programa",
            "servicio",
            "web",
        ]
    )
    has_audience = any(
        token in text
        for token in [
            "para ",
            "dirigido a",
            "emprendedoras",
            "emprendedores",
            "coaches",
            "mujeres",
            "dueños",
            "personas que",
        ]
    )
    has_result = any(
        token in text
        for token in [
            "quieren",
            "quiere",
            "buscan",
            "busca",
            "conseguir",
            "lograr",
            "resultado",
            "claridad",
            "confianza",
            "vender",
            "mejorar",
            "tomar decisiones",
        ]
    )
    has_count = bool(re.search(r"\b\d{1,2}\b", text)) or any(
        re.search(rf"\b{word}\b", text)
        for word in ["cinco", "seis", "siete", "ocho", "nueve", "diez", "once", "doce"]
    )

    return has_offer and has_audience and has_result and has_count


def detect_style_request(user_message: str) -> str:
    """
    Detecta si el usuario pide un ajuste:
    - default
    - emotional
    - direct
    - elegant
    - curious
    """
    text = _normalize(user_message)
    if any(word in text for word in ["emocional", "emocionales", "emotivo", "emotivos", "sentimiento"]):
        return "emotional"
    if any(word in text for word in ["directo", "directos", "claro", "claros", "concreto", "concretos"]):
        return "direct"
    if any(word in text for word in ["elegante", "elegantes", "premium", "sofisticado", "sofisticados"]):
        return "elegant"
    if any(word in text for word in ["curioso", "curiosos", "curiosidad", "intriga", "pregunta"]):
        return "curious"
    return "default"


def generate_headlines_mock(user_message: str, style: str = "default") -> str:
    """
    Genera encabezados simulados en Markdown.
    Luego esta función será reemplazada por inferencia real con un modelo pequeño.
    """
    style = style if style in STYLE_HEADLINES else "default"
    count = _extract_count(user_message)
    headlines = STYLE_HEADLINES[style]
    selected = [headlines[index % len(headlines)] for index in range(count)]
    numbered_headlines = "\n".join(
        f"{index}. {headline}" for index, headline in enumerate(selected, start=1)
    )
    tone_note = STYLE_LABELS[style]

    return f"""## Encabezados generados

{numbered_headlines}

## Mi recomendación

El encabezado #2 es el más fuerte porque combina curiosidad, claridad y una promesa emocional.

## Por qué funciona

Estos encabezados son {tone_note}: conectan la oferta con el resultado deseado y evitan explicar demasiado en la primera línea.

_Nota técnica: esta es una generación mock. Activa USE_REAL_MODEL=true para usar el modelo tiny configurado con MODEL_ID._"""


def build_headline_prompt(user_message: str, style: str = "default") -> list[dict[str, str]]:
    """Builds the constrained Spanish prompt used by the tiny model runtime."""
    count = _extract_count(user_message)
    style_instruction = {
        "default": "claros, persuasivos y listos para usar",
        "emotional": "más emocionales, conectados con el deseo interno de la audiencia",
        "direct": "más directos, concretos y orientados a conversión",
        "elegant": "más elegantes, sobrios y premium",
        "curious": "más curiosos, con intriga y tensión",
    }.get(style, STYLE_LABELS["default"])

    system_prompt = """
Eres Headline Booster, un asistente de copywriting en español.

Tu única tarea es crear encabezados claros y persuasivos.
No hagas diagnóstico.
No hagas estrategia.
No escribas emails.
No agregues pasos extra.
No pidas más datos si el usuario ya dio qué vende, para quién es, resultado deseado y cantidad.

Devuelve siempre Markdown con este formato exacto:

## Encabezados generados

1. ...
2. ...
3. ...

## Mi recomendación

...

## Por qué funciona

...
""".strip()

    user_prompt = f"""
Solicitud del usuario:
{user_message}

Estilo solicitado:
{style_instruction}

Cantidad de encabezados solicitada:
{count}

Genera exactamente {count} encabezados en español.
""".strip()

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]


@lru_cache(maxsize=1)
def get_tiny_model():
    """Loads the configured Hugging Face model once per Space process."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype="auto",
        device_map="auto",
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer, model


def _zerogpu_enabled(func):
    """Applies the ZeroGPU decorator only when the real model runtime is enabled."""
    if USE_REAL_MODEL:
        import spaces

        return spaces.GPU(duration=90)(func)
    return func


@_zerogpu_enabled
def generate_headlines_model(user_message: str, style: str = "default") -> str:
    """Generates headlines with the configured tiny local model."""
    import torch

    tokenizer, model = get_tiny_model()
    messages = build_headline_prompt(user_message, style)
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = tokenizer([prompt], return_tensors="pt")
    device = next(model.parameters()).device
    inputs = {key: value.to(device) for key, value in inputs.items()}

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=REAL_MODEL_MAX_NEW_TOKENS,
            temperature=0.72,
            top_p=0.9,
            do_sample=True,
            repetition_penalty=1.08,
            pad_token_id=tokenizer.eos_token_id,
        )

    generated_tokens = outputs[0][inputs["input_ids"].shape[-1] :]
    generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
    if not generated_text:
        return generate_headlines_mock(user_message, style)
    return generated_text


def generate_headlines(user_message: str, style: str = "default") -> str:
    """Routes generation to the real tiny model or the local mock fallback."""
    if USE_REAL_MODEL:
        return generate_headlines_model(user_message, style)
    return generate_headlines_mock(user_message, style)


def _last_complete_user_request(history: list[dict[str, Any]]) -> str:
    for message in reversed(history or []):
        if message.get("role") == "user" and is_request_complete(message.get("content", "")):
            return message.get("content", "")
    return ""


def chat_response(message, history):
    """
    Maneja la conversación.
    Si falta información, pide solo 4 datos.
    Si hay información suficiente, genera encabezados con el modelo configurado o fallback mock.
    Si el usuario pide ajuste de tono, genera una nueva versión con el mismo runtime.
    """
    history = history or []
    clean_message = (message or "").strip()
    if not clean_message:
        return history, "", gr.update(visible=len(history) == 0), history

    style = detect_style_request(clean_message)
    previous_request = _last_complete_user_request(history)
    is_style_adjustment = style != "default" and previous_request and not is_request_complete(clean_message)

    if is_style_adjustment:
        bot_message = generate_headlines(previous_request, style=style)
    elif is_request_complete(clean_message):
        bot_message = generate_headlines(clean_message, style=style)
    else:
        bot_message = MISSING_INFO_MESSAGE

    updated_history = [
        *history,
        {"role": "user", "content": clean_message},
        {"role": "assistant", "content": bot_message},
    ]
    return updated_history, "", gr.update(visible=False), updated_history


def reset_chat():
    """
    Limpia la conversación.
    """
    return [], "", gr.update(visible=True), []


def build_app() -> gr.Blocks:
    with gr.Blocks(title="Headline Booster") as demo:
        chat_state = gr.State([])
        with gr.Column(elem_id="headline-app"):
            with gr.Row(elem_id="headline-layout"):
                with gr.Column(elem_id="sidebar", scale=0, min_width=280):
                    gr.HTML(
                        """
                        <div class="brand-mark">HB</div>
                        <div class="brand-kicker">headline lab · gradio</div>
                        <h1>Headline Booster</h1>
                        <p>Encabezados claros. Copy que vende.</p>
                        """
                    )
                    new_chat = gr.Button("+ Nueva conversación", elem_id="new-chat", size="lg")
                    gr.HTML(
                        f"""
                        <div class="signal-panel">
                          <div class="signal-label">Signal controls</div>
                          <div class="signal-row">
                            <span class="signal-pill">{RUNTIME_LABEL}</span>
                            <span class="signal-pill">{MODEL_BADGE}</span>
                          </div>
                          <div class="signal-meter"><div class="signal-fill"></div></div>
                        </div>
                        <div class="sidebar-card">
                          <div class="sidebar-title">Conversaciones de ejemplo</div>
                          <div class="example-convo">Taller de Diseño Humano</div>
                          <div class="example-convo">Curso de Copywriting</div>
                          <div class="example-convo">Web para Coaches</div>
                          <div class="example-convo">Mentoría para Emprendedoras</div>
                        </div>
                        """
                    )
                with gr.Column(elem_id="main-panel", scale=1):
                    gr.HTML(
                        """
                        <div class="topbar">
                          <div class="top-left">
                            <h2>Headline Booster</h2>
                            <div class="nav-pills">
                              <span class="nav-pill">Generador</span>
                              <span class="nav-pill">Ejemplos</span>
                              <span class="nav-pill">Notas</span>
                            </div>
                          </div>
                          <div class="status-wrap">
                            <span><span class="status-dot">●</span> disponible</span>
                            <span class="settings-icon">⚙</span>
                          </div>
                        </div>
                        """
                    )
                    welcome = gr.HTML(
                        f"""
                        <section id="welcome-hero">
                          <div class="hero-kicker">portal de encabezados · 01</div>
                          <div class="portal-avatar" aria-label="Booster avatar"></div>
                          <h2>Booster</h2>
                          <h3>Crea encabezados persuasivos en segundos</h3>
                          <p>Dime qué vendes, para quién es, qué resultado prometes y cuántos encabezados quieres. Yo te devuelvo opciones claras, memorables y listas para probar.</p>
                          <div class="hero-actions">
                            <span class="hero-chip">Claro</span>
                            <span class="hero-chip">Persuasivo</span>
                            <span class="hero-chip">Listo para usar</span>
                          </div>
                          <div class="metric-grid">
                            <div class="metric-card"><strong>4 datos</strong><span>sin formularios largos</span></div>
                            <div class="metric-card"><strong>10 ideas</strong><span>{RUNTIME_LABEL}</span></div>
                            <div class="metric-card"><strong>1.5B target</strong><span>Tiny Titan</span></div>
                          </div>
                        </section>
                        """,
                        visible=True,
                    )
                    chatbot = gr.Chatbot(
                        value=[],
                        elem_id="chatbot",
                        show_label=False,
                        height=220,
                        avatar_images=(None, None),
                    )
                    with gr.Column(elem_classes="composer"):
                        with gr.Row():
                            message = gr.Textbox(
                                elem_id="message-input",
                                show_label=False,
                                lines=2,
                                max_lines=6,
                                placeholder="Ej. Vendo un taller de Diseño Humano para mujeres emprendedoras que quieren tomar decisiones con más claridad. Quiero 10 encabezados.",
                                scale=8,
                            )
                            send = gr.Button("Enviar", elem_id="send-btn", variant="primary", scale=1)
                        gr.HTML(f'<div class="helper-text"><span>Presiona Enter para enviar · Shift+Enter para nueva línea</span><span class="runtime-note">gradio · {RUNTIME_LABEL} · sin API externa</span></div>')

        message.submit(
            chat_response,
            inputs=[message, chat_state],
            outputs=[chatbot, message, welcome, chat_state],
        )
        send.click(
            chat_response,
            inputs=[message, chat_state],
            outputs=[chatbot, message, welcome, chat_state],
        )
        new_chat.click(reset_chat, outputs=[chatbot, message, welcome, chat_state])

    return demo


demo = build_app()


if __name__ == "__main__":
    demo.launch(css=CUSTOM_CSS, theme=gr.themes.Soft())
