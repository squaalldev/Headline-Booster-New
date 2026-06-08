"""Headline Booster: a small-model-ready Gradio chatbot for Spanish headlines.

The app can run in mock mode by default or use a local Hugging Face model
when USE_REAL_MODEL=true. The first Tiny Titan target is
Qwen/Qwen2.5-1.5B-Instruct, with Qwen/Qwen2.5-3B-Instruct as the quality fallback.
"""

from __future__ import annotations

import json
import os
import re
from functools import lru_cache
from pathlib import Path
from typing import Any

import gradio as gr


MISSING_INFO_MESSAGE = """Claro. Para crear mejores encabezados necesito solo 3 datos:

1. ¿Qué vendes?
2. ¿Para quién es?
3. ¿Qué resultado quiere conseguir esa persona?

Si no indicas cantidad, generaré 5 encabezados por defecto."""

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
DATA_DIR = Path(os.getenv("HEADLINE_BOOSTER_DATA_DIR", "data"))
CHAT_HISTORY_DIR = DATA_DIR / "chat_histories"
CHAT_NAMESPACE = os.getenv("CHATBOT_USER_NAMESPACE", "default_user")

MODEL_ID = os.getenv("MODEL_ID", "Qwen/Qwen2.5-1.5B-Instruct")
MODEL_FALLBACK_ID = os.getenv("MODEL_FALLBACK_ID", "Qwen/Qwen2.5-3B-Instruct")
REAL_MODEL_MAX_NEW_TOKENS = int(os.getenv("MAX_NEW_TOKENS", "280"))
TINY_TITAN_TARGET = "Qwen/Qwen2.5-1.5B-Instruct"
MODEL_RUNTIME = "Modelo tiny (Qwen 1.5B)"
MOCK_RUNTIME = "Mock local"
AUTO_RUNTIME = os.getenv("USE_REAL_MODEL", "auto").lower()
IS_HUGGING_FACE_SPACE = bool(os.getenv("SPACE_ID") or os.getenv("SPACE_HOST") or os.getenv("HF_SPACE_ID"))
DEFAULT_RUNTIME = MOCK_RUNTIME if AUTO_RUNTIME == "false" else MODEL_RUNTIME
GREETING_MESSAGE = f"""¡Perfecto! Puedo ayudarte a crear encabezados claros y persuasivos.

{MISSING_INFO_MESSAGE}"""


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
    return 5


def _history_path(namespace: str = CHAT_NAMESPACE) -> Path:
    safe_namespace = re.sub(r"[^a-zA-Z0-9_.-]", "_", namespace) or "default_user"
    return CHAT_HISTORY_DIR / f"{safe_namespace}.json"


def _request_namespace(request: gr.Request | None = None) -> str:
    """Returns a local per-user namespace for persisted chat history."""
    configured_namespace = os.getenv("CHATBOT_USER_NAMESPACE")
    if configured_namespace:
        return configured_namespace

    username = getattr(request, "username", None)
    if username:
        return f"user_{username}"

    session_hash = getattr(request, "session_hash", None)
    if session_hash:
        return f"user_{session_hash}"

    return CHAT_NAMESPACE


def load_chat_history(namespace: str = CHAT_NAMESPACE) -> list[dict[str, Any]]:
    """Loads a small single-user chat history from disk when available."""
    path = _history_path(namespace)
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return []
    if not isinstance(data, list):
        return []
    return [item for item in data if isinstance(item, dict) and "role" in item and "content" in item]


def save_chat_history(history: list[dict[str, Any]], namespace: str = CHAT_NAMESPACE) -> None:
    """Persists the latest simple chat history without adding databases or login."""
    try:
        CHAT_HISTORY_DIR.mkdir(parents=True, exist_ok=True)
        _history_path(namespace).write_text(
            json.dumps(history, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    except OSError:
        # Persistence should never break the chatbot on read-only or ephemeral runtimes.
        return


def clear_chat_history(namespace: str = CHAT_NAMESPACE) -> None:
    """Removes the persisted simple chat history for a new conversation."""
    path = _history_path(namespace)
    try:
        if path.exists():
            path.unlink()
    except OSError:
        return


def is_greeting(text: str, history: list[dict[str, Any]] | None = None) -> bool:
    """Detects a short first-message greeting and starts the three-data flow."""
    normalized = _normalize(text)
    greetings = [
        "hola",
        "hey",
        "saludos",
        "buenos días",
        "buenas tardes",
        "buenas noches",
        "hi",
        "hello",
    ]
    return (
        not history
        and len(normalized.split()) < 4
        and any(greeting in normalized for greeting in greetings)
    )


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
    return has_offer and has_audience and has_result


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
No pidas cantidad de encabezados. Si el usuario no indica cantidad, genera 5 encabezados.

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

Cantidad de encabezados:
{count}

Genera exactamente {count} encabezados en español. Si el usuario no pidió una cantidad explícita, usa 5.
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
    """Applies ZeroGPU when the Space has the `spaces` package available."""
    try:
        import spaces
    except ImportError:
        return func
    return spaces.GPU(duration=90)(func)


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


def generate_headlines(user_message: str, style: str = "default", runtime_mode: str = DEFAULT_RUNTIME) -> str:
    """Routes generation to the real tiny model or the local mock fallback."""
    if runtime_mode == MODEL_RUNTIME:
        try:
            return generate_headlines_model(user_message, style)
        except ImportError as error:
            return _model_runtime_error(
                "Faltan dependencias del modelo tiny",
                f"{error}",
            )
        except Exception as error:
            return _model_runtime_error(
                "No pude completar la inferencia con el modelo tiny",
                f"{type(error).__name__}: {error}",
            )
    return generate_headlines_mock(user_message, style)


def _model_runtime_error(title: str, detail: str) -> str:
    return f"""## {title}

Intenté usar el runtime real con `{MODEL_ID}`, pero el modelo no respondió correctamente en este entorno.

**Detalle técnico:** `{detail}`

Para activar la interacción real con el modelo en Hugging Face Spaces:

1. Asegúrate de tener hardware ZeroGPU o GPU disponible.
2. Instala las dependencias de `requirements.txt`.
3. Configura `USE_REAL_MODEL=true` o deja el valor por defecto para usar el modelo de IA.
4. Si necesitas más calidad, usa `MODEL_ID={MODEL_FALLBACK_ID}`.

Para desarrollo visual local puedes usar `USE_REAL_MODEL=false`, pero la interfaz pública no muestra selector de modelo."""


def _last_complete_user_request(history: list[dict[str, Any]]) -> str:
    for message in reversed(history or []):
        if message.get("role") == "user" and is_request_complete(message.get("content", "")):
            return message.get("content", "")
    return ""


def _resolve_bot_message(clean_message: str, history: list[dict[str, Any]], runtime_mode: str) -> str:
    style = detect_style_request(clean_message)
    previous_request = _last_complete_user_request(history)
    is_style_adjustment = style != "default" and previous_request and not is_request_complete(clean_message)

    if is_greeting(clean_message, history):
        return GREETING_MESSAGE
    if is_style_adjustment:
        return generate_headlines(previous_request, style=style, runtime_mode=runtime_mode)
    if is_request_complete(clean_message):
        return generate_headlines(clean_message, style=style, runtime_mode=runtime_mode)
    return MISSING_INFO_MESSAGE


def load_session_history(request: gr.Request):
    """Loads the current user's local chat history when the Gradio session starts."""
    history = load_chat_history(_request_namespace(request))
    return history, gr.update(visible=len(history) == 0), history


def chat_response(message, history, request: gr.Request):
    """
    Handles one chatbot turn.
    Shows a working message while the AI/model path runs, then replaces it with the final answer.
    """
    history = history or []
    clean_message = (message or "").strip()
    if not clean_message:
        yield history, "", gr.update(visible=len(history) == 0), history
        return

    user_entry = {"role": "user", "content": clean_message}
    working_entry = {"role": "assistant", "content": "La IA está trabajando..."}
    working_history = [*history, user_entry, working_entry]
    yield working_history, "", gr.update(visible=False), working_history

    bot_message = _resolve_bot_message(clean_message, history, DEFAULT_RUNTIME)
    final_history = [*history, user_entry, {"role": "assistant", "content": bot_message}]
    save_chat_history(final_history, _request_namespace(request))
    yield final_history, "", gr.update(visible=False), final_history


def reset_chat(request: gr.Request):
    """
    Limpia la conversación local del usuario actual.
    """
    clear_chat_history(_request_namespace(request))
    return [], "", gr.update(visible=True), []


def build_app() -> gr.Blocks:
    with gr.Blocks(title="Headline Booster") as demo:
        initial_history: list[dict[str, Any]] = []
        chat_state = gr.State(initial_history)
        with gr.Row(elem_id="headline-layout"):
            with gr.Column(elem_id="sidebar", scale=0, min_width=450):
                gr.HTML(
                    """
                    <aside class="sidebar-shell">
                      <h1>Chats Anteriores</h1>
                      <p class="sidebar-subtitle">Headline Booster</p>
                    </aside>
                    """
                )
                new_chat = gr.Button("+ Nuevo chat", elem_id="new-chat", size="lg")
                gr.HTML(
                    """
                    <div class="sessions-block">
                      <div class="sessions-title">Sesiones</div>
                      <div class="session-item">Sesión local</div>
                    </div>
                    """
                )

            with gr.Column(elem_id="main-panel", scale=1):
                gr.HTML('<div class="app-menu" aria-hidden="true">⋮</div>')
                welcome = gr.HTML(
                    """
                    <section id="welcome-hero">
                      <div class="hero-logo" aria-label="Headline Booster logo">
                        <div class="hero-icon">✍️</div>
                        <div class="hero-wordmark">HEADLINE BOOSTER</div>
                      </div>
                      <h2>Headline Booster</h2>
                      <p class="byline">By Jesús Cabrera</p>
                      <p class="tagline">✉️ Experto en encabezados claros que conectan ofertas con ventas de forma natural</p>
                    </section>
                    """,
                    visible=True,
                )
                chatbot = gr.Chatbot(
                    value=initial_history,
                    elem_id="chatbot",
                    show_label=False,
                    height=290,
                    avatar_images=(None, None),
                )
                with gr.Column(elem_classes="composer"):
                    with gr.Row(elem_classes="composer-row"):
                        message = gr.Textbox(
                            elem_id="message-input",
                            show_label=False,
                            lines=1,
                            max_lines=4,
                            placeholder="Ej. Vendo un taller de Diseño Humano para mujeres emprendedoras que quieren tomar decisiones con más claridad.",
                            scale=10,
                        )
                        send = gr.Button("↑", elem_id="send-btn", variant="primary", scale=1)

        demo.load(load_session_history, outputs=[chatbot, welcome, chat_state])
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
