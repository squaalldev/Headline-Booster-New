"""Headline Booster: custom frontend + small-model headline optimizer.

This build serves a custom index.html and exposes a compact API for improving
existing Spanish headlines. The model task is intentionally narrow for Tiny Titan:
diagnose a weak headline, generate three stronger versions, and pick a winner.
"""

from __future__ import annotations

import json
import os
import re
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List

try:
    import gradio as gr
except Exception:  # pragma: no cover
    gr = None  # type: ignore[assignment]

try:
    from gradio import Server  # type: ignore[attr-defined]
except Exception:  # pragma: no cover
    Server = None  # type: ignore[assignment]

try:
    from fastapi import Body
    from fastapi.responses import HTMLResponse, JSONResponse
    from fastapi.staticfiles import StaticFiles
except Exception:  # pragma: no cover
    Body = None  # type: ignore[assignment]
    HTMLResponse = None  # type: ignore[assignment]
    JSONResponse = None  # type: ignore[assignment]
    StaticFiles = None  # type: ignore[assignment]


APP_TITLE = "Headline Booster"
APP_BUILD = "headline-doctor-arena-2026-06-08"
ROOT = Path(__file__).resolve().parent
ASSET_DIR = ROOT / "assets"

MODEL_ID = os.getenv("MODEL_ID", "Qwen/Qwen2.5-1.5B-Instruct")
MODEL_FALLBACK_ID = os.getenv("MODEL_FALLBACK_ID", "Qwen/Qwen2.5-3B-Instruct")
REAL_MODEL_MAX_NEW_TOKENS = int(os.getenv("MAX_NEW_TOKENS", "280"))
AUTO_RUNTIME = os.getenv("USE_REAL_MODEL", "auto").lower()
IS_HUGGING_FACE_SPACE = bool(os.getenv("SPACE_ID") or os.getenv("SPACE_HOST") or os.getenv("HF_SPACE_ID"))
USE_REAL_MODEL = AUTO_RUNTIME == "true" or (AUTO_RUNTIME == "auto" and IS_HUGGING_FACE_SPACE)


RESULT_WORDS = [
    "logra",
    "lograr",
    "consigue",
    "conseguir",
    "aumenta",
    "mejora",
    "vende",
    "vender",
    "decide",
    "decidir",
    "claridad",
    "confianza",
    "calma",
    "resultado",
]
AUDIENCE_WORDS = ["para", "emprendedoras", "emprendedores", "mujeres", "coaches", "dueños", "negocios", "personas"]
EMOTION_WORDS = ["duda", "miedo", "confusión", "agotamiento", "presión", "bloqueo", "ansiedad", "frustración", "confianza"]
DIFFERENTIATION_WORDS = ["sin", "aunque", "usando", "con", "método", "sistema", "diseño humano", "ia", "en "]


STYLE_HEADLINES = {
    "direct": "Aprende a tomar decisiones con más claridad usando tu Diseño Humano",
    "emotional": "Deja de dudar de ti cada vez que tienes que tomar una decisión importante",
    "curious": "¿Y si tu cuerpo ya supiera qué decisión tomar?",
}


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip())


def _lower(text: str) -> str:
    return _normalize(text).lower()


def _has_any(text: str, words: List[str]) -> bool:
    lowered = _lower(text)
    return any(word in lowered for word in words)


def _score_headline(headline: str, context: str = "") -> Dict[str, int]:
    text = _lower(f"{headline} {context}")
    title = _lower(headline)
    word_count = len(title.split())

    clarity = 35
    if 5 <= word_count <= 14:
        clarity += 25
    elif word_count > 14:
        clarity += 10
    if _has_any(text, RESULT_WORDS):
        clarity += 20
    if "?" in headline or "¿" in headline:
        clarity += 5

    desire = 25
    if _has_any(text, RESULT_WORDS):
        desire += 25
    if _has_any(text, EMOTION_WORDS):
        desire += 25
    if any(token in title for token in ["deja de", "sin", "más", "mejor"]):
        desire += 10

    specificity = 20
    if _has_any(text, AUDIENCE_WORDS):
        specificity += 25
    if _has_any(text, RESULT_WORDS):
        specificity += 20
    if word_count >= 7:
        specificity += 15

    differentiation = 20
    if _has_any(text, DIFFERENTIATION_WORDS):
        differentiation += 25
    if any(char.isdigit() for char in title):
        differentiation += 10
    if "?" in headline or "¿" in headline:
        differentiation += 10

    return {
        "claridad": max(0, min(100, clarity)),
        "deseo": max(0, min(100, desire)),
        "especificidad": max(0, min(100, specificity)),
        "diferenciacion": max(0, min(100, differentiation)),
    }


def _diagnose_headline(headline: str, context: str = "") -> Dict[str, Any]:
    scores = _score_headline(headline, context)
    text = _lower(f"{headline} {context}")
    missing: List[str] = []

    if not _has_any(text, RESULT_WORDS):
        missing.append("Un resultado concreto")
    if not _has_any(text, AUDIENCE_WORDS):
        missing.append("Una audiencia más definida")
    if not _has_any(text, EMOTION_WORDS):
        missing.append("Una situación emocional reconocible")
    if scores["diferenciacion"] < 55:
        missing.append("Una razón para seguir leyendo")
    if not missing:
        missing.append("Más tensión o curiosidad para hacerlo más competitivo")

    weakest = min(scores, key=scores.get)
    if weakest == "claridad":
        problem = "El titular no comunica con suficiente claridad qué cambio promete."
    elif weakest == "deseo":
        problem = "El titular dice el tema, pero todavía no despierta suficiente deseo."
    elif weakest == "especificidad":
        problem = "El titular es demasiado general y no aterriza bien para quién es ni qué consigue."
    else:
        problem = "El titular se entiende, pero suena parecido a muchos otros y necesita un ángulo más propio."

    if len(headline.split()) <= 4:
        problem = "El titular dice el tema, pero no dice por qué debería importarle a la persona."

    return {"scores": scores, "problema_principal": problem, "falta": missing[:4]}


def _extract_json(text: str) -> Dict[str, Any]:
    cleaned = (text or "").strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    try:
        data = json.loads(cleaned)
        return data if isinstance(data, dict) else {}
    except Exception:
        pass
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start >= 0 and end > start:
        try:
            data = json.loads(cleaned[start : end + 1])
            return data if isinstance(data, dict) else {}
        except Exception:
            return {}
    return {}


@lru_cache(maxsize=1)
def get_tiny_model():
    """Loads the configured Hugging Face model once per process."""
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


def _build_model_prompt(headline: str, context: str = "") -> List[Dict[str, str]]:
    system_prompt = """
Eres Headline Booster, un coach experto en titulares de copywriting en español.
Tu tarea es mejorar UN titular existente.
Responde SOLO JSON válido, sin markdown.
Sé breve: máximo 3 versiones y una explicación corta.
""".strip()

    user_prompt = {
        "titular_original": headline,
        "contexto_opcional": context,
        "instrucciones": [
            "Crea exactamente 3 versiones mejoradas del titular.",
            "La versión 1 debe ser clara/directa.",
            "La versión 2 debe ser emocional.",
            "La versión 3 debe ser curiosa o intrigante.",
            "Elige un ganador entre 1, 2 o 3.",
        ],
        "schema": {
            "versiones": ["directa", "emocional", "curiosa"],
            "ganador_numero": 1,
            "por_que_gana": "explicación breve",
        },
    }
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": json.dumps(user_prompt, ensure_ascii=False)},
    ]


def _generate_with_model(headline: str, context: str = "") -> Dict[str, Any]:
    import torch

    tokenizer, model = get_tiny_model()
    messages = _build_model_prompt(headline, context)
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer([prompt], return_tensors="pt")
    device = next(model.parameters()).device
    inputs = {key: value.to(device) for key, value in inputs.items()}

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=REAL_MODEL_MAX_NEW_TOKENS,
            temperature=0.55,
            top_p=0.88,
            do_sample=True,
            repetition_penalty=1.08,
            pad_token_id=tokenizer.eos_token_id,
        )

    generated_tokens = outputs[0][inputs["input_ids"].shape[-1] :]
    generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
    return _extract_json(generated_text)


def _generate_mock(headline: str, context: str = "") -> Dict[str, Any]:
    base = _normalize(headline)
    ctx = _lower(context)
    if "diseño humano" in _lower(f"{headline} {context}"):
        versions = [STYLE_HEADLINES["direct"], STYLE_HEADLINES["emotional"], STYLE_HEADLINES["curious"]]
    else:
        versions = [
            f"Convierte {base.lower()} en un resultado claro y deseable",
            f"Deja de perder oportunidades con un titular que no conecta",
            f"¿Y si tu titular pudiera despertar más interés en segundos?",
        ]
        if "web" in ctx or "página" in ctx:
            versions[0] = "Convierte tu página web en un mensaje claro que invite a comprar"
    return {
        "versiones": versions,
        "ganador_numero": 3,
        "por_que_gana": "Gana porque abre curiosidad, conecta rápido y crea una razón clara para seguir leyendo.",
    }


def improve_headline(headline: str, context: str = "") -> Dict[str, Any]:
    headline = _normalize(headline)
    context = _normalize(context)
    if len(headline) < 3:
        return {"ok": False, "error": "Pega un titular actual para poder mejorarlo."}

    diagnosis = _diagnose_headline(headline, context)
    runtime = "model" if USE_REAL_MODEL else "mock"

    try:
        model_output = _generate_with_model(headline, context) if USE_REAL_MODEL else _generate_mock(headline, context)
    except Exception as exc:
        model_output = _generate_mock(headline, context)
        runtime = f"mock_after_model_error: {type(exc).__name__}"

    versions = model_output.get("versiones")
    if not isinstance(versions, list) or len(versions) < 3:
        versions = _generate_mock(headline, context)["versiones"]
    versions = [str(item).strip() for item in versions if str(item).strip()][:3]

    try:
        winner_number = int(model_output.get("ganador_numero", 3))
    except Exception:
        winner_number = 3
    winner_number = max(1, min(3, winner_number))
    winner = versions[winner_number - 1]

    why = str(model_output.get("por_que_gana") or "Gana porque mejora claridad, deseo y curiosidad sin alargar demasiado el mensaje.").strip()

    return {
        "ok": True,
        "app_build": APP_BUILD,
        "runtime": runtime,
        "model_id": MODEL_ID if USE_REAL_MODEL else "mock-local",
        "titular_original": headline,
        "contexto": context,
        "diagnostico": diagnosis["scores"],
        "problema_principal": diagnosis["problema_principal"],
        "falta": diagnosis["falta"],
        "versiones": versions,
        "mini_battle": {
            "mas_claro": 1,
            "mas_emocional": 2,
            "mas_curioso": 3,
        },
        "ganador_numero": winner_number,
        "ganador": winner,
        "por_que_gana": why[:360],
    }


def read_index() -> str:
    path = ROOT / "index.html"
    if not path.exists():
        return """
        <!doctype html><html><head><meta charset='utf-8'><title>Headline Booster</title></head>
        <body style='font-family:system-ui;padding:32px'>
        <h1>Headline Booster</h1><p>Missing index.html.</p>
        </body></html>
        """
    return path.read_text(encoding="utf-8")


def runtime_info() -> Dict[str, Any]:
    return {
        "ok": True,
        "app_title": APP_TITLE,
        "app_build": APP_BUILD,
        "model_id": MODEL_ID,
        "fallback_model_id": MODEL_FALLBACK_ID,
        "max_new_tokens": REAL_MODEL_MAX_NEW_TOKENS,
        "use_real_model": USE_REAL_MODEL,
        "no_usage_limit_in_app": True,
    }


if Server is None:
    if gr is None:
        raise RuntimeError("Gradio is not installed. This app requires gradio>=6.0.0.")
    with gr.Blocks(title=APP_TITLE) as demo:  # pragma: no cover
        gr.Markdown("# Headline Booster\n\nThis build expects `gradio.Server`. Please restart with gradio>=6.0.0.")
    demo.launch(show_error=True)
else:
    app = Server()

    @app.api(name="improve_headline")
    def api_improve_headline(headline: str, context: str = "") -> Dict[str, Any]:
        return improve_headline(headline, context)

    @app.get("/", response_class=HTMLResponse)
    async def homepage() -> Any:
        return HTMLResponse(read_index())

    @app.get("/health")
    async def health() -> Any:
        payload = runtime_info()
        return JSONResponse(payload) if JSONResponse is not None else payload

    @app.post("/api/improve_headline")
    async def rest_improve_headline(payload: Dict[str, Any] = Body(default={})) -> Any:
        result = improve_headline(str(payload.get("headline", "")), str(payload.get("context", "")))
        return JSONResponse(result) if JSONResponse is not None else result

    if StaticFiles is not None and ASSET_DIR.exists():
        app.mount("/assets", StaticFiles(directory=str(ASSET_DIR)), name="assets")

    demo = app
    demo.launch(show_error=True)
