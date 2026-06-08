"""Headline Booster: custom frontend + small-model headline optimizer.

This app serves index.html and exposes one compact API endpoint:
POST /api/improve_headline

The task is intentionally narrow for Tiny Titan:
1. diagnose one weak Spanish headline,
2. generate three stronger versions,
3. pick one winner.
"""

from __future__ import annotations

import json
import os
import re
from functools import lru_cache
from pathlib import Path
from typing import Any

from fastapi import Body
from fastapi.responses import HTMLResponse, JSONResponse
from gradio import Server


APP_TITLE = "Headline Booster"
APP_BUILD = "headline-optimizer-clean-2026-06-08"
ROOT = Path(__file__).resolve().parent
INDEX_PATH = ROOT / "index.html"

MODEL_ID = os.getenv("MODEL_ID", "Qwen/Qwen2.5-1.5B-Instruct")
MAX_NEW_TOKENS = int(os.getenv("MAX_NEW_TOKENS", "280"))
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
AUDIENCE_WORDS = [
    "para",
    "emprendedoras",
    "emprendedores",
    "mujeres",
    "coaches",
    "dueños",
    "negocios",
    "personas",
]
EMOTION_WORDS = [
    "duda",
    "miedo",
    "confusión",
    "agotamiento",
    "presión",
    "bloqueo",
    "ansiedad",
    "frustración",
    "confianza",
]
DIFFERENTIATION_WORDS = [
    "sin",
    "aunque",
    "usando",
    "método",
    "sistema",
    "diseño humano",
    "ia",
]

MOCK_HUMAN_DESIGN_HEADLINES = [
    "Aprende a tomar decisiones con más claridad usando tu Diseño Humano",
    "Deja de dudar de ti cada vez que tienes que tomar una decisión importante",
    "¿Y si tu cuerpo ya supiera qué decisión tomar?",
]


def normalize(text: str) -> str:
    """Trim repeated whitespace without changing the user's wording."""
    return re.sub(r"\s+", " ", (text or "").strip())


def lower(text: str) -> str:
    return normalize(text).lower()


def has_any(text: str, words: list[str]) -> bool:
    lowered = lower(text)
    return any(word in lowered for word in words)


def score_headline(headline: str) -> dict[str, int]:
    """Simple deterministic scoring used for the visible diagnosis cards."""
    title = lower(headline)
    word_count = len(title.split())

    clarity = 35
    if 5 <= word_count <= 14:
        clarity += 25
    elif word_count > 14:
        clarity += 10
    if has_any(title, RESULT_WORDS):
        clarity += 20
    if "?" in headline or "¿" in headline:
        clarity += 5

    desire = 25
    if has_any(title, RESULT_WORDS):
        desire += 25
    if has_any(title, EMOTION_WORDS):
        desire += 25
    if any(token in title for token in ["deja de", "sin", "más", "mejor"]):
        desire += 10

    specificity = 20
    if has_any(title, AUDIENCE_WORDS):
        specificity += 25
    if has_any(title, RESULT_WORDS):
        specificity += 20
    if word_count >= 7:
        specificity += 15

    differentiation = 20
    if has_any(title, DIFFERENTIATION_WORDS):
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


def diagnose_headline(headline: str) -> dict[str, Any]:
    """Return the fixed diagnosis structure consumed by index.html."""
    scores = score_headline(headline)
    missing: list[str] = []

    if not has_any(headline, RESULT_WORDS):
        missing.append("Un resultado concreto")
    if not has_any(headline, AUDIENCE_WORDS):
        missing.append("Una audiencia más definida")
    if not has_any(headline, EMOTION_WORDS):
        missing.append("Una situación emocional reconocible")
    if scores["diferenciacion"] < 55:
        missing.append("Una razón para seguir leyendo")
    if not missing:
        missing.append("Más tensión o curiosidad para hacerlo más competitivo")

    weakest = min(scores, key=scores.get)
    problem_by_weakest = {
        "claridad": "El titular no comunica con suficiente claridad qué cambio promete.",
        "deseo": "El titular dice el tema, pero todavía no despierta suficiente deseo.",
        "especificidad": "El titular es demasiado general y no aterriza bien para quién es ni qué consigue.",
        "diferenciacion": "El titular se entiende, pero suena parecido a muchos otros y necesita un ángulo más propio.",
    }
    problem = problem_by_weakest[weakest]

    if len(headline.split()) <= 4:
        problem = "El titular dice el tema, pero no dice por qué debería importarle a la persona."

    return {
        "scores": scores,
        "problema_principal": problem,
        "falta": missing[:4],
    }


def extract_json(text: str) -> dict[str, Any]:
    """Best-effort JSON extraction from small model output."""
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
    """Load the Hugging Face model once per process."""
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


def build_model_prompt(headline: str) -> list[dict[str, str]]:
    """Create a short, schema-bound prompt so a tiny model stays focused."""
    system_prompt = """
Eres Headline Booster, un coach experto en titulares de copywriting en español.
Tu tarea es mejorar UN titular existente.
Responde SOLO JSON válido, sin markdown ni texto extra.
Sé breve: máximo 3 versiones y una explicación corta.
""".strip()

    user_payload = {
        "titular_original": headline,
        "instrucciones": [
            "Crea exactamente 3 versiones mejoradas del titular.",
            "La versión 1 debe ser clara/directa.",
            "La versión 2 debe ser emocional.",
            "La versión 3 debe ser curiosa o intrigante.",
            "Elige un ganador entre 1, 2 o 3.",
        ],
        "schema_obligatorio": {
            "versiones": ["versión clara", "versión emocional", "versión curiosa"],
            "ganador_numero": 1,
            "por_que_gana": "explicación breve",
        },
    }
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False)},
    ]


def generate_with_model(headline: str) -> dict[str, Any]:
    """Generate improved headlines using the configured tiny model."""
    import torch

    tokenizer, model = get_tiny_model()
    prompt = tokenizer.apply_chat_template(
        build_model_prompt(headline),
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = tokenizer([prompt], return_tensors="pt")
    device = next(model.parameters()).device
    inputs = {key: value.to(device) for key, value in inputs.items()}

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            temperature=0.55,
            top_p=0.88,
            do_sample=True,
            repetition_penalty=1.08,
            pad_token_id=tokenizer.eos_token_id,
        )

    generated_tokens = outputs[0][inputs["input_ids"].shape[-1] :]
    generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
    return extract_json(generated_text)


def generate_mock(headline: str) -> dict[str, Any]:
    """Local fallback so the UI still works if model loading fails."""
    base = normalize(headline)
    if "diseño humano" in lower(headline):
        versions = MOCK_HUMAN_DESIGN_HEADLINES
    else:
        versions = [
            f"Convierte {base.lower()} en un resultado claro y deseable",
            "Deja de perder oportunidades con un titular que no conecta",
            "¿Y si tu titular pudiera despertar más interés en segundos?",
        ]

    return {
        "versiones": versions,
        "ganador_numero": 3,
        "por_que_gana": "Gana porque abre curiosidad, conecta rápido y crea una razón clara para seguir leyendo.",
    }


def clean_model_output(model_output: dict[str, Any], headline: str) -> dict[str, Any]:
    """Guarantee the frontend always receives three versions and a valid winner."""
    versions = model_output.get("versiones")
    if not isinstance(versions, list) or len(versions) < 3:
        versions = generate_mock(headline)["versiones"]
    versions = [str(item).strip() for item in versions if str(item).strip()][:3]

    try:
        winner_number = int(model_output.get("ganador_numero", 3))
    except Exception:
        winner_number = 3
    winner_number = max(1, min(3, winner_number))

    why = str(
        model_output.get("por_que_gana")
        or "Gana porque mejora claridad, deseo y curiosidad sin alargar demasiado el mensaje."
    ).strip()

    return {
        "versiones": versions,
        "ganador_numero": winner_number,
        "ganador": versions[winner_number - 1],
        "por_que_gana": why[:360],
    }


def improve_headline(headline: str) -> dict[str, Any]:
    """Main app use case: diagnose, improve, and select the strongest headline."""
    headline = normalize(headline)
    if len(headline) < 3:
        return {"ok": False, "error": "Pega un titular actual para poder mejorarlo."}

    diagnosis = diagnose_headline(headline)
    runtime = "model" if USE_REAL_MODEL else "mock"

    try:
        raw_output = generate_with_model(headline) if USE_REAL_MODEL else generate_mock(headline)
    except Exception as exc:
        raw_output = generate_mock(headline)
        runtime = f"mock_after_model_error:{type(exc).__name__}"

    improved = clean_model_output(raw_output, headline)

    return {
        "ok": True,
        "app_build": APP_BUILD,
        "runtime": runtime,
        "model_id": MODEL_ID if USE_REAL_MODEL else "mock-local",
        "titular_original": headline,
        "diagnostico": diagnosis["scores"],
        "problema_principal": diagnosis["problema_principal"],
        "falta": diagnosis["falta"],
        "versiones": improved["versiones"],
        "mini_battle": {
            "mas_claro": 1,
            "mas_emocional": 2,
            "mas_curioso": 3,
        },
        "ganador_numero": improved["ganador_numero"],
        "ganador": improved["ganador"],
        "por_que_gana": improved["por_que_gana"],
    }


def read_index() -> str:
    """Read the custom frontend."""
    if not INDEX_PATH.exists():
        return """
        <!doctype html><html><head><meta charset='utf-8'><title>Headline Booster</title></head>
        <body style='font-family:system-ui;padding:32px'>
        <h1>Headline Booster</h1><p>Missing index.html.</p>
        </body></html>
        """
    return INDEX_PATH.read_text(encoding="utf-8")


def runtime_info() -> dict[str, Any]:
    return {
        "ok": True,
        "app_title": APP_TITLE,
        "app_build": APP_BUILD,
        "model_id": MODEL_ID,
        "max_new_tokens": MAX_NEW_TOKENS,
        "use_real_model": USE_REAL_MODEL,
    }


app = Server()


@app.get("/", response_class=HTMLResponse)
async def homepage() -> HTMLResponse:
    return HTMLResponse(read_index())


@app.get("/health")
async def health() -> JSONResponse:
    return JSONResponse(runtime_info())


@app.post("/api/improve_headline")
async def rest_improve_headline(payload: dict[str, Any] = Body(default={})) -> JSONResponse:
    result = improve_headline(str(payload.get("headline", "")))
    return JSONResponse(result)


demo = app

demo.launch(show_error=True)
