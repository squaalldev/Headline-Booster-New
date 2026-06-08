"""Road B: The Other Screen - hackathon-ready Gradio Server app.

Custom frontend: index.html
Model runtime: Qwen GGUF through llama.cpp via llama-cpp-python
No mock fallback: if the model/runtime cannot load, the app returns a visible error.
"""

from __future__ import annotations

import ctypes
import datetime as dt
import glob
import html
import json
import os
import re
import requests
import site
import threading
import uuid
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    import gradio as gr
except Exception:  # pragma: no cover
    gr = None  # type: ignore[assignment]

try:
    from gradio import Server  # type: ignore[attr-defined]
except Exception:  # pragma: no cover
    Server = None  # type: ignore[assignment]

try:
    from fastapi.responses import HTMLResponse, JSONResponse
    from fastapi.staticfiles import StaticFiles
except Exception:  # pragma: no cover
    HTMLResponse = None  # type: ignore[assignment]
    JSONResponse = None  # type: ignore[assignment]
    StaticFiles = None  # type: ignore[assignment]


# -----------------------------------------------------------------------------
# CUDA library preparation for pip-installed NVIDIA runtime packages
# -----------------------------------------------------------------------------


def _candidate_site_dirs() -> List[Path]:
    dirs: List[Path] = []
    try:
        dirs.extend(Path(p) for p in site.getsitepackages())
    except Exception:
        pass
    try:
        dirs.append(Path(site.getusersitepackages()))
    except Exception:
        pass
    for pattern in (
        "/usr/local/lib/python*/site-packages",
        "/home/user/.local/lib/python*/site-packages",
    ):
        dirs.extend(Path(p) for p in glob.glob(pattern))

    deduped: List[Path] = []
    seen = set()
    for d in dirs:
        key = str(d)
        if key not in seen and d.exists():
            deduped.append(d)
            seen.add(key)
    return deduped


def _prepare_cuda_runtime_libraries() -> Dict[str, Any]:
    """Expose pip-installed CUDA shared libraries before importing llama_cpp.

    This is harmless on CPU wheels: missing CUDA libraries are reported but not fatal.
    """

    lib_dirs: List[str] = []
    for base in _candidate_site_dirs():
        for pattern in ("nvidia/*/lib", "nvidia/*/lib64", "nvidia/*/bin"):
            for path in base.glob(pattern):
                if path.is_dir():
                    lib_dirs.append(str(path))

    deduped: List[str] = []
    seen = set()
    for path in lib_dirs:
        if path not in seen:
            deduped.append(path)
            seen.add(path)

    if deduped:
        current = os.environ.get("LD_LIBRARY_PATH", "")
        os.environ["LD_LIBRARY_PATH"] = ":".join(deduped + ([current] if current else []))

    loaded: List[str] = []
    missing: List[str] = []
    flags = getattr(ctypes, "RTLD_GLOBAL", 0)
    for name in (
        "libnvJitLink.so.12",
        "libcudart.so.12",
        "libcublasLt.so.12",
        "libcublas.so.12",
    ):
        found: Optional[Path] = None
        for d in deduped:
            candidate = Path(d) / name
            if candidate.exists():
                found = candidate
                break
        if found is None:
            missing.append(name)
            continue
        try:
            ctypes.CDLL(str(found), mode=flags)
            loaded.append(name)
        except Exception:
            missing.append(name)

    return {"lib_dirs": deduped, "loaded": loaded, "missing": missing}


CUDA_RUNTIME_PREP = _prepare_cuda_runtime_libraries()

try:
    from llama_cpp import Llama
except Exception as import_error:  # pragma: no cover - resolved on Space runtime
    Llama = None  # type: ignore[assignment]
    LLAMA_IMPORT_ERROR = import_error
else:
    LLAMA_IMPORT_ERROR = None


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

APP_TITLE = "Road B: The Other Screen"
APP_BUILD = "roadb-modal-gpu-ready-2026-06-05"
SCHEMA_VERSION = "0.9.0"

ROOT = Path(__file__).resolve().parent
ASSET_DIR = ROOT / "assets"
DOCS_DIR = ROOT / "docs"
SAMPLES_DIR = ROOT / "samples"


def clean_env_value(name: str, default: str) -> str:
    """Read a Space variable while tolerating accidental comments/multiline values."""

    raw = os.getenv(name, default)
    lines = []
    for line in str(raw).splitlines():
        value = line.strip().strip('"').strip("'")
        if not value or value.startswith("#"):
            continue
        lines.append(value)
    return lines[-1] if lines else default


MODEL_REPO_ID = clean_env_value("MODEL_REPO_ID", "unsloth/Qwen3.5-9B-GGUF")
# CPU-friendly default. For final GPU judging, set MODEL_FILENAME=Qwen3.5-9B-Q4_K_M.gguf.
MODEL_FILENAME = clean_env_value("MODEL_FILENAME", "Qwen3.5-9B-Q3_K_M.gguf")
MODEL_PATH = clean_env_value("MODEL_PATH", "")

# Optional Modal GPU backend. When MODAL_QWEN_URL is set, HF Space stays CPU-only
# and all Qwen/llama.cpp generation is performed by the Modal endpoint.
MODAL_QWEN_URL = clean_env_value("MODAL_QWEN_URL", "")
MODAL_QWEN_TOKEN = clean_env_value("MODAL_QWEN_TOKEN", "")
MODAL_TIMEOUT = int(clean_env_value("MODAL_TIMEOUT", "900"))



def _gpu_device_visible() -> bool:
    visible = os.getenv("NVIDIA_VISIBLE_DEVICES", "").strip().lower()
    if visible and visible not in {"none", "void", "", "-1"}:
        return True
    return any(Path(p).exists() for p in ("/dev/nvidia0", "/dev/nvidiactl"))


GPU_VISIBLE = _gpu_device_visible()
DEFAULT_N_CTX = "8192" if GPU_VISIBLE else "2048"
DEFAULT_N_GPU_LAYERS = "-1" if GPU_VISIBLE else "0"
DEFAULT_N_BATCH = "512" if GPU_VISIBLE else "64"
DEFAULT_MAX_TOKENS = "850" if GPU_VISIBLE else "520"

N_CTX = int(clean_env_value("N_CTX", DEFAULT_N_CTX))
N_GPU_LAYERS = int(clean_env_value("N_GPU_LAYERS", DEFAULT_N_GPU_LAYERS))
N_BATCH = int(clean_env_value("N_BATCH", DEFAULT_N_BATCH))
N_THREADS_RAW = clean_env_value("N_THREADS", "")
N_THREADS = int(N_THREADS_RAW) if N_THREADS_RAW else None
MAX_TOKENS = int(clean_env_value("MAX_TOKENS", DEFAULT_MAX_TOKENS))
TEMPERATURE = float(clean_env_value("TEMPERATURE", "0.78"))
TOP_P = float(clean_env_value("TOP_P", "0.92"))
SEED_RAW = clean_env_value("ROAD_B_SEED", "0")
SEED = int(SEED_RAW or "0") or -1

MODEL_LOCK = threading.Lock()


# -----------------------------------------------------------------------------
# Prompting
# -----------------------------------------------------------------------------

SYSTEM_PROMPT = """
You are the narrative engine for Road B: The Other Screen, an interactive speculative-fiction game.

The user gives a fork in their life. Road A is the life they chose. Road B is the fictional life they did not choose.
Your job is to generate transmissions from the Road B self.

Hard rules:
- Do not predict reality. Never imply this is what would truly have happened.
- Do not rank Road B as better or worse than Road A.
- Every gain must carry a cost; every loss must contain ambiguity or hidden beauty.
- Do not give medical, legal, financial, or mental-health advice.
- Do not encourage regret, risky decisions, self-harm, obsession, or contact with real people.
- If the input asks for advice or prediction, transform it into fictional, reflective story.
- Write compact, vivid, emotionally specific prose with concrete sensory detail.
- Keep the voice hushed, second-person-adjacent, sometimes uncanny, never marketing-like.
- Return valid JSON only. No markdown fences, no commentary outside JSON.
- Use short complete strings. Do not write long paragraphs that risk being cut off.
""".strip()

CRISIS_PATTERNS = [
    r"\bkill myself\b",
    r"\bsuicide\b",
    r"\bend my life\b",
    r"\bself[- ]?harm\b",
    r"\bi want to die\b",
    r"\bcan't go on\b",
]

ARTIFACT_SPECS: Dict[str, Dict[str, str]] = {
    "cost_ledger": {
        "label": "Cost Ledger",
        "verb": "Open the Cost Ledger",
        "instruction": "Name three costs Road B paid. Each line should be concrete and emotionally specific, not melodramatic.",
        "tone": "ledger, tender, unsparing",
    },
    "beauty_ledger": {
        "label": "Beauty Ledger",
        "verb": "Open the Beauty Ledger",
        "instruction": "Name three forms of beauty Road B found. Each line should be grounded in ordinary detail.",
        "tone": "warm, luminous, specific",
    },
    "typical_tuesday": {
        "label": "A Typical Tuesday",
        "verb": "Visit a Tuesday",
        "instruction": "Write one ordinary Tuesday scene from Road B. Include place, weather/light, work, body, and one private feeling.",
        "tone": "cinematic, mundane, intimate",
    },
    "unsent_letter": {
        "label": "The Unsent Letter",
        "verb": "Read the Unsent Letter",
        "instruction": "Write a short letter Road B never sent to Road A. It should confess one envy and one gratitude.",
        "tone": "letter, restrained, honest",
    },
    "split_moment": {
        "label": "The Moment It Split",
        "verb": "Return to the Split",
        "instruction": "Recreate the exact moment where Road A and Road B diverged. Make it sensory and cinematic.",
        "tone": "threshold, slow-motion, uncanny",
    },
}


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------


def now_iso() -> str:
    return dt.datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def dumps(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":"))


def pretty_dumps(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, indent=2)


def esc(value: Any) -> str:
    return html.escape(str(value if value is not None else ""), quote=True)


def normalize_text(value: str, limit: int = 2400) -> str:
    value = re.sub(r"\s+", " ", (value or "").strip())
    return value[:limit]


def contains_crisis_signal(*texts: str) -> bool:
    joined = "\n".join(t or "" for t in texts).lower()
    return any(re.search(pattern, joined) for pattern in CRISIS_PATTERNS)


def clamp_signal(value: int) -> int:
    return max(0, min(100, int(value)))


def make_session_label() -> str:
    return "echo-" + uuid.uuid4().hex[:4]


def make_universe_id() -> str:
    return "B-" + uuid.uuid4().hex[:2].upper() + "-" + uuid.uuid4().hex[:4].upper()


def public_runtime_info(model_loaded: Optional[bool] = None) -> Dict[str, Any]:
    info: Dict[str, Any] = {
        "app_title": APP_TITLE,
        "app_build": APP_BUILD,
        "schema_version": SCHEMA_VERSION,
        "strict_ai": True,
        "mock_mode": False,
        "runtime": "Modal GPU + llama.cpp" if MODAL_QWEN_URL else "llama.cpp via llama-cpp-python",
        "modal_qwen_enabled": bool(MODAL_QWEN_URL),
        "modal_qwen_url_set": bool(MODAL_QWEN_URL),
        "model_repo_id": MODEL_REPO_ID,
        "model_filename": MODEL_FILENAME,
        "model_path_set": bool(MODEL_PATH),
        "gpu_device_visible": GPU_VISIBLE,
        "cuda_runtime_prep": CUDA_RUNTIME_PREP,
        "n_ctx": N_CTX,
        "n_gpu_layers": N_GPU_LAYERS,
        "n_batch": N_BATCH,
        "n_threads": N_THREADS,
        "max_tokens": MAX_TOKENS,
        "temperature": TEMPERATURE,
        "top_p": TOP_P,
    }
    if model_loaded is not None:
        info["model_loaded"] = model_loaded
    if LLAMA_IMPORT_ERROR is not None:
        info["llama_import_error"] = repr(LLAMA_IMPORT_ERROR)
    return info


def error_response(message: str, *, kind: str = "runtime_error", status: int = 500) -> Dict[str, Any]:
    return {
        "ok": False,
        "kind": kind,
        "status": status,
        "error": message,
        "runtime": public_runtime_info(model_loaded=load_llm.cache_info().currsize > 0 if "load_llm" in globals() else False),
    }


def crisis_payload() -> Dict[str, Any]:
    return {
        "ok": False,
        "kind": "safety",
        "status": 400,
        "error": (
            "Road B is speculative fiction and is not appropriate for crisis support. "
            "Please contact local emergency services or a trusted person right now if you may be in danger."
        ),
        "runtime": public_runtime_info(model_loaded=load_llm.cache_info().currsize > 0),
    }


def _parse_json_string_literal(value: str) -> str:
    try:
        return json.loads('"' + value + '"')
    except Exception:
        return value.replace('\\"', '"').replace("\\n", "\n").replace("\\t", "\t")


def _extract_partial_json_fields(text: str) -> Dict[str, Any]:
    """Recover completed string fields from a truncated JSON object."""

    fields: Dict[str, Any] = {}
    for match in re.finditer(r'"([^"\\]+)"\s*:\s*"((?:\\.|[^"\\])*)"', text, flags=re.DOTALL):
        key = match.group(1).strip()
        value = _parse_json_string_literal(match.group(2)).strip()
        if key and value:
            fields[key] = value

    # Recover simple arrays of strings if present and closed.
    for match in re.finditer(r'"([^"\\]+)"\s*:\s*\[((?:\s*"(?:\\.|[^"\\])*"\s*,?\s*)+)\]', text, flags=re.DOTALL):
        key = match.group(1).strip()
        body = match.group(2)
        values = [_parse_json_string_literal(m.group(1)).strip() for m in re.finditer(r'"((?:\\.|[^"\\])*)"', body)]
        values = [v for v in values if v]
        if key and values:
            fields[key] = values

    return fields


def extract_json(text: str) -> Dict[str, Any]:
    cleaned = (text or "").strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*```$", "", cleaned)

    try:
        value = json.loads(cleaned)
        if isinstance(value, dict):
            value.setdefault("_raw", cleaned)
            return value
        return {"value": value, "_raw": cleaned}
    except Exception:
        pass

    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start >= 0 and end > start:
        try:
            value = json.loads(cleaned[start : end + 1])
            if isinstance(value, dict):
                value.setdefault("_raw", cleaned)
                return value
            return {"value": value, "_raw": cleaned}
        except Exception:
            pass

    partial = _extract_partial_json_fields(cleaned)
    if partial:
        partial["_raw"] = cleaned
        partial["_partial_json"] = True
        return partial

    return {"_raw": cleaned, "_parse_failed": True}


def looks_like_raw_json(text: str) -> bool:
    t = (text or "").strip()
    return t.startswith("{") and '"' in t and ":" in t


def visible_field(output: Dict[str, Any], *keys: str, fallback: str = "") -> str:
    """Return user-visible text without leaking raw JSON."""

    for key in keys:
        value = output.get(key)
        if value is None:
            continue
        text = str(value).strip()
        if not text:
            continue
        if looks_like_raw_json(text):
            nested = extract_json(text)
            for nested_key in ("opening_line", "answer", "body", "final_message", "daily_scene", "insight", "last_line"):
                nested_value = nested.get(nested_key)
                if nested_value and not looks_like_raw_json(str(nested_value)):
                    return str(nested_value).strip()
            continue
        return text
    return fallback


def visible_list(output: Dict[str, Any], key: str, fallback: Optional[List[str]] = None) -> List[str]:
    value = output.get(key)
    if isinstance(value, list):
        return [str(v).strip() for v in value if str(v).strip() and not looks_like_raw_json(str(v))][:5]
    if isinstance(value, str) and value.strip():
        return [line.strip(" -•\t") for line in re.split(r"[\n;]", value) if line.strip()][:5]
    return fallback or []


# -----------------------------------------------------------------------------
# Model loading and calls
# -----------------------------------------------------------------------------


@lru_cache(maxsize=1)
def load_llm() -> Any:
    if Llama is None:
        raise RuntimeError(
            "llama-cpp-python is not available. This submitted build has no mock fallback. "
            "Install llama-cpp-python or use the correct CPU/GPU requirements file. "
            f"Import error: {LLAMA_IMPORT_ERROR!r}"
        )

    kwargs: Dict[str, Any] = {
        "n_ctx": N_CTX,
        "n_gpu_layers": N_GPU_LAYERS,
        "n_batch": N_BATCH,
        "verbose": False,
    }
    if N_THREADS is not None:
        kwargs["n_threads"] = N_THREADS

    if MODEL_PATH:
        return Llama(model_path=MODEL_PATH, **kwargs)

    return Llama.from_pretrained(repo_id=MODEL_REPO_ID, filename=MODEL_FILENAME, **kwargs)


def normalize_chat_messages(messages: List[Dict[str, str]]) -> List[Dict[str, str]]:
    system_parts: List[str] = []
    body: List[Dict[str, str]] = []
    for message in messages:
        role = str(message.get("role", "user") or "user").strip().lower()
        content = str(message.get("content", "") or "").strip()
        if not content:
            continue
        if role == "system":
            system_parts.append(content)
        elif role in {"user", "assistant"}:
            body.append({"role": role, "content": content})
        else:
            body.append({"role": "user", "content": content})
    normalized: List[Dict[str, str]] = []
    if system_parts:
        normalized.append({"role": "system", "content": "\n\n".join(system_parts)})
    normalized.extend(body)
    return normalized


def modal_model_json(messages: List[Dict[str, str]], *, max_tokens: int = MAX_TOKENS) -> Dict[str, Any]:
    if not MODAL_QWEN_URL:
        raise RuntimeError("MODAL_QWEN_URL is not set.")

    headers = {"Content-Type": "application/json"}
    if MODAL_QWEN_TOKEN:
        headers["Authorization"] = f"Bearer {MODAL_QWEN_TOKEN}"

    payload: Dict[str, Any] = {
        "messages": normalize_chat_messages(messages),
        "max_tokens": max_tokens,
        "temperature": TEMPERATURE,
        "top_p": TOP_P,
        "seed": SEED,
        "token": MODAL_QWEN_TOKEN,
    }

    try:
        response = requests.post(MODAL_QWEN_URL, json=payload, headers=headers, timeout=MODAL_TIMEOUT)
        response.raise_for_status()
        data = response.json()
    except Exception as exc:
        raise RuntimeError(f"Modal Qwen endpoint failed: {exc}") from exc

    if not data.get("ok", False):
        raise RuntimeError(str(data.get("error") or "Modal Qwen endpoint returned ok=false."))

    parsed = data.get("parsed")
    if isinstance(parsed, dict):
        return parsed

    raw = data.get("raw") or data.get("content") or ""
    return extract_json(str(raw))


def model_json(messages: List[Dict[str, str]], *, max_tokens: int = MAX_TOKENS) -> Dict[str, Any]:
    if MODAL_QWEN_URL:
        return modal_model_json(messages, max_tokens=max_tokens)

    with MODEL_LOCK:
        llm = load_llm()
        safe_messages = normalize_chat_messages(messages)
        call_kwargs: Dict[str, Any] = {
            "messages": safe_messages,
            "temperature": TEMPERATURE,
            "top_p": TOP_P,
            "max_tokens": max_tokens,
        }
        if SEED >= 0:
            call_kwargs["seed"] = SEED
        try:
            out = llm.create_chat_completion(**call_kwargs)
        except TypeError:
            call_kwargs.pop("seed", None)
            out = llm.create_chat_completion(**call_kwargs)
    content = out["choices"][0]["message"]["content"]
    return extract_json(content)


# -----------------------------------------------------------------------------
# Prompt builders
# -----------------------------------------------------------------------------


def parse_state(state_json: str) -> Dict[str, Any]:
    if not state_json:
        raise ValueError("No active Road B session. Invoke the other self first.")
    state = json.loads(state_json)
    if not isinstance(state, dict):
        raise ValueError("Session state is not valid JSON.")
    return state


def slim_state(state: Dict[str, Any], include_turns: int = 6) -> Dict[str, Any]:
    return {
        "session_id": state.get("session_id"),
        "session_label": state.get("session_label"),
        "universe_id": state.get("universe_id"),
        "signal": state.get("signal"),
        "inputs": state.get("inputs", {}),
        "profile": state.get("profile", {}),
        "opening": state.get("opening", {}),
        "artifacts": state.get("artifacts", []),
        "turns": state.get("turns", [])[-include_turns:],
    }


def append_trace(state: Dict[str, Any], step: str, prompt_summary: str, output: Dict[str, Any]) -> None:
    trace = state.setdefault("trace", [])
    trace.append(
        {
            "time": now_iso(),
            "step": step,
            "prompt_summary": prompt_summary[:500],
            "output_keys": sorted([str(k) for k in output.keys() if not str(k).startswith("_")]),
            "model": MODEL_REPO_ID + "/" + MODEL_FILENAME,
            "runtime": "llama.cpp",
        }
    )
    if len(trace) > 32:
        del trace[:-32]


def build_open_prompt(decision: str, branch: str, current_self: str, divergence: float, honesty: float, tones: List[str], memory_window: str) -> List[Dict[str, str]]:
    tone_text = ", ".join(tones) if tones else "reflective, warm"
    user_prompt = {
        "task": "open_other_screen",
        "decision_hinge": decision,
        "road_b_branch": branch,
        "road_b_current_self": current_self,
        "divergence": divergence,
        "honesty": honesty,
        "tones": tone_text,
        "memory_window": memory_window,
        "required_json_schema": {
            "other_name": "short label such as You@2018",
            "universe_id": "short fictional ID",
            "identity_line": "one sentence identity of Road B self",
            "opening_line": "first message from the other self, 45-75 words",
            "daily_scene": "concrete scene from a typical day, 35-65 words",
            "gift": "what Road B gained, one phrase",
            "cost": "what Road B gave up, one phrase",
            "insight_title": "short title without emoji",
            "insight": "observation linking both paths, non-advice, 25-55 words",
            "question_back": "one question the other self asks the user",
            "souvenir_line": "one sentence worth saving",
        },
    }
    return [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": pretty_dumps(user_prompt)}]


def build_answer_prompt(state: Dict[str, Any], question: str) -> List[Dict[str, str]]:
    user_prompt = {
        "task": "answer_as_road_b_self",
        "session_state": slim_state(state),
        "user_question": question,
        "required_json_schema": {
            "answer": "Road B self's answer, 60-110 words, concrete and balanced",
            "insight_title": "optional short title",
            "insight": "optional observation, 20-50 words",
            "question_back": "optional question back to the user",
        },
    }
    return [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": pretty_dumps(user_prompt)}]


def build_artifact_prompt(state: Dict[str, Any], artifact_type: str) -> List[Dict[str, str]]:
    spec = ARTIFACT_SPECS.get(artifact_type, ARTIFACT_SPECS["cost_ledger"])
    user_prompt = {
        "task": "generate_echo_artifact",
        "artifact_type": artifact_type,
        "artifact_label": spec["label"],
        "artifact_instruction": spec["instruction"],
        "artifact_tone": spec["tone"],
        "session_state": slim_state(state, include_turns=8),
        "required_json_schema": {
            "title": "short artifact title",
            "kicker": "short label such as COST LEDGER // SIGNAL -9",
            "body": "main artifact text, 60-110 words",
            "lines": ["three short content-related lines"],
            "question_back": "one question the artifact asks the user",
            "souvenir_seed": "one short phrase that can appear on a souvenir card",
        },
    }
    return [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": pretty_dumps(user_prompt)}]


def build_lens_prompt(state: Dict[str, Any], mode: str) -> List[Dict[str, str]]:
    user_prompt = {
        "task": "open_companion_lens",
        "mode": mode,
        "session_state": slim_state(state, include_turns=8),
        "required_json_schema": {
            "title": "short lens title",
            "body": "50-90 words, concrete, non-prescriptive",
            "lines": ["three short fragments"],
        },
    }
    return [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": pretty_dumps(user_prompt)}]


def build_final_prompt(state: Dict[str, Any]) -> List[Dict[str, str]]:
    user_prompt = {
        "task": "final_transmission_card",
        "session_state": slim_state(state, include_turns=10),
        "instructions": "Use the collected Echo Artifacts. Do not introduce random unrelated content. Make the final card feel earned from this session.",
        "required_json_schema": {
            "title": "short card title",
            "final_message": "final Road B transmission, 55-95 words",
            "gift": "short phrase from this session",
            "cost": "short phrase from this session",
            "last_line": "one haunting sentence related to the user's fork",
        },
    }
    return [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": pretty_dumps(user_prompt)}]


# -----------------------------------------------------------------------------
# API logic
# -----------------------------------------------------------------------------


def invoke_road_b(
    decision: str,
    branch: str,
    current_self: str,
    divergence: float = 0.62,
    honesty: float = 0.84,
    tones: Optional[List[str]] = None,
    memory_window: str = "Continuous (this session)",
) -> Dict[str, Any]:
    decision = normalize_text(decision, 2200)
    branch = normalize_text(branch, 1200) or "They took the path I did not take."
    current_self = normalize_text(current_self, 2200)
    tones = tones or ["Warm", "Reflective"]

    if len(decision) < 18:
        return error_response("Name the fork with a little more detail before invoking Road B.", kind="validation", status=400)
    if len(current_self) < 12:
        return error_response("Give the other self at least one concrete detail, years later.", kind="validation", status=400)
    if contains_crisis_signal(decision, branch, current_self):
        return crisis_payload()

    session_id = uuid.uuid4().hex
    session_label = make_session_label()
    backend_universe = make_universe_id()

    try:
        output = model_json(build_open_prompt(decision, branch, current_self, divergence, honesty, tones, memory_window))
    except Exception as exc:
        return error_response(str(exc), kind="model_error", status=503)

    universe_id = visible_field(output, "universe_id", fallback=backend_universe)[:80]
    other_name = visible_field(output, "other_name", fallback="You@RoadB")[:80]
    opening_line = visible_field(
        output,
        "opening_line",
        fallback="The signal arrived, but the first words broke in the crossing. Ask Road B one simple question to stabilize the screen.",
    )
    identity_line = visible_field(output, "identity_line", fallback="A fictional self from the road not taken.")
    daily_scene = visible_field(output, "daily_scene", fallback="The other day has not fully come into focus yet.")
    gift = visible_field(output, "gift", fallback="A different kind of courage")
    cost = visible_field(output, "cost", fallback="A tenderness left behind")
    insight = visible_field(output, "insight", fallback="Both roads protect something and ask something in return.")
    question_back = visible_field(output, "question_back", fallback="What did your chosen life protect that mine could not?")
    souvenir_line = visible_field(output, "souvenir_line", fallback="Do not worship the road you did not take.")

    state: Dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "session_id": session_id,
        "session_label": session_label,
        "universe_id": universe_id or backend_universe,
        "started_at": now_iso(),
        "updated_at": now_iso(),
        "signal": clamp_signal(92 - int(float(divergence) * 8)),
        "inputs": {
            "decision": decision,
            "branch": branch,
            "current_self": current_self,
            "divergence": float(divergence),
            "honesty": float(honesty),
            "tones": tones,
            "memory_window": memory_window,
        },
        "profile": {
            "other_name": other_name,
            "identity_line": identity_line,
            "daily_scene": daily_scene,
            "gift": gift,
            "cost": cost,
        },
        "opening": {
            "opening_line": opening_line,
            "insight_title": visible_field(output, "insight_title", fallback="A SIGNAL WORTH KEEPING"),
            "insight": insight,
            "question_back": question_back,
            "souvenir_line": souvenir_line,
        },
        "turns": [],
        "artifacts": [],
        "trace": [],
        "runtime": public_runtime_info(model_loaded=True),
    }
    append_trace(state, "open_other_screen", decision, output)

    return {
        "ok": True,
        "kind": "open_other_screen",
        "state": state,
        "state_json": dumps(state),
        "session_label": session_label,
        "universe_id": state["universe_id"],
        "other_name": other_name,
        "signal": state["signal"],
        "echo_count": 0,
        "final_unlocked": False,
        "opening_line": opening_line,
        "identity_line": identity_line,
        "daily_scene": daily_scene,
        "gift": gift,
        "cost": cost,
        "insight_title": state["opening"]["insight_title"],
        "insight": insight,
        "question_back": question_back,
        "souvenir_line": souvenir_line,
        "runtime": state["runtime"],
    }


def ask_other_self(state_json: str, question: str) -> Dict[str, Any]:
    question = normalize_text(question, 1400)
    if not question:
        return error_response("Write something to your other self first.", kind="validation", status=400)
    if contains_crisis_signal(question):
        return crisis_payload()
    try:
        state = parse_state(state_json)
    except Exception as exc:
        return error_response(str(exc), kind="validation", status=400)

    try:
        output = model_json(build_answer_prompt(state, question), max_tokens=min(MAX_TOKENS, 680))
    except Exception as exc:
        return error_response(str(exc), kind="model_error", status=503)

    answer = visible_field(output, "answer", fallback="The signal came through, but the words did not survive the crossing. Ask again, more simply.")
    insight_title = visible_field(output, "insight_title", fallback="")
    insight = visible_field(output, "insight", fallback="")
    question_back = visible_field(output, "question_back", fallback="")

    state.setdefault("turns", []).append({"role": "you", "content": question, "time": now_iso()})
    state.setdefault("turns", []).append(
        {
            "role": "other_self",
            "content": answer,
            "time": now_iso(),
            "insight_title": insight_title,
            "insight": insight,
            "question_back": question_back,
        }
    )
    state["signal"] = clamp_signal(int(state.get("signal", 80)) - 7)
    state["updated_at"] = now_iso()
    append_trace(state, "ask_other_self", question, output)

    return {
        "ok": True,
        "kind": "ask_other_self",
        "state": state,
        "state_json": dumps(state),
        "answer": answer,
        "signal": state["signal"],
        "echo_count": len(state.get("artifacts", [])),
        "final_unlocked": len(state.get("artifacts", [])) >= 3,
        "insight_title": insight_title,
        "insight": insight,
        "question_back": question_back,
        "runtime": public_runtime_info(model_loaded=True),
    }


def open_artifact(state_json: str, artifact_type: str) -> Dict[str, Any]:
    try:
        state = parse_state(state_json)
    except Exception as exc:
        return error_response(str(exc), kind="validation", status=400)

    artifact_type = (artifact_type or "cost_ledger").strip().lower()
    if artifact_type not in ARTIFACT_SPECS:
        artifact_type = "cost_ledger"
    spec = ARTIFACT_SPECS[artifact_type]

    # Return the existing artifact if already collected, so users do not accidentally spend signal twice.
    for existing in state.get("artifacts", []):
        if existing.get("artifact_type") == artifact_type:
            return {
                "ok": True,
                "kind": "artifact",
                "already_collected": True,
                "state": state,
                "state_json": dumps(state),
                "artifact": existing,
                "signal": state.get("signal", 80),
                "echo_count": len(state.get("artifacts", [])),
                "final_unlocked": len(state.get("artifacts", [])) >= 3,
            }

    try:
        output = model_json(build_artifact_prompt(state, artifact_type), max_tokens=min(MAX_TOKENS, 680))
    except Exception as exc:
        return error_response(str(exc), kind="model_error", status=503)

    title = visible_field(output, "title", fallback=spec["label"])
    kicker = visible_field(output, "kicker", fallback=f"{spec['label'].upper()} // SIGNAL -9")
    body = visible_field(output, "body", fallback="The artifact opened, but the signal did not hold. Try one more Road B question before returning to this room.")
    lines = visible_list(output, "lines")
    if not lines:
        lines = [body]
    question_back = visible_field(output, "question_back", fallback="What does this echo change about the road you chose?")
    souvenir_seed = visible_field(output, "souvenir_seed", fallback=(lines[0] if lines else title))

    artifact = {
        "artifact_type": artifact_type,
        "label": spec["label"],
        "title": title,
        "kicker": kicker,
        "body": body,
        "lines": lines[:5],
        "question_back": question_back,
        "souvenir_seed": souvenir_seed,
        "created_at": now_iso(),
    }
    state.setdefault("artifacts", []).append(artifact)
    state["signal"] = clamp_signal(int(state.get("signal", 80)) - 9)
    state["updated_at"] = now_iso()
    append_trace(state, f"artifact:{artifact_type}", spec["instruction"], output)

    return {
        "ok": True,
        "kind": "artifact",
        "state": state,
        "state_json": dumps(state),
        "artifact": artifact,
        "signal": state["signal"],
        "echo_count": len(state.get("artifacts", [])),
        "final_unlocked": len(state.get("artifacts", [])) >= 3,
        "runtime": public_runtime_info(model_loaded=True),
    }


def open_lens(state_json: str, mode: str) -> Dict[str, Any]:
    try:
        state = parse_state(state_json)
    except Exception as exc:
        return error_response(str(exc), kind="validation", status=400)
    mode = (mode or "reflect").lower().strip()
    if mode not in {"reflect", "discover", "grow"}:
        mode = "reflect"
    try:
        output = model_json(build_lens_prompt(state, mode), max_tokens=min(MAX_TOKENS, 560))
    except Exception as exc:
        return error_response(str(exc), kind="model_error", status=503)
    title = visible_field(output, "title", fallback=mode.title())
    body = visible_field(output, "body", fallback="The lens opened, but the image was unstable. Try again after one more Road B question.")
    lines = visible_list(output, "lines")
    lens = {"mode": mode, "title": title, "body": body, "lines": lines, "time": now_iso()}
    state.setdefault("lenses", []).append(lens)
    state["signal"] = clamp_signal(int(state.get("signal", 80)) - 5)
    state["updated_at"] = now_iso()
    append_trace(state, f"lens:{mode}", mode, output)
    return {
        "ok": True,
        "kind": "lens",
        "mode": mode,
        "state": state,
        "state_json": dumps(state),
        "title": title,
        "body": body,
        "lines": lines,
        "signal": state["signal"],
        "echo_count": len(state.get("artifacts", [])),
        "final_unlocked": len(state.get("artifacts", [])) >= 3,
    }


def final_transmission(state_json: str) -> Dict[str, Any]:
    try:
        state = parse_state(state_json)
    except Exception as exc:
        return error_response(str(exc), kind="validation", status=400)

    artifact_count = len(state.get("artifacts", []))
    if artifact_count < 3:
        return error_response(
            f"Collect {3 - artifact_count} more Echo Artifact(s) before the final transmission can stabilize.",
            kind="locked",
            status=423,
        )

    try:
        output = model_json(build_final_prompt(state), max_tokens=min(MAX_TOKENS, 620))
    except Exception as exc:
        return error_response(str(exc), kind="model_error", status=503)

    final_message = visible_field(output, "final_message", fallback="The final transmission flickered, but one sentence remained: do not worship the road you did not take.")
    final = {
        "title": visible_field(output, "title", fallback="Final Transmission"),
        "final_message": final_message,
        "gift": visible_field(output, "gift", fallback=state.get("profile", {}).get("gift", "A different kind of courage")),
        "cost": visible_field(output, "cost", fallback=state.get("profile", {}).get("cost", "A tenderness left behind")),
        "last_line": visible_field(output, "last_line", fallback=state.get("opening", {}).get("souvenir_line", "Do not worship the road you did not take.")),
        "time": now_iso(),
    }
    state["final"] = final
    state["signal"] = clamp_signal(min(int(state.get("signal", 80)), 22))
    state["updated_at"] = now_iso()
    append_trace(state, "final_transmission", "closing card from collected artifacts", output)

    return {
        "ok": True,
        "kind": "final_transmission",
        "state": state,
        "state_json": dumps(state),
        **final,
        "signal": state["signal"],
        "echo_count": len(state.get("artifacts", [])),
        "final_unlocked": True,
        "runtime": public_runtime_info(model_loaded=True),
    }


def boot_runtime() -> Dict[str, Any]:
    # Do not force-load the local model during boot when Modal is enabled.
    loaded = False if MODAL_QWEN_URL else load_llm.cache_info().currsize > 0
    return {"ok": True, "runtime": public_runtime_info(model_loaded=loaded)}


# -----------------------------------------------------------------------------
# App construction
# -----------------------------------------------------------------------------


def read_index() -> str:
    path = ROOT / "index.html"
    if not path.exists():
        return """
        <!doctype html><html><head><title>Road B</title></head>
        <body style='font-family:system-ui;background:#07040f;color:white;padding:32px'>
        <h1>Road B: The Other Screen</h1>
        <p>Missing index.html. Upload index.html to the Space root.</p>
        </body></html>
        """
    return path.read_text(encoding="utf-8")


if Server is None:
    if gr is None:
        raise RuntimeError("Gradio is not installed. This Space requires gradio>=6.14.0.")
    with gr.Blocks(title=APP_TITLE) as demo:  # pragma: no cover
        gr.Markdown(
            "# Road B requires gradio.Server\n\n"
            "This build uses a custom frontend served by `gradio.Server`. "
            "Please install `gradio>=6.14.0` and restart the Space."
        )
    demo.launch(show_error=True)
else:
    app = Server()

    @app.api(name="boot_runtime")
    def api_boot_runtime() -> Dict[str, Any]:
        return boot_runtime()

    @app.api(name="invoke_road_b")
    def api_invoke_road_b(
        decision: str,
        branch: str,
        current_self: str,
        divergence: float = 0.62,
        honesty: float = 0.84,
        tones: Optional[List[str]] = None,
        memory_window: str = "Continuous (this session)",
    ) -> Dict[str, Any]:
        return invoke_road_b(decision, branch, current_self, divergence, honesty, tones, memory_window)

    @app.api(name="ask_other_self")
    def api_ask_other_self(state_json: str, question: str) -> Dict[str, Any]:
        return ask_other_self(state_json, question)

    @app.api(name="open_artifact")
    def api_open_artifact(state_json: str, artifact_type: str) -> Dict[str, Any]:
        return open_artifact(state_json, artifact_type)

    @app.api(name="open_lens")
    def api_open_lens(state_json: str, mode: str) -> Dict[str, Any]:
        return open_lens(state_json, mode)

    @app.api(name="final_transmission")
    def api_final_transmission(state_json: str) -> Dict[str, Any]:
        return final_transmission(state_json)

    @app.get("/", response_class=HTMLResponse)
    async def homepage() -> Any:
        return HTMLResponse(read_index())

    @app.get("/health")
    async def health() -> Any:
        loaded = False if MODAL_QWEN_URL else load_llm.cache_info().currsize > 0
        payload = public_runtime_info(model_loaded=loaded)
        if JSONResponse is None:
            return payload
        return JSONResponse(payload)

    if StaticFiles is not None:
        if ASSET_DIR.exists():
            app.mount("/assets", StaticFiles(directory=str(ASSET_DIR)), name="assets")
        if DOCS_DIR.exists():
            app.mount("/docs", StaticFiles(directory=str(DOCS_DIR)), name="docs")
        if SAMPLES_DIR.exists():
            app.mount("/samples", StaticFiles(directory=str(SAMPLES_DIR)), name="samples")

    # Expose a conventional name for Gradio's Space loader/hot-reload tooling.
    demo = app
    demo.launch(show_error=True)
