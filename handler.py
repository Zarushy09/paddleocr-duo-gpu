"""
handler.py — Worker de Runpod Serverless para el DUO-GPU (rama `serverless`).

Diseño: el worker NO reimplementa el OCR. Arranca el MISMO `app.py` del pod
(uvicorn, mismos workers, mismo entorno de Python que la imagen base de Paddle)
en 127.0.0.1 y le reenvía cada trabajo a `POST /ocr`. Así la salida es la del
pod byte a byte: mismo código, mismas versiones, mismos pesos, misma GPU.

Este archivo corre en un entorno aparte (/opt/rp, Python 3.11 con el SDK de
Runpod). El SDK exige fastapi>=0.141 y el app.py certificado usa fastapi 0.115:
mezclarlos en un solo entorno cambiaría el servidor del OCR. Por eso van separados.

Contrato del trabajo (input):
  {"image_base64": "...", "imagen_nombre": "..."}   → OCR normal
  {"warmup": true}                                    → solo confirma que está listo
  {"diagnostico": true}                               → versiones, pesos, GPU y env

Salida: {"duo": <JSON de /ocr tal cual>, "rp": {...tiempos del worker...}}
  `duo` va envuelto porque el SDK de Runpod hace job_output.pop("error") sobre el
  dict de primer nivel: devolver el JSON del DUO directo le quitaría la clave
  `error` y, cuando no es null, marcaría el trabajo como fallido.
Si el servidor local no responde: {"error": "..."} → el trabajo queda FAILED
y n8n lo trata como «no respondió» (rama de Reintento).
"""

import asyncio
import json
import os
import subprocess
import sys
import time
import urllib.request

import aiohttp
import runpod

APP_DIR = os.getenv("DUO_APP_DIR", "/app")
PORT = int(os.getenv("DUO_PORT", "8014"))
WORKERS = int(os.getenv("UVICORN_WORKERS", "5"))
# Trabajos simultáneos por worker de Runpod. Por defecto = workers de uvicorn,
# igual que el pod: cada proceso uvicorn atiende una imagen a la vez por motor.
CONCURRENCY = int(os.getenv("JOB_CONCURRENCY", str(WORKERS)))
STARTUP_TIMEOUT = int(os.getenv("DUO_STARTUP_TIMEOUT", "600"))
REQUEST_TIMEOUT = float(os.getenv("DUO_REQUEST_TIMEOUT", "120"))
BASE = f"http://127.0.0.1:{PORT}"


def _log(msg: str) -> None:
    print(f"[duo-serverless] {msg}", flush=True)


def _app_python() -> str:
    """Python de la imagen base (el que tiene paddle). Lo fija el Dockerfile."""
    try:
        with open(os.path.join(APP_DIR, ".app_python"), encoding="utf-8") as f:
            p = f.read().strip()
            if p:
                return p
    except OSError:
        pass
    return "python"


APP_PYTHON = _app_python()


def _app_env() -> dict:
    """Entorno del servidor OCR: el del contenedor, sin rastros del venv del handler."""
    env = dict(os.environ)
    for k in ("PYTHONPATH", "VIRTUAL_ENV", "PYTHONHOME", "UV_PYTHON"):
        env.pop(k, None)
    venv_bin = os.path.dirname(sys.executable)
    env["PATH"] = os.pathsep.join(p for p in env.get("PATH", "").split(os.pathsep) if p and p != venv_bin)
    env["UVICORN_WORKERS"] = str(WORKERS)
    return env


def start_server() -> subprocess.Popen:
    cmd = [APP_PYTHON, "-m", "uvicorn", "app:app", "--host", "127.0.0.1",
           "--port", str(PORT), "--workers", str(WORKERS)]
    _log(f"arrancando el DUO: {' '.join(cmd)}")
    return subprocess.Popen(cmd, cwd=APP_DIR, env=_app_env())


def wait_ready(proc: subprocess.Popen) -> float:
    """Espera a que /health responda. Devuelve los segundos que tardó."""
    t0 = time.time()
    while time.time() - t0 < STARTUP_TIMEOUT:
        if proc.poll() is not None:
            raise RuntimeError(f"uvicorn terminó al arrancar (código {proc.returncode})")
        try:
            with urllib.request.urlopen(BASE + "/health", timeout=3) as r:
                if r.status == 200:
                    return time.time() - t0
        except Exception:  # noqa: BLE001 — todavía cargando modelos
            pass
        time.sleep(1)
    raise RuntimeError(f"/health no respondió en {STARTUP_TIMEOUT} s")


def run_diag() -> dict:
    """Corre diag.py con el Python de la app (mismo entorno que el OCR)."""
    try:
        out = subprocess.run([APP_PYTHON, os.path.join(APP_DIR, "diag.py"), "--json"],
                             cwd=APP_DIR, env=_app_env(), capture_output=True,
                             text=True, timeout=180)
        return json.loads(out.stdout) if out.returncode == 0 else {"error_diag": out.stderr[-2000:]}
    except Exception as e:  # noqa: BLE001
        return {"error_diag": str(e)}


# ---------------------------------------------------------------- arranque
_T_BOOT = time.time()
SERVER = start_server()
STARTUP_S = wait_ready(SERVER)
_log(f"DUO listo en {STARTUP_S:.1f} s (workers={WORKERS}, concurrencia={CONCURRENCY})")
DIAG = run_diag()
_log("diagnóstico: " + json.dumps({k: DIAG.get(k) for k in ("versiones", "modelos_sha256", "gpu", "env")}, ensure_ascii=False))

_session = None


async def _get_session() -> aiohttp.ClientSession:
    global _session
    if _session is None or _session.closed:
        _session = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=REQUEST_TIMEOUT))
    return _session


async def _health() -> dict:
    s = await _get_session()
    async with s.get(BASE + "/health") as r:
        return await r.json()


async def handler(job: dict) -> dict:
    t0 = time.time()
    inp = job.get("input") or {}

    if SERVER.poll() is not None:
        # El servidor OCR murió: fallar y reciclar el worker.
        return {"error": f"servidor DUO caído (código {SERVER.returncode})", "refresh_worker": True}

    if inp.get("warmup"):
        return {"warm": True, "health": await _health(),
                "rp": {"arranque_s": round(STARTUP_S, 1), "worker_ms": int((time.time() - t0) * 1000)}}

    if inp.get("diagnostico"):
        return {"diagnostico": DIAG, "health": await _health(),
                "rp": {"arranque_s": round(STARTUP_S, 1)}}

    img = inp.get("image_base64")
    if not img or not isinstance(img, str):
        return {"error": "input.image_base64 vacío o inválido"}

    payload = {"image_base64": img, "imagen_nombre": inp.get("imagen_nombre")}
    try:
        s = await _get_session()
        async with s.post(BASE + "/ocr", json=payload) as r:
            body = await r.text()
            if r.status != 200:
                return {"error": f"DUO HTTP {r.status}: {body[:500]}"}
            duo = json.loads(body)
    except (aiohttp.ClientError, asyncio.TimeoutError) as e:
        return {"error": f"DUO sin respuesta: {type(e).__name__}: {e}"}

    return {"duo": duo, "rp": {"worker_ms": int((time.time() - t0) * 1000),
                               "arranque_s": round(STARTUP_S, 1)}}


if __name__ == "__main__":
    runpod.serverless.start({"handler": handler, "concurrency_modifier": lambda _c: CONCURRENCY})
