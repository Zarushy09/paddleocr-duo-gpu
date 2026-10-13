"""
handler.py — Worker de Runpod Serverless para el DUO-GPU (rama `serverless`).

Diseño: el worker NO reimplementa el OCR. Arranca el MISMO `app.py` del pod
(uvicorn, mismos workers, mismo entorno de Python que la imagen base de Paddle)
en 127.0.0.1 y le reenvía cada trabajo a `POST /ocr`. Así la salida es la del
pod byte a byte: mismo código, mismas versiones, mismos pesos, misma GPU.

Este archivo corre en un entorno aparte (/opt/rp, Python 3.11 con el SDK de
Runpod). El SDK exige fastapi>=0.141 y el app.py certificado usa fastapi 0.115:
mezclarlos en un solo entorno cambiaría el servidor del OCR. Por eso van separados.

Arranque en frío (v2, 13-oct-2026). Solo cambia el ORDEN del arranque; app.py,
pesos, variables y la salida del OCR quedan iguales:
  1. uvicorn se lanza ANTES de importar el SDK de Runpod y aiohttp (~1,5 s que
     ahora corren mientras el DUO carga los modelos).
  2. /health se consulta cada 0,25 s (antes cada 1 s).
  3. diag.py ya no corre al arrancar: se calcula la primera vez que llega
     {"diagnostico": true} y queda guardado.
  4. La salida lleva rp.boot con los tiempos de cada fase, para medir desde n8n.

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

import time

_T_BOOT = time.time()  # primera línea útil: todo rp.boot se mide desde aquí

import asyncio  # noqa: E402
import base64  # noqa: E402
import json  # noqa: E402
import os  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
import threading  # noqa: E402
import urllib.request  # noqa: E402
from concurrent.futures import ThreadPoolExecutor  # noqa: E402

APP_DIR = os.getenv("DUO_APP_DIR", "/app")
PORT = int(os.getenv("DUO_PORT", "8014"))
WORKERS = int(os.getenv("UVICORN_WORKERS", "5"))
# Trabajos simultáneos por worker de Runpod. Por defecto = workers de uvicorn,
# igual que el pod: cada proceso uvicorn atiende una imagen a la vez por motor.
CONCURRENCY = int(os.getenv("JOB_CONCURRENCY", str(WORKERS)))
STARTUP_TIMEOUT = int(os.getenv("DUO_STARTUP_TIMEOUT", "600"))
REQUEST_TIMEOUT = float(os.getenv("DUO_REQUEST_TIMEOUT", "120"))
HEALTH_POLL_S = float(os.getenv("DUO_HEALTH_POLL_S", "0.25"))
BASE = f"http://127.0.0.1:{PORT}"
# Si un proceso del OCR se cae a mitad de un trabajo (visto: SIGSEGV de Paddle en
# MatmulKernel con el driver de algunos hosts), uvicorn lo vuelve a levantar solo y
# el otro proceso sigue vivo. El handler espera /health y repite la MISMA imagen
# una vez: la salida es determinista, así que reintentar no cambia el resultado.
RETRY_WAIT = float(os.getenv("DUO_RETRY_WAIT", "90"))
WARM_IMAGE = os.path.join(APP_DIR, "warm.jpg")


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
        time.sleep(HEALTH_POLL_S)
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
# 1) El DUO primero: la carga de modelos es lo más largo del arranque.
_T_SERVER = time.time()
SERVER = start_server()

# 2) Mientras el DUO carga, importar el SDK de Runpod y aiohttp (~1,5 s).
import aiohttp  # noqa: E402
import runpod  # noqa: E402

_T_IMPORTS = time.time()

# 3) Esperar a que el DUO esté sano.
wait_ready(SERVER)
STARTUP_S = time.time() - _T_SERVER  # desde que se lanzó uvicorn hasta /health 200 (mismo sentido que antes)
_log(f"DUO listo en {STARTUP_S:.1f} s (workers={WORKERS}, concurrencia={CONCURRENCY})")


def warm_processes() -> dict:
    """Primera inferencia de cada proceso uvicorn ANTES de aceptar trabajos.

    La primera pasada por la GPU de cada proceso tarda ~6 s (carga de kernels);
    se manda una imagen con texto varias veces en paralelo para que pasen todos.
    No cambia ninguna salida: cada petición al DUO es independiente.
    """
    if not os.path.isfile(WARM_IMAGE):
        return {"omitido": "sin warm.jpg"}
    with open(WARM_IMAGE, "rb") as f:
        data = json.dumps({"image_base64": base64.b64encode(f.read()).decode(),
                           "imagen_nombre": "warmup.jpg"}).encode()

    def uno(_i):
        t = time.time()
        try:
            req = urllib.request.Request(BASE + "/ocr", data=data, method="POST",
                                         headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=REQUEST_TIMEOUT) as r:
                ok = json.loads(r.read().decode()).get("success") is True
            return {"ok": ok, "ms": int((time.time() - t) * 1000)}
        except Exception as e:  # noqa: BLE001
            return {"ok": False, "ms": int((time.time() - t) * 1000), "error": f"{type(e).__name__}: {e}"}

    t0 = time.time()
    with ThreadPoolExecutor(max_workers=WORKERS * 2) as ex:
        res = list(ex.map(uno, range(WORKERS * 2)))
    if not all(r["ok"] for r in res):
        # Si el calentado tumbó un proceso, uvicorn lo relanza: esperar a que esté sano.
        wait_ready(SERVER)
    return {"s": round(time.time() - t0, 1), "peticiones": res}


_T_WARM0 = time.time()
WARM = warm_processes()
_T_READY = time.time()
_log(f"calentado: {json.dumps(WARM, ensure_ascii=False)}")

# Tiempos del arranque (segundos). total = desde que Python empezó hasta aceptar trabajos.
BOOT = {
    "imports_s": round(_T_IMPORTS - _T_SERVER, 2),      # SDK + aiohttp, en paralelo a la carga
    "modelos_s": round(STARTUP_S, 2),                   # uvicorn lanzado → /health 200
    "calentado_s": round(_T_READY - _T_WARM0, 2),       # primera pasada de cada proceso
    "total_s": round(_T_READY - _T_BOOT, 2),
}
_log("arranque: " + json.dumps(BOOT))

# 4) diag.py ya no va en el arranque: se calcula al primer {"diagnostico": true}.
_DIAG = None
_DIAG_LOCK = threading.Lock()


def get_diag() -> dict:
    global _DIAG
    with _DIAG_LOCK:
        if _DIAG is None:
            _DIAG = run_diag()
            _log("diagnóstico: " + json.dumps({k: _DIAG.get(k) for k in ("versiones", "modelos_sha256", "gpu", "env")}, ensure_ascii=False))
        return _DIAG


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


async def _esperar_sano(limite_s: float) -> bool:
    t = time.time()
    while time.time() - t < limite_s:
        if SERVER.poll() is not None:
            return False
        try:
            await _health()
            return True
        except Exception:  # noqa: BLE001
            await asyncio.sleep(1)
    return False


async def _post_ocr(payload: dict):
    s = await _get_session()
    async with s.post(BASE + "/ocr", json=payload) as r:
        return r.status, await r.text()


async def handler(job: dict) -> dict:
    t0 = time.time()
    inp = job.get("input") or {}

    if SERVER.poll() is not None:
        # El servidor OCR murió: fallar y reciclar el worker.
        return {"error": f"servidor DUO caído (código {SERVER.returncode})", "refresh_worker": True}

    if inp.get("warmup"):
        return {"warm": True, "health": await _health(),
                "rp": {"arranque_s": round(STARTUP_S, 1), "boot": BOOT,
                       "worker_ms": int((time.time() - t0) * 1000)}}

    if inp.get("diagnostico"):
        diag = await asyncio.get_running_loop().run_in_executor(None, get_diag)
        return {"diagnostico": diag, "health": await _health(),
                "rp": {"arranque_s": round(STARTUP_S, 1), "boot": BOOT}}

    img = inp.get("image_base64")
    if not img or not isinstance(img, str):
        return {"error": "input.image_base64 vacío o inválido"}

    payload = {"image_base64": img, "imagen_nombre": inp.get("imagen_nombre")}
    reintento = None
    try:
        try:
            status, body = await _post_ocr(payload)
        except (aiohttp.ServerDisconnectedError, aiohttp.ClientConnectionError) as e:
            # Se cayó el proceso que atendía esta imagen: esperar /health y repetir una vez.
            caida = f"{type(e).__name__}: {e}"
            _log(f"CAIDA del proceso OCR ({caida}) con {inp.get('imagen_nombre')!r}: reintento")
            t_r = time.time()
            if not await _esperar_sano(RETRY_WAIT):
                return {"error": f"DUO sin respuesta tras caída: {caida}"}
            status, body = await _post_ocr(payload)
            reintento = {"caida": caida, "espera_ms": int((time.time() - t_r) * 1000)}
        if status != 200:
            return {"error": f"DUO HTTP {status}: {body[:500]}"}
        duo = json.loads(body)
    except (aiohttp.ClientError, asyncio.TimeoutError) as e:
        return {"error": f"DUO sin respuesta: {type(e).__name__}: {e}"}

    rp = {"worker_ms": int((time.time() - t0) * 1000), "arranque_s": round(STARTUP_S, 1), "boot": BOOT}
    if reintento:
        rp["reintento"] = reintento
    return {"duo": duo, "rp": rp}


if __name__ == "__main__":
    runpod.serverless.start({"handler": handler, "concurrency_modifier": lambda _c: CONCURRENCY})
