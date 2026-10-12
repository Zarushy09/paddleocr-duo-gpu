"""
diag.py — Huella del entorno del DUO, para comparar el pod con el worker serverless.

Corre igual en el pod y dentro del worker (solo librería estándar; NO importa paddle,
así que no toca la GPU ni tarda). Reporta:
  - versiones de los paquetes que deciden la salida del OCR y huella de `pip freeze`
  - sha256 de los pesos de los modelos en caché (official_models)
  - GPU, driver y CUDA (nvidia-smi)
  - variables de entorno del DUO, leídas del proceso uvicorn en marcha si lo hay
  - sha256 de app.py

Uso en el pod (terminal web de Runpod):
  curl -sL https://raw.githubusercontent.com/Zarushy09/paddleocr-duo-gpu/serverless/diag.py | python - --json
  ... | python - --freeze      → imprime el `pip freeze` completo (para constraints-pod.txt)
"""

import hashlib
import json
import os
import platform
import subprocess
import sys
from importlib import metadata

CLAVE = ["paddlepaddle-gpu", "paddlepaddle", "paddleocr", "paddlex", "numpy",
         "opencv-python-headless", "opencv-contrib-python", "opencv-python",
         "pyclipper", "shapely", "Pillow", "scikit-image", "fastapi", "pydantic",
         "uvicorn", "ujson", "pypdfium2"]
ENV_DUO = ["DUO_RESIZE_MAX_DIM", "OCR_THREADS", "UVICORN_WORKERS", "DUO_SHED_INFLIGHT",
           "DUO_DEVICE", "BACKEND_TIMEOUT", "PADDLE_PDX_MODEL_SOURCE", "PADDLE_PDX_CACHE_HOME",
           "FLAGS_use_cuda_managed_memory", "CUDA_VISIBLE_DEVICES"]


def freeze() -> list:
    out = set()
    for d in metadata.distributions():
        name = d.metadata.get("Name")
        if name:
            out.add(f"{name}=={d.version}")
    return sorted(out, key=str.lower)


def versiones() -> dict:
    v = {}
    for p in CLAVE:
        try:
            v[p] = metadata.version(p)
        except metadata.PackageNotFoundError:
            pass
    return v


def cache_dir() -> str:
    return os.environ.get("PADDLE_PDX_CACHE_HOME") or os.path.join(os.path.expanduser("~"), ".paddlex")


def modelos() -> dict:
    base = os.path.join(cache_dir(), "official_models")
    res = {}
    if not os.path.isdir(base):
        return res
    for m in sorted(os.listdir(base)):
        d = os.path.join(base, m)
        if not os.path.isdir(d):
            continue
        h = hashlib.sha256()
        for root, _dirs, files in sorted(os.walk(d)):
            for fn in sorted(files):
                if fn.endswith(".lock"):
                    continue
                p = os.path.join(root, fn)
                h.update(os.path.relpath(p, d).encode())
                with open(p, "rb") as f:
                    for chunk in iter(lambda: f.read(1 << 20), b""):
                        h.update(chunk)
        res[m] = h.hexdigest()[:16]
    return res


def freeze_vs_pod(fz: list):
    """Diferencias contra freeze-pod.txt (si existe junto a este script)."""
    ref = os.path.join(os.path.dirname(os.path.abspath(__file__)), "freeze-pod.txt")
    if not os.path.isfile(ref):
        return None
    norm = lambda x: x.strip().lower().replace("_", "-")  # noqa: E731
    pod = {norm(x) for x in open(ref, encoding="utf-8") if x.strip()}
    aqui = {norm(x) for x in fz if x.strip()}
    return {"solo_aqui": sorted(aqui - pod), "solo_pod": sorted(pod - aqui)}


def modelos_archivos() -> list:
    """Huella archivo por archivo (sha256[:16], tamaño, ruta): separa pesos de metadatos."""
    base = os.path.join(cache_dir(), "official_models")
    out = []
    if not os.path.isdir(base):
        return out
    for root, _dirs, files in sorted(os.walk(base)):
        for fn in sorted(files):
            p = os.path.join(root, fn)
            h = hashlib.sha256()
            with open(p, "rb") as f:
                for chunk in iter(lambda: f.read(1 << 20), b""):
                    h.update(chunk)
            out.append(f"{h.hexdigest()[:16]} {os.path.getsize(p)} {os.path.relpath(p, base)}")
    return sorted(out, key=lambda x: x.split(" ", 2)[2])


def gpu() -> dict:
    try:
        o = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version,memory.total",
                            "--format=csv,noheader"], capture_output=True, text=True, timeout=10)
        name, driver, mem = [x.strip() for x in o.stdout.strip().splitlines()[0].split(",")]
        cuda = ""
        o2 = subprocess.run(["nvidia-smi"], capture_output=True, text=True, timeout=10)
        for line in o2.stdout.splitlines():
            if "CUDA Version" in line:
                cuda = line.split("CUDA Version:")[1].split()[0]
        return {"nombre": name, "driver": driver, "memoria": mem, "cuda_driver": cuda}
    except Exception as e:  # noqa: BLE001
        return {"error": str(e)}


def env_uvicorn() -> dict:
    """Variables del DUO tal como las ve el proceso uvicorn en marcha (si lo hay)."""
    for pid in os.listdir("/proc"):
        if not pid.isdigit():
            continue
        try:
            with open(f"/proc/{pid}/cmdline", "rb") as f:
                cmd = f.read().replace(b"\0", b" ").decode(errors="replace")
            if "uvicorn" not in cmd or "app:app" not in cmd:
                continue
            with open(f"/proc/{pid}/environ", "rb") as f:
                pares = [x.split(b"=", 1) for x in f.read().split(b"\0") if b"=" in x]
            env = {k.decode(): v.decode(errors="replace") for k, v in pares}
            return {"origen": f"uvicorn pid {pid}", **{k: env[k] for k in ENV_DUO if k in env}}
        except OSError:
            continue
    return {"origen": "shell (no hay uvicorn en marcha)", **{k: os.environ[k] for k in ENV_DUO if k in os.environ}}


def app_sha() -> str:
    for c in [os.path.join(os.getcwd(), "app.py"), "/app/app.py", "/root/paddleocr-duo-gpu/app.py"]:
        if os.path.isfile(c):
            with open(c, "rb") as f:
                return f"{c}: {hashlib.sha256(f.read()).hexdigest()[:16]}"
    return ""


def main() -> None:
    if "--freeze" in sys.argv:
        print("\n".join(freeze()))
        return
    fz = freeze()
    data = {
        "python": platform.python_version(),
        "versiones": versiones(),
        "freeze_sha256": hashlib.sha256("\n".join(fz).encode()).hexdigest()[:16],
        "paquetes": len(fz),
        "modelos_sha256": modelos(),
        "modelos_archivos": modelos_archivos(),
        "freeze_vs_pod": freeze_vs_pod(fz),
        "gpu": gpu(),
        "env": env_uvicorn(),
        "app_py": app_sha(),
    }
    print(json.dumps(data, ensure_ascii=False, indent=None if "--json" in sys.argv else 2))


if __name__ == "__main__":
    main()
