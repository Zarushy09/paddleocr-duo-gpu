#!/usr/bin/env bash
# =============================================================================
# bootstrap.sh — DUO-GPU self-install para Runpod (CERO docker build)
#
# Corre DENTRO del pod sobre la imagen oficial:
#   paddlepaddle/paddle:3.2.2-gpu-cuda12.6-cudnn9.5
# Descarga app.py + requirements.txt desde el repo, instala deps, CALIENTA los
# modelos (una inferencia dummy por pipeline → descarga + cachea pesos en el disco
# del pod) y lanza uvicorn en :8014. Idempotente: si /root/.duo_ready existe,
# salta instalacion/calentado y relanza directo (re-starts rapidos).
#
# START COMMAND en Runpod (con fallback curl→wget):
#   bash -c "curl -sL <RAW_URL_BOOTSTRAP> | bash || wget -qO- <RAW_URL_BOOTSTRAP> | bash"
# =============================================================================
set -euo pipefail

# Base RAW del repo. Overrideable por env (util para forks/branches).
RAW_BASE="${RAW_BASE:-https://raw.githubusercontent.com/Zarushy09/paddleocr-duo-gpu/main}"
APP_DIR="${DUO_DIR:-/root/paddleocr-duo-gpu}"
READY="/root/.duo_ready"
PORT="${PORT:-8014}"

log(){ echo -e "\n\033[1;36m[duo-gpu]\033[0m $*"; }

fetch(){  # fetch <archivo>: curl con fallback a wget
  curl -fsSL "$RAW_BASE/$1" -o "$1" || wget -qO "$1" "$RAW_BASE/$1"
}

mkdir -p "$APP_DIR"
cd "$APP_DIR"

log "⬇️  Descargando app.py + requirements.txt desde $RAW_BASE ..."
fetch app.py
fetch requirements.txt

if [ -f "$READY" ]; then
  log "✅ $READY presente — salto pip install y calentado (re-start rapido)."
else
  log "📦 Instalando dependencias (pip install -r requirements.txt) ..."
  python -m pip install --no-cache-dir -r requirements.txt

  log "🔥 Calentando modelos: inferencia dummy con medium y small (descarga + cachea pesos) ..."
  python - <<'PYWARM'
import numpy as np

# Device: GPU si el pod la expone, si no CPU (mismo criterio que app.py).
def resolve_device():
    try:
        import paddle
        if paddle.device.is_compiled_with_cuda() and paddle.device.cuda.device_count() > 0:
            return "gpu"
    except Exception as e:
        print(f"[warm] deteccion GPU fallo ({e}) -> cpu", flush=True)
    return "cpu"

DEVICE = resolve_device()
print(f"[warm] device = {DEVICE}", flush=True)

from paddleocr import PaddleOCR

# Kwargs de calidad IDENTICOS a app.py / motores certificados.
kw = dict(
    device=DEVICE,
    use_textline_orientation=True,
    use_doc_orientation_classify=True,
    use_doc_unwarping=False,
    text_det_limit_type="min",
    text_det_limit_side_len=64,
    text_det_thresh=0.2,
    text_det_box_thresh=0.4,
    text_det_unclip_ratio=2.0,
)

dummy = np.full((480, 640, 3), 255, dtype=np.uint8)  # lienzo blanco para forzar el pipeline

print("[warm] medium: instanciando + inferencia dummy ...", flush=True)
medium = PaddleOCR(text_detection_model_name="PP-OCRv6_medium_det",
                   text_recognition_model_name="PP-OCRv6_medium_rec", **kw)
medium.predict(dummy)
print("[warm] medium OK", flush=True)

print("[warm] small: instanciando + inferencia dummy ...", flush=True)
small = PaddleOCR(text_detection_model_name="PP-OCRv6_small_det",
                  text_recognition_model_name="PP-OCRv6_small_rec", **kw)
small.predict(dummy)
print("[warm] small OK", flush=True)

print("[warm] ambos pipelines calientes y cacheados.", flush=True)
PYWARM

  touch "$READY"
  log "🧊→🔥 Modelos cacheados. Marcador creado: $READY"
fi

log "🚀 Lanzando uvicorn app:app en :$PORT (2 workers) ..."
python -m uvicorn app:app --host 0.0.0.0 --port "$PORT" --workers 2 &
UVPID=$!

# Espera activa a que /health responda para anunciar "listo" (hasta ~4 min).
for _ in $(seq 1 120); do
  if curl -fsS "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then
    log "🟢 DUO-GPU listo en :$PORT"
    break
  fi
  sleep 2
done

wait "$UVPID"
