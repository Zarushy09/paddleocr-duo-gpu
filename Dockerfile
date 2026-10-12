# =============================================================================
# DUO-GPU para Runpod Serverless (rama `serverless`)
#
# Misma base, mismo app.py, mismos requirements y mismos pesos que el pod de
# producción: la salida del OCR debe ser idéntica byte a byte. Lo único nuevo es
# el handler de Runpod, que vive en su propio entorno (/opt/rp) para no tocar el
# Python del OCR.
#
# Runpod lo construye desde GitHub (Serverless → New Endpoint → Import Git
# Repository → rama `serverless`, Dockerfile en la raíz). Sin GPU en el build:
# los pesos se bajan con device=cpu y quedan dentro de la imagen.
# =============================================================================
FROM paddlepaddle/paddle:3.2.2-gpu-cuda12.6-cudnn9.5

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PADDLE_PDX_CACHE_HOME=/opt/paddlex \
    PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK=True \
    DUO_APP_DIR=/app \
    DUO_PORT=8014 \
    UVICORN_WORKERS=5

# Mismas librerías nativas que instala el bootstrap.sh del pod (cv2 → libGL.so.1).
RUN apt-get update \
 && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 1) Entorno del OCR: el Python de la imagen base + requirements.txt del pod,
#    con las versiones fijadas a las del pod (constraints-pod.txt).
COPY requirements.txt constraints-pod.txt ./
RUN python -m pip install -r requirements.txt -c constraints-pod.txt
# Informativo: conflictos de dependencias quedan en el log del build.
RUN python -m pip check || echo "[build] pip check con avisos (ver arriba)"
RUN command -v python > /app/.app_python

# 2) Pesos dentro de la imagen (el worker arranca sin descargar nada).
COPY warm_models.py diag.py ./
RUN DUO_DEVICE=cpu python warm_models.py
# Huella en el log del build: versiones, pesos (sha256) y freeze. Se compara con el pod.
RUN python diag.py

# 3) Entorno del handler: Python 3.11 aislado con el SDK de Runpod.
COPY requirements-handler.txt ./
ENV UV_PYTHON_INSTALL_DIR=/opt/uvpython
RUN python -m pip install --target /opt/uvtool "uv==0.12.24" \
 && /opt/uvtool/bin/uv venv /opt/rp --python 3.11 \
 && /opt/uvtool/bin/uv pip install --python /opt/rp/bin/python -r requirements-handler.txt \
 && /opt/rp/bin/python -c "import runpod, aiohttp; print('[build] handler: runpod', runpod.__version__)" \
 && rm -rf /opt/uvtool /root/.cache/uv

# 4) Código: app.py sin cambios (el mismo de `main`), handler y diagnóstico.
COPY app.py handler.py ./

CMD ["/opt/rp/bin/python", "-u", "/app/handler.py"]
