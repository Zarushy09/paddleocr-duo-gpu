"""
warm_models.py — Descarga y deja en la imagen los pesos de los dos motores.

Se ejecuta en el `docker build` (sin GPU: device=cpu). Usa los MISMOS modelos y
kwargs de calidad que app.py y el calentado del bootstrap.sh del pod, y corre una
inferencia sobre un lienzo blanco para que PaddleX baje también los modelos de
orientación (documento y línea de texto). Al final verifica que la caché quedó
completa; si falta algo, el build falla aquí y no en el primer arranque.
"""

import os
import sys

import numpy as np
from paddleocr import PaddleOCR

kw = dict(
    device="cpu",
    use_textline_orientation=True,
    use_doc_orientation_classify=True,
    use_doc_unwarping=False,
    text_det_limit_type="min",
    text_det_limit_side_len=64,
    text_det_thresh=0.2,
    text_det_box_thresh=0.4,
    text_det_unclip_ratio=2.0,
)
dummy = np.full((480, 640, 3), 255, dtype=np.uint8)

for nombre, det, rec in [("medium", "PP-OCRv6_medium_det", "PP-OCRv6_medium_rec"),
                         ("small", "PP-OCRv6_small_det", "PP-OCRv6_small_rec")]:
    print(f"[warm] {nombre}: instanciando + inferencia dummy ...", flush=True)
    eng = PaddleOCR(text_detection_model_name=det, text_recognition_model_name=rec, **kw)
    eng.predict(dummy)
    print(f"[warm] {nombre} OK", flush=True)

cache = os.environ.get("PADDLE_PDX_CACHE_HOME") or os.path.join(os.path.expanduser("~"), ".paddlex")
base = os.path.join(cache, "official_models")
presentes = sorted(os.listdir(base)) if os.path.isdir(base) else []
print(f"[warm] modelos en {base}: {presentes}", flush=True)

requeridos = ["PP-OCRv6_medium_det", "PP-OCRv6_medium_rec", "PP-OCRv6_small_det", "PP-OCRv6_small_rec"]
faltan = [m for m in requeridos if m not in presentes]
# Además de los 4 de texto deben estar los de orientación (documento y línea).
if faltan or len(presentes) < 6:
    print(f"[warm] ERROR: caché incompleta. Faltan {faltan}; presentes={len(presentes)}", flush=True)
    sys.exit(1)
print("[warm] caché completa: la imagen arranca sin descargar nada.", flush=True)
