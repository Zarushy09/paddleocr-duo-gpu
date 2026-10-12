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

# Imagen con texto para calentar los procesos del worker al arrancar (handler.py).
# Se genera aquí, en el entorno del OCR (Pillow), y queda en /app/warm.jpg.
try:
    from PIL import Image, ImageDraw, ImageFont

    lineas = ["FACTURA DE CREDITO FISCAL", "RNC: 101137142", "NCF: B0100000001",
              "FECHA: 12/10/2026", "SUBTOTAL 1,000.00", "ITBIS 18% 180.00", "TOTAL RD$ 1,180.00"]
    lienzo = Image.new("RGB", (900, 120 + 70 * len(lineas)), "white")
    pincel = ImageDraw.Draw(lienzo)
    try:
        fuente = ImageFont.load_default(size=40)
    except Exception:  # noqa: BLE001
        fuente = ImageFont.load_default()
    for i, t in enumerate(lineas):
        pincel.text((40, 50 + 70 * i), t, fill="black", font=fuente)
    destino = os.path.join(os.path.dirname(os.path.abspath(__file__)), "warm.jpg")
    lienzo.save(destino, quality=92)
    dummy = np.array(lienzo)
    print(f"[warm] imagen de calentado: {destino}", flush=True)
except Exception as e:  # noqa: BLE001
    print(f"[warm] AVISO: sin imagen de calentado ({e}); se usa lienzo en blanco", flush=True)

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

# Puerta de pesos: cada archivo de pesos/config debe ser IDÉNTICO al del pod de producción.
import hashlib  # noqa: E402

manifiesto = os.path.join(os.path.dirname(os.path.abspath(__file__)), "modelos-pod.txt")
if os.path.isfile(manifiesto):
    malos = []
    for linea in open(manifiesto, encoding="utf-8"):
        linea = linea.strip()
        if not linea or linea.startswith("#"):
            continue
        h_pod, tam_pod, ruta = linea.split(" ", 2)
        p = os.path.join(base, ruta)
        if not os.path.isfile(p):
            malos.append(f"FALTA {ruta}")
            continue
        h = hashlib.sha256()
        with open(p, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        h_aqui, tam_aqui = h.hexdigest()[:16], os.path.getsize(p)
        if h_aqui != h_pod or str(tam_aqui) != tam_pod:
            malos.append(f"DISTINTO {ruta}: pod {h_pod}/{tam_pod} vs build {h_aqui}/{tam_aqui}")
    for m in sorted(os.listdir(base)):
        meta = os.path.join(base, m, ".cache", "huggingface", "download", "inference.pdiparams.metadata")
        if os.path.isfile(meta):
            print(f"[warm] {m} revisión HF: {open(meta).readline().strip()}", flush=True)
    if malos:
        print("[warm] ERROR: los pesos NO son los del pod:\n  " + "\n  ".join(malos), flush=True)
        sys.exit(1)
    print("[warm] pesos idénticos a los del pod (manifiesto modelos-pod.txt).", flush=True)
else:
    print("[warm] AVISO: sin modelos-pod.txt, no se verificaron los pesos contra el pod.", flush=True)
