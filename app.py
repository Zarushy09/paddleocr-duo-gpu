"""
PaddleOCR DUO GPU (:8014) — DUO v2.2+shed en UN SOLO contenedor, ambos motores EN PROCESO.

Deriva del duo v2.2 (agregador HTTP a :8012 medium / :8013 small). Diferencia:
en vez de POST HTTP a dos backends, carga AMBOS pipelines PaddleOCR EN PROCESO
(configs medium y small IDENTICAS a las certificadas — mismos modelos y parametros
de calidad) con device GPU (auto-fallback a CPU si no hay GPU). Cada request corre
ambos motores (en paralelo via threads; si la GPU serializa, degrada a secuencial
de facto — se mide y reporta small_ms/medium_ms/wall_ocr_ms/parallel en fusion_stats).

MISMA API /ocr (image_base64, imagen_nombre) y MISMO esquema de respuesta
(success, text, blocks, total_lines, processing_time_ms, error, alternates,
engines_used, fusion_stats{small,medium,fusion,solo_small,solo_medium,alternates,
exif_transposed,resized,load_shed, + timings}). Misma fusion IoU>0.45 + alternas +
prepaso exif/resize. Mismo load shedding adaptativo (DUO_SHED_INFLIGHT).
"""

import asyncio
import base64
import io
import logging
import os
import re
import statistics
import threading
import time
import unicodedata
from typing import Optional

# ===== THREAD CAPS (anti-oversubscription) — identico a los motores certificados =====
_OCR_THREADS = os.getenv("OCR_THREADS", "4")
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "MKLDNN_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, _OCR_THREADS)

from fastapi import FastAPI
from pydantic import BaseModel
from PIL import Image, ImageOps
import numpy as np
import cv2
cv2.setNumThreads(int(_OCR_THREADS))

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
logging.basicConfig(level=LOG_LEVEL, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("paddleocr-duo-gpu")

# ===== Parametros del agregador (identicos al duo v2.2) =====
BACKEND_TIMEOUT = float(os.getenv("BACKEND_TIMEOUT", "90"))
IOU_THRESHOLD = 0.45
MIN_DIGITS_DIVERGENCE = 6
# Techo de resolucion del pre-paso del DUO: solo reescala si la imagen es PESADA
# (> RESIZE_MIN_MP MP) Y su lado largo supera RESIZE_MAX_DIM px. Certificadas (<=4 MP) intactas.
RESIZE_MAX_DIM = 2000
RESIZE_MIN_MP = 4.0
# Load shedding adaptativo POR WORKER. Default 64 = "adaptativo dormido": bajo carga
# normal jamas se dispara (imposible tener >64 requests en vuelo por worker aqui), asi
# que ambos motores corren siempre; queda como red de seguridad ante tormenta.
DUO_SHED_INFLIGHT = int(os.getenv("DUO_SHED_INFLIGHT", "64"))
in_flight = 0

# Techo de resolucion DE MOTOR (identico a los motores certificados: 4096px lado largo).
MAX_IMAGE_DIMENSION = 4096

# ===== AUTO-SANACION 180° v2 (SOLO medium — identico al motor v6 medium certificado) =====
ROT_HDR_RE = re.compile(r'FACTURA|RNC|NCF|FECHA DE EMISI', re.I)
ROT_TOT_RE = re.compile(r'TOTAL|SUB-?TOTAL', re.I)
ROT_GEO_MARGIN = 0.15
ROT_CONF_THRESHOLD = 0.75
ROT_MIN_SOSPECHOSOS = 4
ROT_MIN_RATIO = 0.10

# ===== KILL-SWITCH CJK (identico a ambos motores certificados) =====
CJK_RE = re.compile(r'[\u2E80-\u2EFF\u3000-\u303F\u3040-\u30FF\u3100-\u312F'
                    r'\u31C0-\u31EF\u3200-\u33FF\u3400-\u4DBF\u4E00-\u9FFF'
                    r'\uF900-\uFAFF\uFE30-\uFE4F]')


def limpiar_cjk(s: str) -> str:
    s = unicodedata.normalize('NFKC', s)   # rescata dígitos/letras fullwidth (６→6) ANTES de filtrar
    s = CJK_RE.sub('', s)
    return re.sub(r'\s{2,}', ' ', s).strip()


# ===== Resolucion de device (GPU si hay, si no CPU-fallback) =====
def _resolve_device() -> str:
    want = os.getenv("DUO_DEVICE", "auto").lower()
    if want in ("gpu", "cpu"):
        return want
    try:
        import paddle
        if paddle.device.is_compiled_with_cuda() and paddle.device.cuda.device_count() > 0:
            return "gpu"
    except Exception as e:  # noqa: BLE001
        logger.warning(f"[duo-gpu] deteccion de GPU fallo ({e}); uso CPU-fallback")
    return "cpu"


DEVICE = _resolve_device()

from paddleocr import PaddleOCR

# ===== Carga de AMBOS pipelines EN PROCESO — configs IDENTICAS a las certificadas =====
# Los kwargs de calidad (limit_type/side_len/thresh/box_thresh/unclip_ratio + los flags
# de orientacion) son BYTE-a-BYTE los del v6 medium y v6small. Unica diferencia: device.
_common_kwargs = dict(
    cpu_threads=int(_OCR_THREADS),
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

logger.info(f"Loading PP-OCRv6 pipelines EN PROCESO (device={DEVICE})...")
medium_engine = PaddleOCR(
    text_detection_model_name="PP-OCRv6_medium_det",
    text_recognition_model_name="PP-OCRv6_medium_rec",
    **_common_kwargs,
)
small_engine = PaddleOCR(
    text_detection_model_name="PP-OCRv6_small_det",
    text_recognition_model_name="PP-OCRv6_small_rec",
    **_common_kwargs,
)
logger.info(f"PP-OCRv6 medium + small cargados (device={DEVICE}, OCR_THREADS={_OCR_THREADS})")

# Un lock POR motor: predict() no es re-entrante por instancia. Locks separados =>
# small y medium pueden correr concurrentes (distinto lock); dos requests simultaneas
# se serializan sobre el MISMO motor (mismo lock). Preserva correctitud sin matar el
# paralelismo small||medium.
_medium_lock = threading.Lock()
_small_lock = threading.Lock()

app = FastAPI(title="PaddleOCR DUO GPU", version="gpu-1")


class OCRRequest(BaseModel):
    image_base64: str
    imagen_nombre: Optional[str] = None


class OCRResponse(BaseModel):
    success: bool
    text: str
    blocks: list
    total_lines: int
    processing_time_ms: int
    error: Optional[str] = None
    alternates: list = []
    engines_used: list = []
    fusion_stats: Optional[dict] = None


@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "service": "paddleocr-duo-gpu",
        "version": "gpu-1",
        "device": DEVICE,
        "engines": ["small", "medium"],
        "shed_inflight": DUO_SHED_INFLIGHT,
        "max_dimension": MAX_IMAGE_DIMENSION,
    }


# ===================== PRE-PASO DE MOTOR (identico a los motores certificados) ==========
def enhance_for_ocr(img_array: np.ndarray) -> np.ndarray:
    """Enhance image for better OCR: sharpen + adaptive contrast (identico a v6/v6small)."""
    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()
    if sharpness < 500:
        gaussian = cv2.GaussianBlur(img_array, (0, 0), 2.0)
        sharpened = cv2.addWeighted(img_array, 1.5, gaussian, -0.5, 0)
        lab = cv2.cvtColor(sharpened, cv2.COLOR_RGB2LAB)
        l_channel = lab[:, :, 0]
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        lab[:, :, 0] = clahe.apply(l_channel)
        enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        logger.info(f"Enhanced image (sharpness={sharpness:.0f} < 500)")
        return enhanced
    logger.info(f"Image sharp enough (sharpness={sharpness:.0f}), no enhancement needed")
    return img_array


def engine_preprocess(image_bytes: bytes) -> np.ndarray:
    """Decodifica bytes -> RGB, techo 4096 lado largo, enhance. Identico a v6/v6small."""
    image = Image.open(io.BytesIO(image_bytes))
    if image.mode != "RGB":
        image = image.convert("RGB")
    w, h = image.size
    if max(w, h) > MAX_IMAGE_DIMENSION:
        scale = MAX_IMAGE_DIMENSION / max(w, h)
        new_w = int(w * scale)
        new_h = int(h * scale)
        image = image.resize((new_w, new_h), Image.LANCZOS)
        logger.info(f"Resized image from {w}x{h} to {new_w}x{new_h}")
    img_array = np.array(image)
    img_array = enhance_for_ocr(img_array)
    return img_array


def run_ocr_pass(engine, img_array: np.ndarray):
    """Pipeline de una pasada: predict + armado de blocks/lines (identico a v6)."""
    result = engine.predict(img_array)
    blocks = []
    lines = []
    if result and len(result) > 0:
        res = result[0]
        rec_texts = res.get("rec_texts", [])
        rec_scores = res.get("rec_scores", [])
        rec_polys = res.get("rec_polys", [])
        for i in range(len(rec_texts)):
            text = limpiar_cjk(rec_texts[i])
            if not text:
                continue  # block 100% CJK: se descarta
            confidence = float(rec_scores[i]) if i < len(rec_scores) else 0.0
            box_coords = rec_polys[i] if i < len(rec_polys) else []
            if len(box_coords) == 0:
                continue
            x_coords = [float(p[0]) for p in box_coords]
            y_coords = [float(p[1]) for p in box_coords]
            block = {
                "text": text,
                "confidence": confidence,
                "box": {
                    "x_min": min(x_coords),
                    "y_min": min(y_coords),
                    "x_max": max(x_coords),
                    "y_max": max(y_coords),
                }
            }
            blocks.append(block)
            lines.append(text)
    return blocks, lines


def _y_centroide_promedio(blocks: list, patron) -> Optional[float]:
    ys = [(b["box"]["y_min"] + b["box"]["y_max"]) / 2 for b in blocks if patron.search(b["text"])]
    return sum(ys) / len(ys) if ys else None


# ===================== RUNNERS EN PROCESO (reemplazan el POST HTTP) ====================
def run_small(image_bytes: bytes) -> dict:
    """Motor SMALL en proceso. Sin auto-sanacion 180° (identico a v6small)."""
    start = time.time()
    img_array = engine_preprocess(image_bytes)
    with _small_lock:
        blocks, lines = run_ocr_pass(small_engine, img_array)
    pt = int((time.time() - start) * 1000)
    full_text = "\n".join(lines)
    logger.info(f"[small] {len(blocks)} blocks, {len(full_text)} chars, {pt}ms")
    return {"success": True, "text": full_text, "blocks": blocks,
            "total_lines": len(blocks), "processing_time_ms": pt}


def run_medium(image_bytes: bytes) -> dict:
    """Motor MEDIUM en proceso, CON auto-sanacion 180° v2 (identico a v6 medium)."""
    start = time.time()
    img_array = engine_preprocess(image_bytes)
    with _medium_lock:
        blocks, lines = run_ocr_pass(medium_engine, img_array)
        # ===== AUTO-SANACION 180°: señal geometrica (principal) O confianza (respaldo) =====
        total = len(blocks)
        senales = []
        if total > 0:
            c_hdr = _y_centroide_promedio(blocks, ROT_HDR_RE)
            c_tot = _y_centroide_promedio(blocks, ROT_TOT_RE)
            if c_hdr is not None and c_tot is not None and c_hdr > c_tot + ROT_GEO_MARGIN * img_array.shape[0]:
                senales.append("geo")
            sospechosos = sum(1 for b in blocks if b["confidence"] < ROT_CONF_THRESHOLD)
            if sospechosos >= ROT_MIN_SOSPECHOSOS and (sospechosos / total) >= ROT_MIN_RATIO:
                senales.append("conf")
        if senales:
            sum_normal = sum(b["confidence"] for b in blocks)
            blocks_rot, lines_rot = run_ocr_pass(medium_engine, cv2.rotate(img_array, cv2.ROTATE_180))
            sum_rot = sum(b["confidence"] for b in blocks_rot)
            elegido = "rotado" if sum_rot > sum_normal else "normal"
            if elegido == "rotado":
                blocks, lines = blocks_rot, lines_rot
            logger.info(f"🔄 [180] señal={'+'.join(senales)} sum_normal={sum_normal:.1f} sum_rot={sum_rot:.1f} → {elegido}")
    pt = int((time.time() - start) * 1000)
    full_text = "\n".join(lines)
    logger.info(f"[medium] {len(blocks)} blocks, {len(full_text)} chars, {pt}ms")
    return {"success": True, "text": full_text, "blocks": blocks,
            "total_lines": len(blocks), "processing_time_ms": pt}


# ===================== PRE-PASO DEL DUO (exif/resize) — identico al duo v2.2 ===========
def preprocess_image(image_base64: str):
    """Pre-paso ANTES del dispatch a AMBOS motores: exif_transpose + techo de resolucion
    del duo. Sin transformaciones devuelve el base64 original intacto. Identico al duo v2.2.
    Devuelve (base64, exif_transposed, resized, orientation)."""
    try:
        raw = base64.b64decode(image_base64)
        img = Image.open(io.BytesIO(raw))
        orientation = img.getexif().get(274, 1)  # 274 = tag EXIF Orientation
        w0, h0 = img.size
        needs_exif = orientation != 1
        needs_resize = max(w0, h0) > RESIZE_MAX_DIM and (w0 * h0) > RESIZE_MIN_MP * 1_000_000
        if not needs_exif and not needs_resize:
            return image_base64, False, False, orientation
        if needs_exif:
            img = ImageOps.exif_transpose(img)
            logger.info(f"🧭 [duo] exif_transpose aplicado (orientation={orientation})")
        quality = 95
        resized = False
        w_pre, h_pre = img.size
        if needs_resize:
            if img.mode not in ("RGB", "L"):
                img = img.convert("RGB")
            w_pre, h_pre = img.size
            img.thumbnail((RESIZE_MAX_DIM, RESIZE_MAX_DIM), Image.LANCZOS)
            quality = 92
            resized = True
        if img.mode not in ("RGB", "L"):
            img = img.convert("RGB")
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=quality)
        out = buf.getvalue()
        if resized:
            logger.info(
                f"📏 [duo] resize {w_pre}x{h_pre}→{img.size[0]}x{img.size[1]} "
                f"({len(raw) / 1_048_576:.1f} MB→{len(out) / 1_048_576:.1f} MB)"
            )
        return base64.b64encode(out).decode(), needs_exif, resized, orientation
    except Exception as e:
        logger.warning(f"[duo] preprocess_image fallo ({e}); se despachan bytes originales")
        return image_base64, False, False, None


# ===================== FUSION (identica al duo v2.2) ===================================
def iou(a: dict, b: dict) -> float:
    ix_min = max(a["x_min"], b["x_min"])
    iy_min = max(a["y_min"], b["y_min"])
    ix_max = min(a["x_max"], b["x_max"])
    iy_max = min(a["y_max"], b["y_max"])
    iw = max(0.0, ix_max - ix_min)
    ih = max(0.0, iy_max - iy_min)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = (a["x_max"] - a["x_min"]) * (a["y_max"] - a["y_min"])
    area_b = (b["x_max"] - b["x_min"]) * (b["y_max"] - b["y_min"])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def digit_seq(text: str) -> str:
    return "".join(re.findall(r"\d+", text))


def fuse_blocks(small_blocks: list, medium_blocks: list):
    """Union con dedup: mismo bloque si IoU>0.45, gana el de mayor confidence.
    Preserva divergencias numericas como alternates. Identico al duo v2.2."""
    matched_small = set()
    fused = []
    alternates = []
    solo_medium = 0
    for mb in medium_blocks:
        best_i, best_iou = -1, 0.0
        for i, sb in enumerate(small_blocks):
            if i in matched_small:
                continue
            v = iou(mb["box"], sb["box"])
            if v > best_iou:
                best_i, best_iou = i, v
        if best_iou > IOU_THRESHOLD:
            matched_small.add(best_i)
            sb = small_blocks[best_i]
            medium_wins = mb.get("confidence", 0) >= sb.get("confidence", 0)
            winner, loser = (mb, sb) if medium_wins else (sb, mb)
            fused.append(winner)
            dw, dl = digit_seq(winner["text"]), digit_seq(loser["text"])
            if dw != dl and (len(dw) >= MIN_DIGITS_DIVERGENCE or len(dl) >= MIN_DIGITS_DIVERGENCE):
                alternates.append({
                    "primary": winner["text"],
                    "alt": loser["text"],
                    "engine_primary": "medium" if medium_wins else "small",
                    "conf_primary": winner.get("confidence", 0),
                    "conf_alt": loser.get("confidence", 0),
                })
        else:
            fused.append(mb)
            solo_medium += 1
    solo_small = len(small_blocks) - len(matched_small)
    for i, sb in enumerate(small_blocks):
        if i not in matched_small:
            fused.append(sb)
    return fused, solo_small, solo_medium, alternates


def reorder_and_text(blocks: list):
    """Agrupa en bandas de linea por y; dentro de banda ordena por x. Identico al duo v2.2."""
    if not blocks:
        return [], ""
    heights = [b["box"]["y_max"] - b["box"]["y_min"] for b in blocks]
    tol = statistics.median(heights) or 1.0
    by_y = sorted(blocks, key=lambda b: (b["box"]["y_min"] + b["box"]["y_max"]) / 2)
    bands = []
    for b in by_y:
        yc = (b["box"]["y_min"] + b["box"]["y_max"]) / 2
        if bands and abs(yc - bands[-1]["yref"]) <= tol:
            band = bands[-1]
            band["items"].append(b)
            band["yref"] = band["yref"] + (yc - band["yref"]) / len(band["items"])
        else:
            bands.append({"yref": yc, "items": [b]})
    ordered = []
    lines = []
    for band in bands:
        row = sorted(band["items"], key=lambda b: b["box"]["x_min"])
        ordered.extend(row)
        lines.append(" ".join(b["text"] for b in row))
    return ordered, "\n".join(lines)


def alternates_section(alternates: list) -> str:
    if not alternates:
        return ""
    lines = [f"principal: {a['primary']} | alterna: {a['alt']}" for a in alternates]
    return "\n=== LECTURAS ALTERNAS (motores en desacuerdo) ===\n" + "\n".join(lines)


# ===================== ENDPOINT /ocr (misma logica de shed que el duo v2.2) ============
@app.post("/ocr", response_model=OCRResponse)
async def process_ocr(request: OCRRequest):
    global in_flight
    in_flight += 1
    inflight_now = in_flight
    try:
        return await _process_ocr(request, inflight_now)
    finally:
        in_flight -= 1


async def _process_ocr(request: OCRRequest, inflight_now: int) -> "OCRResponse":
    start = time.time()
    nombre = request.imagen_nombre or "-"
    dispatch_b64, exif_transposed, resized, _orientation = preprocess_image(request.image_base64)
    dispatch_bytes = base64.b64decode(dispatch_b64)
    elapsed = lambda: int((time.time() - start) * 1000)
    exif_str = "yes" if exif_transposed else "no"
    resized_str = "yes" if resized else "no"
    load_shed = inflight_now > DUO_SHED_INFLIGHT

    if load_shed:
        # TORMENTA: demasiadas requests en vuelo en este worker. Solo small (motor barato).
        logger.warning(f"🛡️ [duo] shed: in_flight={inflight_now} → small-only")
        try:
            r_small = await asyncio.to_thread(run_small, dispatch_bytes)
        except BaseException as e:  # noqa: BLE001
            err = f"small (shed): {e}"
            logger.error(f"[duo] shed y small fallo ({nombre}): {err}")
            return OCRResponse(success=False, text="", blocks=[], total_lines=0,
                               processing_time_ms=elapsed(), error=err,
                               alternates=[], engines_used=[],
                               fusion_stats={"small": 0, "medium": 0, "fusion": 0,
                                             "solo_small": 0, "solo_medium": 0,
                                             "alternates": 0, "exif_transposed": exif_transposed,
                                             "resized": resized, "load_shed": True})
        n = len(r_small.get("blocks", []))
        pt = elapsed()
        logger.info(
            f"🛡️ [duo] shed small-only small={n} | {len(r_small.get('text', ''))} chars, {pt}ms "
            f"({nombre}) | in_flight={inflight_now} | exif={exif_str} | resized={resized_str}"
        )
        return OCRResponse(success=True, text=r_small.get("text", ""), blocks=r_small.get("blocks", []),
                           total_lines=r_small.get("total_lines", n),
                           processing_time_ms=pt, error=None,
                           alternates=[], engines_used=["small"],
                           fusion_stats={"small": n, "medium": 0, "fusion": n,
                                         "solo_small": 0, "solo_medium": 0,
                                         "alternates": 0, "exif_transposed": exif_transposed,
                                         "resized": resized, "load_shed": True,
                                         "small_ms": r_small.get("processing_time_ms", 0),
                                         "medium_ms": 0})

    # Camino normal: AMBOS motores en proceso, en paralelo (threads). Si la GPU serializa,
    # el wall se aproxima a la suma y parallel=False lo delata; si solapan, parallel=True.
    ocr_start = time.time()
    r_small, r_medium = await asyncio.gather(
        asyncio.to_thread(run_small, dispatch_bytes),
        asyncio.to_thread(run_medium, dispatch_bytes),
        return_exceptions=True,
    )
    wall_ocr_ms = int((time.time() - ocr_start) * 1000)
    small_ok = not isinstance(r_small, BaseException)
    medium_ok = not isinstance(r_medium, BaseException)
    small_ms = r_small.get("processing_time_ms", 0) if small_ok else 0
    medium_ms = r_medium.get("processing_time_ms", 0) if medium_ok else 0
    # parallel=True si el wall combinado es sensiblemente menor que la suma de motores
    # (solaparon en el device); False si la GPU los serializo (wall ≈ suma).
    parallel = bool(small_ok and medium_ok and wall_ocr_ms < (small_ms + medium_ms) * 0.9)

    if not small_ok and not medium_ok:
        err = f"small: {r_small} | medium: {r_medium}"
        logger.error(f"[duo] ambos motores fallaron ({nombre}): {err}")
        return OCRResponse(success=False, text="", blocks=[], total_lines=0,
                           processing_time_ms=elapsed(), error=err,
                           alternates=[], engines_used=[],
                           fusion_stats={"small": 0, "medium": 0, "fusion": 0,
                                         "solo_small": 0, "solo_medium": 0,
                                         "alternates": 0, "exif_transposed": exif_transposed,
                                         "resized": resized, "load_shed": False,
                                         "small_ms": small_ms, "medium_ms": medium_ms,
                                         "wall_ocr_ms": wall_ocr_ms, "parallel": parallel})

    if small_ok != medium_ok:
        # Resiliencia: responde tal cual el motor vivo (solo se ajusta el wall del duo)
        alive, name = (r_small, "small") if small_ok else (r_medium, "medium")
        dead = r_medium if small_ok else r_small
        n = len(alive.get("blocks", []))
        logger.warning(f"[duo] motor caido ({nombre}): {'medium' if small_ok else 'small'} -> {dead}; respondo {name} tal cual")
        return OCRResponse(success=True, text=alive.get("text", ""), blocks=alive.get("blocks", []),
                           total_lines=alive.get("total_lines", n),
                           processing_time_ms=elapsed(), error=None,
                           alternates=[], engines_used=[name],
                           fusion_stats={"small": n if small_ok else 0,
                                         "medium": n if medium_ok else 0,
                                         "fusion": n, "solo_small": 0, "solo_medium": 0,
                                         "alternates": 0, "exif_transposed": exif_transposed,
                                         "resized": resized, "load_shed": False,
                                         "small_ms": small_ms, "medium_ms": medium_ms,
                                         "wall_ocr_ms": wall_ocr_ms, "parallel": parallel})

    fused, solo_small, solo_medium, alternates = fuse_blocks(r_small["blocks"], r_medium["blocks"])
    ordered, text = reorder_and_text(fused)
    text += alternates_section(alternates)
    stats = {"small": len(r_small["blocks"]), "medium": len(r_medium["blocks"]),
             "fusion": len(ordered), "solo_small": solo_small, "solo_medium": solo_medium,
             "alternates": len(alternates), "exif_transposed": exif_transposed,
             "resized": resized, "load_shed": False,
             "small_ms": small_ms, "medium_ms": medium_ms,
             "wall_ocr_ms": wall_ocr_ms, "parallel": parallel}
    pt = elapsed()
    logger.info(
        f"🤝 [duo] small={stats['small']} medium={stats['medium']} "
        f"→ fusion={stats['fusion']} (+{solo_small} solo-small, +{solo_medium} solo-medium) "
        f"| {len(text)} chars, {pt}ms ({nombre}) | alt={len(alternates)} | exif={exif_str} | "
        f"resized={resized_str} | small={small_ms}ms medium={medium_ms}ms wall={wall_ocr_ms}ms parallel={parallel}"
    )
    return OCRResponse(success=True, text=text, blocks=ordered, total_lines=len(ordered),
                       processing_time_ms=pt, error=None,
                       alternates=alternates, engines_used=["small", "medium"],
                       fusion_stats=stats)
