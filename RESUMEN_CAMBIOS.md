# DUO-GPU — Imagen/entrega para Runpod (vía GitHub, CERO docker build)

Un solo servicio, **ambos motores PaddleOCR en GPU y en proceso**. Deriva del
**duo v2.2+shed** (agregador que hacía POST HTTP a `:8012` medium / `:8013` small):
misma API, mismo esquema de respuesta, misma fusión IoU + alternas + prepaso
exif/resize, mismo load-shed adaptativo. **Diferencia:** en vez de HTTP a dos
backends, carga los dos pipelines `PP-OCRv6_medium` y `PP-OCRv6_small` **dentro del
proceso** con `device=gpu` (auto-fallback a CPU si no hay GPU), y corre ambos por
request en paralelo (threads; si la GPU serializa, degrada a secuencial de facto —
se **mide y reporta** en `fusion_stats`).

> **NO se tocó ningún servicio en producción** (`:8011` v5, `:8012` v6 medium,
> `:8013` v6small, `:8014` duo v2.2). Esto vive en carpeta y repo nuevos.

---

## Entrega: 3 archivos en el repo GitHub PÚBLICO `paddleocr-duo-gpu`

| Archivo | Rol |
|---|---|
| `app.py` | El agregador DUO-GPU (ambos pipelines en proceso, misma API `/ocr`). |
| `requirements.txt` | `paddleocr==3.7.0` (versión de producción) + FastAPI/uvicorn/Pillow/opencv/pydantic. **`paddlepaddle-gpu` NO se instala: lo aporta la imagen base del pod.** |
| `bootstrap.sh` | Self-install idempotente que corre dentro del pod: descarga código, instala `libgl1`+`libglib2.0-0` (cv2 necesita `libGL.so.1`), `pip install`, **calienta modelos** (dummy medium+small → descarga y cachea pesos), `touch /root/.duo_ready`, lanza `uvicorn ... :8014 --workers ${UVICORN_WORKERS:-5}`. |

### RAW URLs (rellenar con el usuario/branch reales)

```
app.py            → https://raw.githubusercontent.com/Zarushy09/paddleocr-duo-gpu/main/app.py
requirements.txt  → https://raw.githubusercontent.com/Zarushy09/paddleocr-duo-gpu/main/requirements.txt
bootstrap.sh      → https://raw.githubusercontent.com/Zarushy09/paddleocr-duo-gpu/main/bootstrap.sh
```

---

## Cómo desplegar en Runpod

1. **Crear pod** con la imagen oficial (Docker Image / Template):

   ```
   paddlepaddle/paddle:3.2.2-gpu-cuda12.6-cudnn9.5
   ```

2. **Exponer el puerto** `8014` (HTTP).

3. **START COMMAND** del pod (descarga y ejecuta el bootstrap, con fallback curl→wget):

   ```bash
   bash -c "curl -sL https://raw.githubusercontent.com/Zarushy09/paddleocr-duo-gpu/main/bootstrap.sh | bash || wget -qO- https://raw.githubusercontent.com/Zarushy09/paddleocr-duo-gpu/main/bootstrap.sh | bash"
   ```

   El primer arranque instala deps + descarga/calienta modelos (varios minutos) y
   crea `/root/.duo_ready`. Los siguientes arranques saltan esa fase y relanzan
   uvicorn en segundos. Al final de los logs verás: **`🟢 DUO-GPU listo en :8014`**.

---

## Prueba (curl)

`GET /health` (device + engines):

```bash
curl -s http://<POD_HOST>:8014/health
# {"status":"healthy","service":"paddleocr-duo-gpu","version":"gpu-1",
#  "device":"gpu","engines":["small","medium"],"workers":5,
#  "shed_inflight":64,"max_dimension":4096}
```

`POST /ocr` (base64 de una imagen; misma API que el duo actual):

```bash
IMG_B64=$(base64 -w0 factura.jpg)
curl -s -X POST http://<POD_HOST>:8014/ocr \
  -H 'Content-Type: application/json' \
  -d "{\"image_base64\":\"$IMG_B64\",\"imagen_nombre\":\"factura.jpg\"}" \
  | python3 -m json.tool
```

Respuesta (mismo esquema que el duo v2.2, con timings extra en `fusion_stats`):

```jsonc
{
  "success": true,
  "text": "…",
  "blocks": [ /* {text, confidence, box:{x_min,y_min,x_max,y_max}} */ ],
  "total_lines": 64,
  "processing_time_ms": 1234,
  "error": null,
  "alternates": [ /* {primary, alt, engine_primary, conf_primary, conf_alt} */ ],
  "engines_used": ["small", "medium"],
  "fusion_stats": {
    "small": 60, "medium": 62, "fusion": 64,
    "solo_small": 1, "solo_medium": 3, "alternates": 3,
    "exif_transposed": false, "resized": false, "load_shed": false,
    "small_ms": 300, "medium_ms": 900, "wall_ocr_ms": 950, "parallel": true
  }
}
```

- `parallel: true` → small y medium **solaparon** en la GPU (wall < suma·0.9).
- `parallel: false` → la GPU los **serializó** (wall ≈ suma). Ambos casos son
  correctos; el campo es solo telemetría de cómo se comportó el device.

---

## Equivalencia con lo certificado

- **Configs de motor idénticas** a v6 medium y v6small: `use_textline_orientation`,
  `use_doc_orientation_classify=True`, `use_doc_unwarping=False`,
  `text_det_limit_type="min"`, `text_det_limit_side_len=64`, `text_det_thresh=0.2`,
  `text_det_box_thresh=0.4`, `text_det_unclip_ratio=2.0`. Único cambio: `device`.
- **Medium** conserva su **auto-sanación 180°** (señal geométrica ó confianza);
  **small** no la tiene (igual que en producción).
- **Pre-paso de motor** (RGB, techo 4096px, `enhance_for_ocr` sharpen+CLAHE) y
  **kill-switch CJK** idénticos byte a byte.
- **Agregador**: `preprocess_image` (exif_transpose + techo `DUO_RESIZE_MAX_DIM`, 2000px
  por defecto, sólo si además la imagen supera 4 MP),
  fusión dedup IoU>0.45 con alternas de divergencia numérica (≥6 dígitos),
  reordenado por bandas de línea, y `DUO_SHED_INFLIGHT` (default **64** =
  "adaptativo dormido": bajo carga normal nunca descarta, ambos motores siempre).

## Variables de entorno (opcionales)

| Env | Default | Efecto |
|---|---|---|
| `DUO_DEVICE` | `auto` | `gpu` / `cpu` fuerzan device; `auto` detecta. |
| `DUO_SHED_INFLIGHT` | `64` | Umbral de load-shed **por worker** (in_flight > umbral → sólo small). **`0` fuerza modo un-motor (small-only) permanente**: `in_flight` siempre vale ≥1 dentro de `/ocr`, así que el shed dispara en todas las requests. Ver §"Modo un-motor". |
| `DUO_RESIZE_MAX_DIM` | `2000` | Techo de resolución **del agregador** (`preprocess_image`), en px de lado largo. Sólo actúa si además la imagen supera 4 MP. **No es el techo del motor**: ése es `MAX_IMAGE_DIMENSION = 4096`, fijo. Ver §"Modo un-motor" para el valor de equivalencia. |
| `UVICORN_WORKERS` | `5` | Procesos uvicorn que lanza `bootstrap.sh`. **Cada worker carga los 2 pipelines** ⇒ 5 workers = 10 pipelines en VRAM. `app.py` sólo la lee para reportarla en `/health`. |
| `OCR_THREADS` | `4` | Hilos CPU (cv2 + paddle cpu_threads; sólo relevante en CPU-fallback). |
| `PORT` | `8014` | Puerto de escucha. |
| `RAW_BASE` | `…/paddleocr-duo-gpu/main` | Base RAW desde la que `bootstrap.sh` descarga `app.py` y `requirements.txt`. Apuntarla a un **commit inmutable** en vez de a `main` para fijar la versión que corre el pod. |

---

## Modo un-motor (small-only) y equivalencia con el small standalone

Configuración usada en el cutover de producción, para que el pod GPU sea equivalente al
motor `PP-OCRv6_small` standalone que corría en Oracle (`:8013`):

```
DUO_SHED_INFLIGHT=0
DUO_RESIZE_MAX_DIM=100000
```

**`DUO_SHED_INFLIGHT=0`** → todas las requests entran por la rama de shed: sólo corre el
motor small, la respuesta trae `engines_used:["small"]`, `alternates:[]` y
`fusion_stats.load_shed:true`, y el campo `text` se devuelve **tal cual lo produce el motor**
(`"\n".join(lines)`, orden crudo del motor) — es decir, sin la reordenación por bandas de la
fusión y sin la sección de lecturas alternas, exactamente igual que el small standalone.
Ojo: **los dos pipelines se siguen cargando en VRAM**; esta palanca ahorra cómputo por
request, no memoria.

**`DUO_RESIZE_MAX_DIM=100000`** → desactiva de hecho el techo del agregador. El motivo es de
equivalencia byte a byte, y por eso no basta con igualarlo al techo del motor (4096):

- Con el default **2000**, una imagen de 3000×2000 (6 MP) llega al motor reescalada a 2000 px,
  mientras el small standalone la procesa a 3000 px completos. Imágenes distintas ⇒ OCR distinto.
- Con **4096**, esa clase se arregla, pero las imágenes de más de 4096 px **siguen reescalándose
  por una ruta distinta** a la del standalone: el agregador usa `img.thumbnail(...)` + re-encode
  JPEG q92, y el standalone usa `int(w*scale)` + `LANCZOS` sobre los bytes originales. Para
  5000×4000 eso da 4096×**3277** frente a 4096×**3276**, más una recompresión JPEG. La
  divergencia se reduce, pero no desaparece.
- Con **100000** la condición `max(w,h) > DUO_RESIZE_MAX_DIM` no se cumple nunca en la práctica,
  `preprocess_image` devuelve el base64 **intacto** (cero re-encode) y todo el trabajo del techo
  lo hace `engine_preprocess`, que es **AST-idéntica** al `preprocess_image` del servicio v6small
  certificado (mismo `convert("RGB")`, mismo `MAX_IMAGE_DIMENSION=4096`, mismo `int(w*scale)`,
  mismo `LANCZOS`, mismo `enhance_for_ocr`). La clase entera de divergencia desaparece en vez
  de reducirse.

### Divergencia deliberada que se conserva: `exif_transpose`

El agregador aplica `ImageOps.exif_transpose` (y el re-encode JPEG q95 que conlleva) a las
imágenes con `EXIF orientation != 1`, **antes** de la decisión de shed. El small standalone de
Oracle **no lo hace**. Esa diferencia **se mantiene a propósito**: es una mejora, no un bug —
Oracle procesaba esas imágenes giradas. Quien evalúe el gate de equivalencia debe esperar que
las imágenes con EXIF rotado difieran del baseline de Oracle, y que la lectura del pod sea la
correcta de las dos.
