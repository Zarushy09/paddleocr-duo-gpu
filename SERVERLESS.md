# DUO-GPU en Runpod Serverless (rama `serverless`)

**Objetivo:** mover el OCR del pod fijo (`pregnant_brown_fish`, RTX 4000 Ada, USD 0.29/h las 24 h)
a un endpoint serverless que escala a cero, **sin cambiar ni un carácter de la salida**.

**Regla de oro:** el pod sigue siendo producción hasta que el endpoint pase las puertas del §6.
El cambio de URL en n8n (§7) va en ventana de mantenimiento, con el pod encendido como respaldo.

---

## 1. Qué hay en la rama

| Archivo | Para qué |
|---|---|
| `app.py`, `requirements.txt` | **Sin cambios**: los mismos de `main`, los que corre el pod. |
| `Dockerfile` | Misma base que el pod (`paddlepaddle/paddle:3.2.2-gpu-cuda12.6-cudnn9.5`), mismos paquetes, **pesos dentro de la imagen** (se bajan en el build con `device=cpu`). |
| `constraints-pod.txt` | Versiones fijadas a las del pod. **Confirmar con el `pip freeze` del pod antes del primer build.** |
| `warm_models.py` | Baja los 6 modelos (texto medium/small + orientación) y falla el build si falta alguno. |
| `handler.py` | Worker de Runpod. Arranca el mismo `app.py` (uvicorn, 5 workers) en `127.0.0.1:8014` y le reenvía cada trabajo. Vive en su propio entorno (`/opt/rp`, Python 3.11): el SDK de Runpod exige fastapi ≥0.141 y el OCR usa 0.115. |
| `requirements-handler.txt` | `runpod==1.12.0` y `aiohttp` (solo para el handler). |
| `diag.py` | Huella del entorno (versiones, sha256 de los pesos, GPU, variables del DUO). Corre igual en el pod y en el worker. |
| `tools/comparar_pod_vs_serverless.py` | Puerta de byte-identidad: misma imagen al pod y al endpoint, compara campo por campo y mide el arranque en frío. Corre desde el Contabo. |

### Contrato del endpoint

Petición (`POST https://api.runpod.ai/v2/<ENDPOINT_ID>/runsync`, cabecera `Authorization: Bearer <clave>`):
```json
{ "input": { "image_base64": "<...>", "imagen_nombre": "<...>" },
  "policy": { "executionTimeout": 30000, "ttl": 45000 } }
```
Respuesta:
```json
{ "id": "sync-…", "status": "COMPLETED", "delayTime": 812, "executionTime": 2410,
  "output": { "duo": { …el JSON de /ocr del pod, tal cual… }, "rp": { "worker_ms": 2398, "arranque_s": 31.4 } } }
```
- La salida del DUO va dentro de `output.duo` porque el SDK de Runpod quita la clave `error` del
  primer nivel (y si no es null marca el trabajo como fallido).
- Otros inputs: `{"warmup": true}` (despierta el worker y confirma que está listo) y
  `{"diagnostico": true}` (huella completa del entorno).
- Si el OCR local no responde, el trabajo termina `FAILED` con `error`: para n8n es «no respondió».

---

## 2. Confirmar las versiones del pod (5 minutos, antes del primer build)

En Runpod → Pods → `pregnant_brown_fish` → **Connect → Web Terminal**:
```bash
curl -sL https://raw.githubusercontent.com/Zarushy09/paddleocr-duo-gpu/serverless/diag.py | python - --json
curl -sL https://raw.githubusercontent.com/Zarushy09/paddleocr-duo-gpu/serverless/diag.py | python - --freeze
```
Pegar las dos salidas en el chat. Con el `--freeze` se reescribe `constraints-pod.txt` (sin líneas
`@ file://`); con el `--json` se comparan pesos, GPU y variables (`DUO_RESIZE_MAX_DIM`,
`OCR_THREADS`, `UVICORN_WORKERS`…) contra el worker. **Si el pod tiene alguna de esas variables
definida, el endpoint debe llevar la misma.**

---

## 3. Conectar GitHub a Runpod (una vez)

Runpod → **Settings → Connections → GitHub → Connect** → *Only select repositories* →
`paddleocr-duo-gpu` → **Save**.

---

## 4. Crear el endpoint

Runpod → **Serverless → New Endpoint → Import Git Repository** → `paddleocr-duo-gpu`.

| Campo | Valor | Por qué |
|---|---|---|
| Branch | `serverless` | `main` sigue siendo la del pod. |
| Dockerfile Path | `Dockerfile` | Está en la raíz de la rama. |
| Endpoint Name | `duo-ocr-serverless` | |
| Endpoint Type | **Queue** | Cola con reintentos; `/runsync` y `/run`. |
| GPU | **16 GB** (incluye RTX 4000 Ada) | Después se restringe a RTX 4000 Ada sola (§5). |
| Active workers | **0** | Escala a cero. |
| Max workers | **2** | Para pruebas; la cuenta permite 5 en total. |
| GPUs per worker | 1 | |
| Idle timeout | **60 s** | Un lote amortiza un solo arranque. Se ajusta con lo medido. |
| Execution timeout | **120 s** | Tope por trabajo (n8n manda el suyo de 30 s). |
| FlashBoot | **On** | Acelera los arranques repetidos. |
| Auto-scaling | **Request count = 5** | Un worker atiende 5 imágenes a la vez (5 procesos uvicorn). |
| Advanced → CUDA | **12.6 o superior** (marcar 12.6, 12.7, 12.8, 12.9, 13.0) | La imagen es CUDA 12.6: un host con driver más viejo no la arranca. |
| Environment variables | `UVICORN_WORKERS=5` + las que tenga el pod (§2) | |

**Deploy Endpoint** → pestaña **Builds**: *Pending → Building → Uploading → Testing → Completed*.
Límites de Runpod: el `docker build` debe terminar en 30 min y todo el proceso en 160 min.
En el log del build deben aparecer `[warm] caché completa` y la línea JSON de `diag.py`.

Actualizar el worker más adelante = **crear un Release en GitHub** (un push solo no reconstruye).
Volver atrás = pestaña Builds → ⋯ → **Rollback**.

---

## 5. Dejarlo en la misma GPU que el pod

La categoría de 16 GB mezcla A4000, A4500 y RTX 4000 Ada. La byte-identidad se certifica **por tipo
de GPU**, así que al principio el endpoint usa solo la del pod:
`gpuTypeIds: ["NVIDIA RTX 4000 Ada Generation"]` (no confundir con `…RTX 4000 SFF Ada Generation`).
Se fija con el MCP de Runpod (`update-endpoint`) o por la API REST. Más adelante se pueden sumar
A4000/A4500 repitiendo la puerta del §6 en cada una.

---

## 6. Puertas antes de tocar producción

1. **Diagnóstico igual al pod:** `{"diagnostico": true}` → mismas versiones, mismos sha256 de pesos,
   mismas variables del DUO, misma GPU.
2. **Byte-identidad:** desde el Contabo,
   ```bash
   export RUNPOD_API_KEY=...   # clave restringida al endpoint
   python3 comparar_pod_vs_serverless.py --endpoint <ID> --imagenes <carpeta Batería Dorada> \
       --salida /root/serverless_vs_pod_$(date +%Y%m%d_%H%M)
   ```
   Pasa solo con `text` y `blocks` idénticos en el 100 % de las imágenes. **Una diferencia de texto
   = no se promueve** y se analiza el `diff_*.json`.
3. **Arranque en frío medido:** `delayTime` del primer trabajo tras ≥ idle timeout sin uso.
   Meta del plan de julio: < 30 s. Mínimo aceptable: que quepa en la cascada actual de n8n.
4. **Carga:** `closedloop_duo_gpu.py` apuntado al endpoint con 0 fallos.
5. **Costo proyectado** con `delayTime` + `executionTime` reales (16 GB flex: USD 0.00016/s
   según la página de configuración de Runpod).

---

## 7. Cambio en n8n (después de las puertas, en ventana)

En los dos nodos de OCR (principal y Reintento), del SANDBOX primero:
- URL → `https://api.runpod.ai/v2/<ENDPOINT_ID>/runsync`; credencial *Header Auth*
  `Authorization: Bearer <clave restringida>`.
- Body → `{"input": {"image_base64": …, "imagen_nombre": …}, "policy": {"executionTimeout": 30000, "ttl": 45000}}`.
- IF `¿respondió?` → `$json.status === 'COMPLETED' && $json.output && $json.output.duo && $json.output.duo.success`
  (`IN_QUEUE` / `IN_PROGRESS` cuentan como «no respondió»).
- Un nodo que suba `output.duo` al primer nivel, para que `Parse PaddleOCR` no cambie.
- Pod encendido durante la vigilancia; se apaga cuando la rama GV no se active por arranques en frío.

---

## 8. Rollback

- Antes del §7: no hay nada que revertir; el pod nunca dejó de ser producción.
- Después del §7: volver a publicar la versión anterior de los flujos de n8n (URL del pod).
- Imagen: pestaña Builds → Rollback a un build anterior.
