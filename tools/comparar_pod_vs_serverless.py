#!/usr/bin/env python3
"""
comparar_pod_vs_serverless.py — Puerta de byte-identidad pod ↔ serverless.

Manda cada imagen al pod de producción (/ocr) y al endpoint serverless (/runsync)
y compara las dos respuestas del DUO campo por campo, sin los campos de tiempo.
Mide también el arranque en frío (delayTime) y el tiempo de ejecución.

Correr DESDE el Contabo (como los closedloop), solo librería estándar:

  export RUNPOD_API_KEY=...          # clave restringida al endpoint
  python3 comparar_pod_vs_serverless.py \
      --endpoint <ENDPOINT_ID> \
      --imagenes /root/bateria_dorada \
      --salida /root/serverless_vs_pod_$(date +%Y%m%d_%H%M)

Puerta (doctrina del proyecto): `text` idéntico en el 100 % de las imágenes y
`blocks` idénticos. Una sola diferencia de texto = no se promueve.
"""

import argparse
import base64
import glob
import json
import os
import statistics
import sys
import time
import urllib.error
import urllib.request

POD_DEFAULT = "https://m6sl8l3btyvtln-8014.proxy.runpod.net/ocr"
# El proxy de Runpod (Cloudflare) bloquea el User-Agent de Python.
UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/129.0 Safari/537.36"
TIEMPOS_FS = {"small_ms", "medium_ms", "wall_ocr_ms", "parallel"}
EXT = (".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff")


def post(url, body, headers, timeout):
    req = urllib.request.Request(url, data=json.dumps(body).encode(), method="POST",
                                 headers={"Content-Type": "application/json", "User-Agent": UA, **headers})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


def get(url, headers, timeout):
    req = urllib.request.Request(url, headers={"User-Agent": UA, **headers})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


def llamar_serverless(endpoint, key, inp, timeout_s):
    base = f"https://api.runpod.ai/v2/{endpoint}"
    h = {"Authorization": f"Bearer {key}"}
    t0 = time.time()
    r = post(f"{base}/runsync?wait={int(timeout_s * 1000)}",
             {"input": inp, "policy": {"executionTimeout": int(timeout_s * 1000), "ttl": int(timeout_s * 3000)}},
             h, timeout_s + 30)
    # /runsync puede devolver un estado no final: seguir con /status hasta terminar.
    while r.get("status") in ("IN_QUEUE", "IN_PROGRESS") and time.time() - t0 < timeout_s * 3:
        time.sleep(1)
        r = get(f"{base}/status/{r['id']}", h, 30)
    r["_wall_ms"] = int((time.time() - t0) * 1000)
    return r


def normalizar(duo):
    d = dict(duo or {})
    d.pop("processing_time_ms", None)
    fs = dict(d.get("fusion_stats") or {})
    for k in TIEMPOS_FS:
        fs.pop(k, None)
    d["fusion_stats"] = fs
    return d


def pct(xs, p):
    if not xs:
        return None
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(round(p / 100 * (len(xs) - 1))))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--endpoint", required=True, help="ID del endpoint serverless")
    ap.add_argument("--imagenes", required=True, help="carpeta o patrón glob de imágenes")
    ap.add_argument("--pod", default=POD_DEFAULT, help="URL /ocr del pod de producción")
    ap.add_argument("--salida", default=f"serverless_vs_pod_{time.strftime('%Y%m%d_%H%M')}")
    ap.add_argument("--timeout", type=float, default=120)
    ap.add_argument("--sin-warmup", action="store_true", help="no mandar warmup al inicio")
    a = ap.parse_args()

    key = os.environ.get("RUNPOD_API_KEY")
    if not key:
        sys.exit("Falta RUNPOD_API_KEY en el entorno.")
    pat = os.path.join(a.imagenes, "*") if os.path.isdir(a.imagenes) else a.imagenes
    files = sorted(f for f in glob.glob(pat) if f.lower().endswith(EXT))
    if not files:
        sys.exit(f"No hay imágenes en {a.imagenes}")
    os.makedirs(a.salida, exist_ok=True)

    print(f"Imágenes: {len(files)} | pod: {a.pod} | endpoint: {a.endpoint}")
    if not a.sin_warmup:
        r = llamar_serverless(a.endpoint, key, {"warmup": True}, a.timeout)
        print(f"[warmup] status={r.get('status')} delayTime={r.get('delayTime')} ms "
              f"executionTime={r.get('executionTime')} ms arranque_duo={((r.get('output') or {}).get('rp') or {}).get('arranque_s')} s")
        r = llamar_serverless(a.endpoint, key, {"diagnostico": True}, a.timeout)
        with open(os.path.join(a.salida, "diagnostico_serverless.json"), "w", encoding="utf-8") as f:
            json.dump(r.get("output"), f, ensure_ascii=False, indent=2)
        print(f"[diagnostico] guardado en {a.salida}/diagnostico_serverless.json")

    filas, txt_ok, blk_ok, todo_ok = [], 0, 0, 0
    delays, execs, pod_ms = [], [], []
    for i, fn in enumerate(files, 1):
        nombre = os.path.basename(fn)
        with open(fn, "rb") as f:
            b64 = base64.b64encode(f.read()).decode()
        inp = {"image_base64": b64, "imagen_nombre": nombre}
        fila = {"imagen": nombre}
        try:
            t0 = time.time()
            pod = post(a.pod, inp, {}, a.timeout)
            fila["pod_wall_ms"] = int((time.time() - t0) * 1000)
            pod_ms.append(pod.get("processing_time_ms", 0))
        except Exception as e:  # noqa: BLE001
            pod = None
            fila["pod_error"] = str(e)
        try:
            sv = llamar_serverless(a.endpoint, key, inp, a.timeout)
            fila.update(status=sv.get("status"), delayTime=sv.get("delayTime"),
                        executionTime=sv.get("executionTime"), sv_wall_ms=sv.get("_wall_ms"))
            if sv.get("delayTime") is not None:
                delays.append(sv["delayTime"])
            if sv.get("executionTime") is not None:
                execs.append(sv["executionTime"])
            duo = (sv.get("output") or {}).get("duo")
            if duo is None:
                fila["sv_error"] = sv.get("error") or json.dumps(sv.get("output"))[:300]
        except Exception as e:  # noqa: BLE001
            duo = None
            fila["sv_error"] = str(e)

        if pod is not None and duo is not None:
            a_, b_ = normalizar(pod), normalizar(duo)
            fila["text_igual"] = a_.get("text") == b_.get("text")
            fila["blocks_iguales"] = a_.get("blocks") == b_.get("blocks")
            fila["todo_igual"] = a_ == b_
            txt_ok += fila["text_igual"]
            blk_ok += fila["blocks_iguales"]
            todo_ok += fila["todo_igual"]
            if not fila["todo_igual"]:
                with open(os.path.join(a.salida, f"diff_{i:03d}_{nombre}.json"), "w", encoding="utf-8") as f:
                    json.dump({"pod": a_, "serverless": b_}, f, ensure_ascii=False, indent=1)
        filas.append(fila)
        marca = "✅" if fila.get("todo_igual") else ("🟡" if fila.get("text_igual") else "❌")
        print(f"{marca} {i:3d}/{len(files)} {nombre[:48]:48} text={fila.get('text_igual')} "
              f"blocks={fila.get('blocks_iguales')} delay={fila.get('delayTime')} exec={fila.get('executionTime')} "
              f"{fila.get('pod_error') or fila.get('sv_error') or ''}")

    n = len(files)
    resumen = {
        "imagenes": n, "text_identico": txt_ok, "blocks_identicos": blk_ok, "todo_identico": todo_ok,
        "delayTime_ms": {"primero": delays[0] if delays else None, "p50": pct(delays, 50), "p95": pct(delays, 95), "max": max(delays) if delays else None},
        "executionTime_ms": {"p50": pct(execs, 50), "p95": pct(execs, 95), "max": max(execs) if execs else None},
        "pod_processing_ms": {"p50": pct(pod_ms, 50), "p95": pct(pod_ms, 95)},
        "puerta_byte_identidad": "PASA" if txt_ok == n and blk_ok == n else "NO PASA",
    }
    with open(os.path.join(a.salida, "resultado.json"), "w", encoding="utf-8") as f:
        json.dump({"resumen": resumen, "filas": filas}, f, ensure_ascii=False, indent=1)
    print("\n" + json.dumps(resumen, ensure_ascii=False, indent=1))
    print(f"\nDetalle y diferencias en {a.salida}/")


if __name__ == "__main__":
    main()
