"""
Passport Photo Processing API  —  Free-Tier Edition (512 MB)
=============================================================
Optimised for Render's free plan:
  • OpenCV Haar cascades  (bundled, ~2 MB)  replaces MediaPipe (~150 MB)
  • rembg  u2netp  model   (~20 MB RAM)     replaces u2net (~350 MB)
  • Aggressive gc + explicit array deletion throughout
  • Max input dimension capped at 1500 px to limit working-set size

REST-only (no HTML served).  CORS configured for a separate frontend.
"""

from __future__ import annotations

import gc
import io
import os
import uuid
import time
import asyncio
import logging
import tempfile
import shutil
import threading
from datetime import datetime, timezone
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np
from PIL import Image
from rembg import remove, new_session
from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded


# ═══════════════════════════════════════════════════════════════════════════════
#  CONFIGURATION  (all tunable via environment variables)
# ═══════════════════════════════════════════════════════════════════════════════

MAX_UPLOAD_MB: int = int(os.getenv("MAX_FILE_SIZE_MB", "10"))
MAX_UPLOAD_BYTES: int = MAX_UPLOAD_MB * 1_048_576
MIN_DIMENSION: int = 200
MAX_DIMENSION: int = int(os.getenv("MAX_INPUT_DIMENSION", "1500"))
ALLOWED_SUFFIXES = frozenset(
    {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff", ".tif"}
)

PASSPORT_W: int = 413   # 35 mm @ 300 DPI
PASSPORT_H: int = 531   # 45 mm @ 300 DPI

# Light-blue background  —  standard for many Asian passport specs
BG_RGB: tuple = (217, 237, 249)

TTL_SECONDS: int = int(os.getenv("RESULT_EXPIRY_MINUTES", "10")) * 60
PORT: int = int(os.getenv("PORT", "8000"))
HOST: str = os.getenv("HOST", "0.0.0.0")

CORS_ORIGINS: List[str] = [
    o.strip()
    for o in os.getenv("CORS_ORIGINS", "*").split(",")
    if o.strip()
]


# ═══════════════════════════════════════════════════════════════════════════════
#  LOGGING
# ═══════════════════════════════════════════════════════════════════════════════

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("passport")


# ═══════════════════════════════════════════════════════════════════════════════
#  EPHEMERAL TEMP DIRECTORY  (removed on shutdown + by janitor)
# ═══════════════════════════════════════════════════════════════════════════════

TMP: Path = Path(tempfile.mkdtemp(prefix="passport_"))


# ═══════════════════════════════════════════════════════════════════════════════
#  LAZY MODEL LOADING  (nothing loaded at import — fast cold-start)
# ═══════════════════════════════════════════════════════════════════════════════

# --- rembg / u2netp (background removal) ---
_rembg_session = None
_rembg_lock = threading.Lock()


def _get_rembg():
    global _rembg_session
    if _rembg_session is None:
        with _rembg_lock:
            if _rembg_session is None:
                log.info("Loading U²-Net Portable model (u2netp — one-time)…")
                t0 = time.monotonic()
                _rembg_session = new_session("u2netp")
                log.info(
                    "u2netp ready (%.1f s).",
                    time.monotonic() - t0,
                )
    return _rembg_session


# --- OpenCV Haar cascades (face + eye detection) ---
_face_cascade = None
_eye_cascade = None
_cascade_lock = threading.Lock()


def _load_cascades():
    global _face_cascade, _eye_cascade
    if _face_cascade is not None:
        return
    with _cascade_lock:
        if _face_cascade is not None:
            return
        face_xml = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        eye_xml = cv2.data.haarcascades + "haarcascade_eye.xml"
        _face_cascade = cv2.CascadeClassifier(face_xml)
        _eye_cascade = cv2.CascadeClassifier(eye_xml)
        if _face_cascade.empty():
            raise RuntimeError(f"Cannot load face cascade: {face_xml}")
        if _eye_cascade.empty():
            raise RuntimeError(f"Cannot load eye cascade: {eye_xml}")
        log.info("Haar cascades loaded (face + eye).")


# ═══════════════════════════════════════════════════════════════════════════════
#  IN-MEMORY TASK STORE  (no database)
# ═══════════════════════════════════════════════════════════════════════════════

class Task:
    __slots__ = (
        "id", "inp", "out",
        "status", "progress", "error",
        "t_created", "t_finished",
    )

    def __init__(self, tid: str, inp: str):
        self.id: str = tid
        self.inp: str = inp
        self.out: Optional[str] = None
        self.status: str = "queued"
        self.progress: int = 0
        self.error: Optional[str] = None
        self.t_created: float = time.time()
        self.t_finished: Optional[float] = None

    def as_dict(self) -> dict:
        d: dict = {
            "task_id": self.id,
            "status": self.status,
            "progress": self.progress,
        }
        if self.status == "completed":
            d["result_url"] = f"/api/result/{self.id}"
        if self.status == "failed":
            d["error"] = self.error
        return d


_tasks: Dict[str, Task] = {}
_store_lock = threading.Lock()
_process_lock = threading.Lock()  # serialises image processing


# ═══════════════════════════════════════════════════════════════════════════════
#  UPLOAD VALIDATION
# ═══════════════════════════════════════════════════════════════════════════════

def _validate(data: bytes, name: str, ctype: Optional[str] = None) -> None:
    if not data:
        raise ValueError("Empty file")
    if len(data) > MAX_UPLOAD_BYTES:
        raise ValueError(
            f"File too large ({len(data) / 1_048_576:.1f} MB). "
            f"Maximum is {MAX_UPLOAD_MB} MB."
        )
    ext = Path(name).suffix.lower()
    if ext not in ALLOWED_SUFFIXES:
        raise ValueError(
            f"Unsupported format '{ext}'. "
            f"Accepted: {', '.join(sorted(ALLOWED_SUFFIXES))}"
        )
    if ctype and not ctype.startswith("image/"):
        raise ValueError(f"Content-Type '{ctype}' is not an image")
    try:
        Image.open(io.BytesIO(data)).verify()
    except Exception:
        raise ValueError("Corrupt or unreadable image file")


# ═══════════════════════════════════════════════════════════════════════════════
#  IMAGE PROCESSING PIPELINE  (every step frees intermediates + calls gc)
# ═══════════════════════════════════════════════════════════════════════════════

# --- 1. Face detection (Haar cascade) ----------------------------------------

def _detect_face(gray: np.ndarray):
    """
    Returns (x, y, w, h) of the largest detected face, or None.
    Tries strict parameters first, then relaxed.
    """
    _load_cascades()
    for params in [
        dict(scaleFactor=1.1, minNeighbors=5, minSize=(80, 80)),
        dict(scaleFactor=1.05, minNeighbors=3, minSize=(60, 60)),
    ]:
        faces = _face_cascade.detectMultiScale(
            gray, flags=cv2.CASCADE_SCALE_IMAGE, **params
        )
        if len(faces) > 0:
            areas = [w * h for (_, _, w, h) in faces]
            return faces[int(np.argmax(areas))]
    return None


# --- 2. Eye detection (Haar cascade, upper 60% of face) ----------------------

def _detect_eyes(gray: np.ndarray, fx: int, fy: int, fw: int, fh: int):
    """
    Returns ((lx, ly), (rx, ry)) eye centres in full-image coordinates,
    or None if fewer than two eyes are found.
    """
    _load_cascades()
    roi = gray[fy: fy + int(fh * 0.60), fx: fx + fw]
    if roi.size == 0:
        return None

    min_eye = max(12, int(fw * 0.08))
    max_eye = int(fw * 0.40)

    for neighbours in (5, 3):
        eyes = _eye_cascade.detectMultiScale(
            roi, scaleFactor=1.05, minNeighbors=neighbours,
            minSize=(min_eye, min_eye), maxSize=(max_eye, max_eye),
        )
        if len(eyes) >= 2:
            break
    else:
        return None

    # Convert to full-image coords, filter out detections near face edges
    cx_face = fw / 2
    candidates = []
    for ex, ey, ew, eh in eyes:
        ecx = fx + ex + ew // 2
        ecy = fy + ey + eh // 2
        # keep only detections roughly centred in the face
        if fx + fw * 0.05 < ecx < fx + fw * 0.95:
            candidates.append((ecx, ecy))

    if len(candidates) < 2:
        return None

    # Pick the pair with the widest horizontal gap
    best, best_d = None, 0.0
    for i in range(len(candidates)):
        for j in range(i + 1, len(candidates)):
            d = abs(candidates[i][0] - candidates[j][0])
            if d > best_d:
                best_d = d
                best = (candidates[i], candidates[j])

    if best and best_d > fw * 0.15:
        return best
    return None


# --- 3. Alignment (rotate so eyes are level) ---------------------------------

def _align(bgr: np.ndarray, eye_pair) -> np.ndarray:
    if eye_pair is None:
        return bgr
    (lx, ly), (rx, ry) = eye_pair
    if lx > rx:
        lx, ly, rx, ry = rx, ry, lx, ly
    angle = np.degrees(np.arctan2(ry - ly, rx - lx))
    if abs(angle) < 1.0:
        return bgr
    h, w = bgr.shape[:2]
    M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
    rotated = cv2.warpAffine(
        bgr, M, (w, h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE,
    )
    return rotated


# --- 4. Background removal (rembg / u2netp) ----------------------------------

def _remove_bg(rgb: np.ndarray) -> np.ndarray:
    """Returns BGRA array."""
    pil = Image.fromarray(rgb)
    session = _get_rembg()

    rgba_pil = None
    # Try with alpha matting first (better edges)
    try:
        rgba_pil = remove(
            pil, session=session,
            alpha_matting=True,
            alpha_matting_foreground_threshold=240,
            alpha_matting_background_threshold=10,
            alpha_matting_erode_size=10,
            post_process_mask=True,
        )
    except (MemoryError, RuntimeError) as exc:
        log.warning("Alpha-matting failed (%s); retrying without it.", exc)
        del rgba_pil
        gc.collect()
        rgba_pil = None

    if rgba_pil is None:
        try:
            rgba_pil = remove(
                pil, session=session,
                alpha_matting=False,
                post_process_mask=True,
            )
        except (MemoryError, RuntimeError):
            gc.collect()
            rgba_pil = remove(pil, session=session)

    result = np.array(rgba_pil)
    del rgba_pil
    pil.close()

    if result.ndim == 2:
        result = cv2.cvtColor(result, cv2.COLOR_GRAY2BGRA)
    elif result.shape[2] == 3:
        alpha = np.full(result.shape[:2], 255, dtype=np.uint8)
        result = np.dstack([result, alpha])

    bgra = cv2.cvtColor(result, cv2.COLOR_RGBA2BGRA)
    del result
    gc.collect()
    return bgra


# --- 5. Shadow reduction (CLAHE on L channel, blended) -----------------------

def _soften_shadows(bgra: np.ndarray) -> np.ndarray:
    bgr = bgra[:, :, :3]
    alpha = bgra[:, :, 3]
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    l_ch, a_ch, b_ch = cv2.split(lab)
    del lab

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_enhanced = clahe.apply(l_ch)
    l_out = cv2.addWeighted(l_enhanced, 0.55, l_ch, 0.45, 0)
    del l_ch, l_enhanced

    bgr_out = cv2.cvtColor(cv2.merge([l_out, a_ch, b_ch]), cv2.COLOR_LAB2BGR)
    del l_out, a_ch, b_ch
    return np.dstack([bgr_out, alpha])


# --- 6. White-balance correction (gray-world, foreground only) ---------------

def _white_balance(bgra: np.ndarray) -> np.ndarray:
    bgr = bgra[:, :, :3].astype(np.float32)
    alpha = bgra[:, :, 3]
    mask = alpha > 128
    if mask.sum() < 100:
        return bgra
    channels = list(cv2.split(bgr))
    del bgr
    avgs = [float(c[mask].mean()) for c in channels]
    mid = sum(avgs) / 3.0
    gains = [np.clip(mid / max(v, 1e-6), 0.80, 1.20) for v in avgs]
    corrected = [
        np.clip(c * g, 0, 255).astype(np.uint8)
        for c, g in zip(channels, gains)
    ]
    del channels
    merged = cv2.merge(corrected)
    del corrected
    result = np.dstack([merged, alpha])
    del merged
    return result


# --- 7. Passport-spec crop (face-centred, ICAO proportions) ------------------

def _passport_crop(bgra: np.ndarray, face) -> np.ndarray:
    h, w = bgra.shape[:2]
    fx, fy, fw, fh = int(face[0]), int(face[1]), int(face[2]), int(face[3])

    if fh < 80:
        raise ValueError(
            "Face is too small in the photo. "
            "Please upload a closer photo with a clearly visible face."
        )

    face_cx = fx + fw // 2
    # Estimate eye-line at ~40 % from top of face bounding box
    eye_y = fy + int(fh * 0.40)

    aspect = PASSPORT_W / PASSPORT_H
    crop_h = int(fh / 0.55)
    crop_w = int(crop_h * aspect)

    crop_top = int(eye_y - crop_h * 0.40)
    crop_left = int(face_cx - crop_w / 2)
    crop_bot = crop_top + crop_h
    crop_right = crop_left + crop_w

    # Padding if crop extends beyond image bounds
    pad_t = max(0, -crop_top)
    pad_l = max(0, -crop_left)
    pad_b = max(0, crop_bot - h)
    pad_r = max(0, crop_right - w)

    y1, x1 = max(0, crop_top), max(0, crop_left)
    y2, x2 = min(h, crop_bot), min(w, crop_right)
    cropped = bgra[y1:y2, x1:x2].copy()

    if pad_t or pad_l or pad_b or pad_r:
        cropped = cv2.copyMakeBorder(
            cropped, pad_t, pad_b, pad_l, pad_r,
            cv2.BORDER_CONSTANT, value=[0, 0, 0, 0],
        )
    return cropped


# --- 8. Composite onto blue background + resize ------------------------------

def _composite(bgra: np.ndarray) -> np.ndarray:
    h, w = bgra.shape[:2]
    bg = np.full((h, w, 3), BG_RGB[::-1], dtype=np.uint8)  # RGB→BGR
    alpha = bgra[:, :, 3:4].astype(np.float32) / 255.0
    fg = bgra[:, :, :3].astype(np.float32)
    blended = (fg * alpha + bg * (1.0 - alpha)).astype(np.uint8)
    del bg, alpha, fg, bgra
    result = cv2.resize(
        blended, (PASSPORT_W, PASSPORT_H), interpolation=cv2.INTER_LANCZOS4,
    )
    del blended
    gc.collect()
    return result


# --- Pipeline orchestrator ---------------------------------------------------

def _run_pipeline(task: Task) -> None:
    try:
        task.status = "processing"
        task.progress = 5

        # ── 1. Load image ────────────────────────────────────────────────
        bgr = cv2.imread(task.inp, cv2.IMREAD_COLOR)
        if bgr is None:
            raise ValueError("Could not decode image file.")
        h, w = bgr.shape[:2]
        if min(h, w) < MIN_DIMENSION:
            raise ValueError(
                f"Image too small ({w}×{h}). "
                f"Minimum {MIN_DIMENSION}px on the shortest side."
            )
        if max(h, w) > MAX_DIMENSION:
            scale = MAX_DIMENSION / max(h, w)
            bgr = cv2.resize(
                bgr, (int(w * scale), int(h * scale)),
                interpolation=cv2.INTER_AREA,
            )
        task.progress = 10

        # ── 2. Detect face ───────────────────────────────────────────────
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        face = _detect_face(gray)
        if face is None:
            raise ValueError(
                "No face detected. Please upload a clear, front-facing photo "
                "with the full face visible and eyes open."
            )
        task.progress = 20

        # ── 3. Detect eyes + align ───────────────────────────────────────
        fx, fy, fw, fh = [int(v) for v in face]
        eyes = _detect_eyes(gray, fx, fy, fw, fh)
        del gray
        bgr = _align(bgr, eyes)
        del eyes
        gc.collect()
        task.progress = 28

        # ── 4. Re-detect face on aligned image ───────────────────────────
        gray2 = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        face2 = _detect_face(gray2)
        del gray2
        if face2 is None:
            face2 = face
        task.progress = 32

        # ── 5. Background removal ────────────────────────────────────────
        log.info("[%s] Removing background (u2netp)…", task.id[:8])
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        del bgr
        fg = _remove_bg(rgb)
        del rgb
        gc.collect()
        task.progress = 58

        # ── 6. Shadow reduction ──────────────────────────────────────────
        fg = _soften_shadows(fg)
        gc.collect()
        task.progress = 68

        # ── 7. White-balance correction ──────────────────────────────────
        fg = _white_balance(fg)
        gc.collect()
        task.progress = 76

        # ── 8. Passport-spec crop ────────────────────────────────────────
        fg = _passport_crop(fg, face2)
        del face2
        gc.collect()
        task.progress = 86

        # ── 9. Composite + resize ────────────────────────────────────────
        final = _composite(fg)
        del fg
        gc.collect()
        task.progress = 93

        # ── 10. Save high-quality JPEG ───────────────────────────────────
        out_path = TMP / f"{task.id}.jpg"
        Image.fromarray(cv2.cvtColor(final, cv2.COLOR_BGR2RGB)).save(
            str(out_path), "JPEG", quality=95, subsampling=0,
        )
        del final
        gc.collect()

        task.out = str(out_path)
        task.progress = 100
        task.status = "completed"
        task.t_finished = time.time()
        log.info(
            "[%s] Done in %.1fs",
            task.id[:8], task.t_finished - task.t_created,
        )

    except ValueError as exc:
        task.status = "failed"
        task.error = str(exc)
        task.t_finished = time.time()
        log.warning("[%s] Failed: %s", task.id[:8], exc)
        gc.collect()

    except Exception as exc:
        task.status = "failed"
        task.error = "Internal processing error."
        task.t_finished = time.time()
        log.exception("[%s] Unexpected error: %s", task.id[:8], exc)
        gc.collect()


def _guarded_pipeline(task: Task) -> None:
    """Acquire the serial-processing lock, then run."""
    with _process_lock:
        _run_pipeline(task)


# ═══════════════════════════════════════════════════════════════════════════════
#  CLEANUP JANITOR  (sweeps expired temp files every 60 s)
# ═══════════════════════════════════════════════════════════════════════════════

def _sweep_expired() -> None:
    now = time.time()
    dead: list = []
    with _store_lock:
        for tid, t in _tasks.items():
            if t.t_finished and (now - t.t_finished) > TTL_SECONDS:
                dead.append(tid)
        for tid in dead:
            t = _tasks.pop(tid)
            for p in (t.inp, t.out):
                if p and os.path.exists(p):
                    try:
                        os.remove(p)
                    except OSError:
                        pass
    if dead:
        log.info("Swept %d expired task(s).", len(dead))


# ═══════════════════════════════════════════════════════════════════════════════
#  FASTAPI APPLICATION
# ═══════════════════════════════════════════════════════════════════════════════

@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info(
        "Passport Photo API starting  |  temp=%s  |  max_dim=%d",
        TMP, MAX_DIMENSION,
    )
    janitor = asyncio.create_task(_janitor_loop())
    yield
    janitor.cancel()
    shutil.rmtree(TMP, ignore_errors=True)
    log.info("Stopped — temp directory removed.")


async def _janitor_loop():
    while True:
        await asyncio.sleep(60)
        try:
            _sweep_expired()
        except Exception:
            log.exception("Janitor sweep failed.")


app = FastAPI(
    title="Passport Photo API",
    description=(
        "Automated passport-photo processing: face detection, "
        "background removal (U²-Net Portable), shadow reduction, "
        "white-balance correction, and ICAO-spec cropping."
    ),
    version="1.0.0-free",
    lifespan=lifespan,
)

# ── CORS ─────────────────────────────────────────────────────────────────────

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
    allow_headers=["*"],
    expose_headers=["Content-Disposition"],
)

# ── Rate limiting ────────────────────────────────────────────────────────────

_limiter = Limiter(
    key_func=get_remote_address,
    default_limits=["60/minute"],
)
app.state.limiter = _limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)


# ═══════════════════════════════════════════════════════════════════════════════
#  REST ENDPOINTS  — no HTML, no static files, pure API
# ═══════════════════════════════════════════════════════════════════════════════

@app.get("/api/health")
async def health():
    return {
        "status": "healthy",
        "version": "1.0.0-free",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "photo_requirements": {
            "formats": sorted(ALLOWED_SUFFIXES),
            "max_size": f"{MAX_UPLOAD_MB} MB",
            "min_dimension": f"{MIN_DIMENSION}px shortest side",
            "max_dimension": f"{MAX_DIMENSION}px longest side",
            "face": "Clearly visible, front-facing, eyes open",
        },
        "output": {
            "size": f"{PASSPORT_W}x{PASSPORT_H}px (35x45mm @ 300 DPI)",
            "background": f"RGB{BG_RGB} light blue",
            "format": "JPEG quality 95, 4:4:4 chroma",
        },
        "config": {
            "result_ttl_minutes": TTL_SECONDS // 60,
            "processing": "one image at a time",
            "model": "u2netp (lightweight)",
        },
    }


@app.post("/api/upload")
@_limiter.limit("10/minute")
async def upload(request: Request, file: UploadFile = File(...)):
    """Upload a photo. Returns a task_id for polling and download."""
    raw = await file.read()
    _validate(raw, file.filename or "unknown.jpg", file.content_type)

    tid = uuid.uuid4().hex
    ext = Path(file.filename or "photo.jpg").suffix.lower()
    inp_path = TMP / f"{tid}{ext}"
    inp_path.write_bytes(raw)

    task = Task(tid, str(inp_path))
    with _store_lock:
        _tasks[tid] = task

    threading.Thread(
        target=_guarded_pipeline, args=(task,), daemon=True,
    ).start()

    log.info("[%s] Queued  %s  (%.0f KB)", tid[:8], ext, len(raw) / 1024)

    return JSONResponse(
        status_code=202,
        content={
            "task_id": tid,
            "status": "queued",
            "message": "Upload accepted. Processing has started.",
            "status_url": f"/api/status/{tid}",
            "result_url": f"/api/result/{tid}",
        },
    )


@app.get("/api/status/{task_id}")
@_limiter.limit("120/minute")
async def get_status(request: Request, task_id: str):
    """Poll progress (0-100%)."""
    with _store_lock:
        task = _tasks.get(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found or expired.")
    return task.as_dict()


@app.get("/api/result/{task_id}")
@_limiter.limit("30/minute")
async def get_result(request: Request, task_id: str):
    """Download the finished passport photo as JPEG."""
    with _store_lock:
        task = _tasks.get(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found or expired.")
    if task.status == "failed":
        raise HTTPException(
            status_code=422, detail=f"Processing failed: {task.error}",
        )
    if task.status != "completed":
        raise HTTPException(
            status_code=202,
            detail="Still processing. Poll /api/status/{task_id}.",
        )
    if not task.out or not os.path.exists(task.out):
        raise HTTPException(status_code=410, detail="Result file has expired.")

    return FileResponse(
        path=task.out,
        media_type="image/jpeg",
        filename=f"passport_{task_id[:8]}.jpg",
        headers={"Cache-Control": "no-store, no-cache, must-revalidate"},
    )


@app.delete("/api/task/{task_id}")
@_limiter.limit("30/minute")
async def delete_task(request: Request, task_id: str):
    """Manually delete a task and its temporary files."""
    with _store_lock:
        task = _tasks.pop(task_id, None)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found.")
    for p in (task.inp, task.out):
        if p and os.path.exists(p):
            try:
                os.remove(p)
            except OSError:
                pass
    return {"detail": "Deleted", "task_id": task_id}


# ═══════════════════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host=HOST, port=PORT, workers=1, log_level="info")
