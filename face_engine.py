"""
face_engine.py
--------------
Turns a photo into a 512-number "face fingerprint" (embedding) using InsightFace,
and generates small thumbnails for the directory preview.
"""

import os
import base64
import threading
from typing import Optional

import numpy as np
import cv2

_MODEL_NAME = os.environ.get("INSIGHTFACE_MODEL", "buffalo_l")
_DET_SIZE = int(os.environ.get("DET_SIZE", "640"))

_app = None
_lock = threading.Lock()


def _get_app():
    global _app
    if _app is None:
        with _lock:
            if _app is None:
                from insightface.app import FaceAnalysis
                a = FaceAnalysis(name=_MODEL_NAME, providers=["CPUExecutionProvider"])
                a.prepare(ctx_id=0, det_size=(_DET_SIZE, _DET_SIZE))
                _app = a
    return _app


def _largest_face(faces):
    return max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))


def embedding_from_image(img) -> Optional[np.ndarray]:
    if img is None:
        return None
    app = _get_app()
    with _lock:
        faces = app.get(img)
    if not faces:
        return None
    face = _largest_face(faces)
    return np.asarray(face.embedding, dtype=np.float32)


def embedding_from_bytes(data: bytes) -> Optional[np.ndarray]:
    if not data:
        return None
    arr = np.frombuffer(data, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    return embedding_from_image(img)


def make_thumbnail(data: bytes, max_size: int = 150) -> Optional[str]:
    """Resize an image to a small thumbnail and return it as a base64 JPEG string."""
    if not data:
        return None
    arr = np.frombuffer(data, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        return None
    h, w = img.shape[:2]
    scale = max_size / max(h, w)
    if scale < 1:
        img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    _, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 70])
    return base64.b64encode(buf.tobytes()).decode("ascii")
