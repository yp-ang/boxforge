"""Step 08: face detection + recognition (docs/steps/08-faces.md).

Read ARCHITECTURE §5.4 before using this for anything beyond your own photos: face
*recognition* (matching an identity against a gallery, as opposed to just "is there a
face here") is biometric processing and is regulated differently from object detection
in several jurisdictions, and InsightFace's pretrained weights below are licensed for
non-commercial research use only.

Detection is SCRFD, recognition is ArcFace — both ONNX, both from InsightFace's
`buffalo_l` pack, downloaded on first use exactly like services/trainer.py fetches
yolo11n.pt on demand. Pipeline: detect faces -> align each to ArcFace's 112x112
template using the 5 landmarks SCRFD already gives us -> embed -> cosine-match against
a gallery built from FaceEmbedding rows.
"""
from __future__ import annotations

import shutil
import tempfile
import threading
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.request import urlopen

import cv2
import numpy as np
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.config import settings

BUFFALO_L_URL = "https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip"
DET_MODEL_FILE = "det_10g.onnx"
EMBED_MODEL_FILE = "w600k_r50.onnx"
BUFFALO_L_FILES = (DET_MODEL_FILE, EMBED_MODEL_FILE)

DET_INPUT_SIZE = (640, 640)     # (width, height)
DET_STRIDES = (8, 16, 32)
DET_NUM_ANCHORS = 2
DEFAULT_DET_CONF = 0.5
DEFAULT_DET_NMS = 0.4

EMBED_DIM = 512
# Doc §4: 0.28 loose, 0.35 balanced, 0.45 strict. Calibrate on your own data rather than
# trusting this — it's a starting point, not a claim about your gallery.
DEFAULT_MATCH_THRESHOLD = 0.35

# Similarity transform target: ArcFace's canonical 112x112 template (doc §2). This is
# the one place these numbers are written down — every aligned crop must land here or
# the embedding model sees a face in a pose it was never trained on.
ARCFACE_5PT = np.array([[38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366],
                        [41.5493, 92.3655], [70.7299, 92.2041]], dtype=np.float32)


def buffalo_l_dir() -> Path:
    return settings.pretrained_dir / "buffalo_l"


def buffalo_l_ready() -> bool:
    d = buffalo_l_dir()
    return all((d / f).is_file() for f in BUFFALO_L_FILES)


def download_buffalo_l(log=print) -> None:
    """Fetch the ~325MB buffalo_l release zip and keep only the two files this pipeline
    needs (det_10g.onnx, w600k_r50.onnx) — not the gender/age/3D-landmark models also in
    the pack. Safe to call repeatedly: a no-op once both files are present."""
    if buffalo_l_ready():
        log("buffalo_l already present, skipping download")
        return
    dest = buffalo_l_dir()
    dest.mkdir(parents=True, exist_ok=True)

    fd, tmp_name = tempfile.mkstemp(suffix=".zip")
    tmp_path = Path(tmp_name)
    try:
        import os
        os.close(fd)
        log(f"downloading {BUFFALO_L_URL} ...")
        with urlopen(BUFFALO_L_URL, timeout=120) as resp, open(tmp_path, "wb") as out:
            shutil.copyfileobj(resp, out)
        log("extracting det_10g.onnx and w600k_r50.onnx ...")
        with zipfile.ZipFile(tmp_path) as zf:
            for name in BUFFALO_L_FILES:
                with zf.open(name) as src, open(dest / name, "wb") as out:
                    shutil.copyfileobj(src, out)
        log(f"buffalo_l ready at {dest}")
    finally:
        tmp_path.unlink(missing_ok=True)


def _providers() -> list[str]:
    import onnxruntime as ort

    avail = ort.get_available_providers()
    return [p for p in ("CUDAExecutionProvider", "CoreMLExecutionProvider") if p in avail] \
        + ["CPUExecutionProvider"]


# --- SCRFD detection (step 08 §2) ---------------------------------------------------

@dataclass
class FaceDet:
    score: float
    bbox: list[float]         # x1,y1,x2,y2 in original-image pixel space
    landmarks: np.ndarray     # (5, 2) — eyes, nose, mouth corners, original-image pixels


def _distance2bbox(points: np.ndarray, distance: np.ndarray) -> np.ndarray:
    x1 = points[:, 0] - distance[:, 0]
    y1 = points[:, 1] - distance[:, 1]
    x2 = points[:, 0] + distance[:, 2]
    y2 = points[:, 1] + distance[:, 3]
    return np.stack([x1, y1, x2, y2], axis=-1)


def _distance2kps(points: np.ndarray, distance: np.ndarray) -> np.ndarray:
    preds = []
    for i in range(0, distance.shape[1], 2):
        preds.append(points[:, i % 2] + distance[:, i])
        preds.append(points[:, i % 2 + 1] + distance[:, i + 1])
    return np.stack(preds, axis=-1)


def _iou(box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    ix1 = np.maximum(box[0], boxes[:, 0])
    iy1 = np.maximum(box[1], boxes[:, 1])
    ix2 = np.minimum(box[2], boxes[:, 2])
    iy2 = np.minimum(box[3], boxes[:, 3])
    inter = np.clip(ix2 - ix1, 0, None) * np.clip(iy2 - iy1, 0, None)
    area_a = (box[2] - box[0]) * (box[3] - box[1])
    area_b = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    return inter / np.clip(area_a + area_b - inter, 1e-9, None)


def _nms(boxes: np.ndarray, scores: np.ndarray, iou_thresh: float) -> list[int]:
    order = list(scores.argsort()[::-1])
    keep: list[int] = []
    while order:
        i = order.pop(0)
        keep.append(int(i))
        if not order:
            break
        rest = np.array(order)
        order = list(rest[_iou(boxes[i], boxes[rest]) <= iou_thresh])
    return keep


class ScrfdDetector:
    """Decode matches InsightFace's own scrfd.py: per-stride anchor centres, boxes and
    landmarks encoded as distances from those centres, NMS across the pooled candidates
    from all three strides (verified against a real buffalo_l checkpoint — see the step
    08 test suite)."""

    def __init__(self, model_path: Path):
        import onnxruntime as ort

        self.session = ort.InferenceSession(str(model_path), providers=_providers())
        self.input_name = self.session.get_inputs()[0].name

    def detect(self, bgr: np.ndarray, conf: float = DEFAULT_DET_CONF,
              iou: float = DEFAULT_DET_NMS) -> list[FaceDet]:
        h, w = bgr.shape[:2]
        iw, ih = DET_INPUT_SIZE
        im_ratio = h / w
        model_ratio = ih / iw
        if im_ratio > model_ratio:
            new_h, new_w = ih, int(ih / im_ratio)
        else:
            new_w, new_h = iw, int(iw * im_ratio)
        det_scale = new_h / h

        resized = cv2.resize(bgr, (new_w, new_h))
        canvas = np.zeros((ih, iw, 3), dtype=np.uint8)
        canvas[:new_h, :new_w, :] = resized

        blob = cv2.dnn.blobFromImage(canvas, 1.0 / 128.0, DET_INPUT_SIZE, (127.5,) * 3,
                                     swapRB=True)
        outs = self.session.run(None, {self.input_name: blob})

        fmc = len(DET_STRIDES)
        all_scores, all_boxes, all_kps = [], [], []
        for idx, stride in enumerate(DET_STRIDES):
            scores = outs[idx].reshape(-1)
            bbox_preds = outs[idx + fmc] * stride
            kps_preds = outs[idx + fmc * 2] * stride
            fh, fw = ih // stride, iw // stride
            centers = np.stack(np.mgrid[:fh, :fw][::-1], axis=-1).astype(np.float32)
            centers = (centers * stride).reshape(-1, 2)
            if DET_NUM_ANCHORS > 1:
                centers = np.stack([centers] * DET_NUM_ANCHORS, axis=1).reshape(-1, 2)

            above = np.nonzero(scores >= conf)[0]
            if above.size == 0:
                continue
            boxes = _distance2bbox(centers, bbox_preds) / det_scale
            kps = _distance2kps(centers, kps_preds).reshape(-1, 5, 2) / det_scale
            all_scores.append(scores[above])
            all_boxes.append(boxes[above])
            all_kps.append(kps[above])

        if not all_scores:
            return []
        scores = np.concatenate(all_scores)
        boxes = np.concatenate(all_boxes)
        kps = np.concatenate(all_kps)
        keep = _nms(boxes, scores, iou)
        return [FaceDet(score=float(scores[i]), bbox=[float(v) for v in boxes[i]],
                        landmarks=kps[i]) for i in keep]


def align(bgr: np.ndarray, landmarks5: np.ndarray) -> np.ndarray:
    """Similarity transform onto ArcFace's canonical template (doc §2). Skipping this
    and feeding a raw crop is the most common reason a face pipeline 'sort of works' —
    recognition accuracy falls off a cliff with pose variation."""
    M, _ = cv2.estimateAffinePartial2D(landmarks5.astype(np.float32), ARCFACE_5PT,
                                       method=cv2.LMEDS)
    return cv2.warpAffine(bgr, M, (112, 112), borderValue=0.0)


class ArcFaceEmbedder:
    def __init__(self, model_path: Path):
        import onnxruntime as ort

        self.session = ort.InferenceSession(str(model_path), providers=_providers())
        self.input_name = self.session.get_inputs()[0].name

    def embed(self, aligned_bgr: np.ndarray) -> np.ndarray:
        blob = cv2.dnn.blobFromImage(aligned_bgr, 1.0 / 127.5, (112, 112), (127.5,) * 3,
                                     swapRB=True)
        emb = self.session.run(None, {self.input_name: blob})[0][0]
        return emb / np.linalg.norm(emb)


# --- gallery (step 08 §3) -----------------------------------------------------------

class Gallery:
    """The whole gallery held as one (N, 512) matrix, per ARCHITECTURE §2 — at personal
    scale a matmul is faster than standing up FAISS."""

    def __init__(self):
        self.M = np.zeros((0, EMBED_DIM), dtype=np.float32)
        self.ids: list[int] = []

    def load(self, db: Session) -> None:
        from app.models import FaceEmbedding

        rows = db.scalars(select(FaceEmbedding)).all()
        self.M = (np.stack([np.frombuffer(r.vector_blob, dtype=np.float32) for r in rows])
                  if rows else np.zeros((0, EMBED_DIM), dtype=np.float32))
        self.ids = [r.identity_id for r in rows]

    def match(self, emb: np.ndarray, threshold: float = DEFAULT_MATCH_THRESHOLD
             ) -> tuple[int | None, float]:
        if not self.ids:
            return None, 0.0
        sims = self.M @ emb
        i = int(sims.argmax())
        return (self.ids[i], float(sims[i])) if sims[i] >= threshold else (None, float(sims[i]))


# --- combined engine + process-wide caches ------------------------------------------

class FaceEngine:
    def __init__(self):
        d = buffalo_l_dir()
        self.detector = ScrfdDetector(d / DET_MODEL_FILE)
        self.embedder = ArcFaceEmbedder(d / EMBED_MODEL_FILE)


_engine: FaceEngine | None = None
_gallery: Gallery | None = None
_lock = threading.Lock()


def get_engine() -> FaceEngine:
    global _engine
    with _lock:
        if _engine is None:
            if not buffalo_l_ready():
                raise RuntimeError(
                    "face models not downloaded yet — POST /api/faces/models/download first"
                )
            _engine = FaceEngine()
        return _engine


def get_gallery(db: Session) -> Gallery:
    """Cached like get_engine(), but reloaded on every call that follows a gallery
    change (see invalidate_gallery()) — a stale gallery would silently keep matching
    against a deleted identity's embeddings."""
    global _gallery
    with _lock:
        if _gallery is None:
            _gallery = Gallery()
            _gallery.load(db)
        return _gallery


def invalidate_gallery() -> None:
    global _gallery
    with _lock:
        _gallery = None


@dataclass
class IdentifiedFace:
    score: float
    bbox: list[float]
    identity_id: int | None
    similarity: float


def identify_faces(engine: FaceEngine, gallery: Gallery, bgr: np.ndarray,
                   det_conf: float = DEFAULT_DET_CONF,
                   match_threshold: float = DEFAULT_MATCH_THRESHOLD) -> list[IdentifiedFace]:
    dets = engine.detector.detect(bgr, conf=det_conf)
    out = []
    for d in dets:
        aligned = align(bgr, d.landmarks)
        emb = engine.embedder.embed(aligned)
        identity_id, sim = gallery.match(emb, threshold=match_threshold)
        out.append(IdentifiedFace(score=d.score, bbox=d.bbox, identity_id=identity_id,
                                  similarity=sim))
    return out


def draw_faces(bgr: np.ndarray, faces: list[IdentifiedFace], names: dict[int, str]) -> None:
    """Mutates bgr in place. Always shows the similarity score next to the name (doc
    §4) — 'Alice 0.36' and 'Alice 0.71' are very different claims."""
    for f in faces:
        x1, y1, x2, y2 = (int(round(v)) for v in f.bbox)
        name = names.get(f.identity_id, "unknown") if f.identity_id is not None else "unknown"
        color = (66, 133, 244) if f.identity_id is not None else (150, 150, 150)
        cv2.rectangle(bgr, (x1, y1), (x2, y2), (0, 0, 0), 4)
        cv2.rectangle(bgr, (x1, y1), (x2, y2), color, 2)

        label = f"{name} {f.similarity:.2f}"
        (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        chip_top = max(0, y1 - th - baseline - 4)
        cv2.rectangle(bgr, (x1, chip_top), (x1 + tw + 6, chip_top + th + baseline + 4),
                      color, -1)
        cv2.putText(bgr, label, (x1 + 3, chip_top + th + 1), cv2.FONT_HERSHEY_SIMPLEX,
                   0.5, (255, 255, 255), 1, cv2.LINE_AA)


def build_gallery_npz(db: Session) -> dict[str, Any]:
    """Doc §5 export: enough for another process to run the same matching without this
    app or its database — vectors, the identity id each belongs to, and id->name."""
    from app.models import FaceEmbedding, FaceIdentity

    rows = db.scalars(select(FaceEmbedding)).all()
    vectors = (np.stack([np.frombuffer(r.vector_blob, dtype=np.float32) for r in rows])
              if rows else np.zeros((0, EMBED_DIM), dtype=np.float32))
    ids = np.array([r.identity_id for r in rows], dtype=np.int64)
    names = {i.id: i.name for i in db.scalars(select(FaceIdentity))}
    return {"vectors": vectors, "identity_ids": ids, "names": names}
