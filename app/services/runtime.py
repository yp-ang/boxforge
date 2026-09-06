"""Inference runtime for the verify playground (step 06).

Loads model.onnx — the artifact step 05 exports — never best.pt. This is what proves
the file handed to other software actually works, per README ground rule 3. Shares its
letterbox/decode/NMS with services/exporter.py, which already got that pipeline
bit-exact against Ultralytics' own postprocessing (step 05 §4); duplicating it here
would risk the two silently drifting apart.

onnxruntime and cv2 stay cheap enough to import at module scope (unlike ultralytics/
torch in exporter.py) — the web process already needs cv2 for jobs unrelated to
training, and onnxruntime is the one runtime this whole step exists to exercise.
"""
from __future__ import annotations

import json
import threading
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from app.services.exporter import decode, letterbox

DEFAULT_IOU = 0.45

# Fixed, so the same class is always the same colour across images/frames — a green box
# that turns red between two review sessions is confusing in a way that's easy to avoid.
PALETTE = [
    (66, 133, 244), (219, 68, 55), (244, 180, 0), (15, 157, 88),
    (171, 71, 188), (255, 112, 67), (0, 172, 193), (156, 204, 101),
    (240, 98, 146), (121, 85, 72), (158, 158, 158), (57, 73, 171),
]


def color_for_class(cls: int) -> tuple[int, int, int]:
    return PALETTE[cls % len(PALETTE)]


@dataclass
class Det:
    cls: int
    label: str
    score: float
    xyxy: list[float]          # x1,y1,x2,y2 in original-image pixel space

    def as_dict(self) -> dict:
        return {"cls": self.cls, "label": self.label, "score": self.score, "xyxy": self.xyxy}


class OnnxDetector:
    """One ONNX Runtime session per model. Construction takes a second or two, so
    callers must go through get_detector() below rather than build one per frame."""

    def __init__(self, model_dir: Path):
        import onnxruntime as ort

        meta = json.loads((model_dir / "metadata.json").read_text())
        classes_path = model_dir / "classes.json"
        raw_classes = json.loads(classes_path.read_text()) if classes_path.is_file() \
            else meta["classes"]
        self.classes = {int(k): v for k, v in raw_classes.items()}
        self.imgsz = int(meta["imgsz"])
        self.session = ort.InferenceSession(
            str(model_dir / "model.onnx"), providers=self._providers()
        )
        self.input_name = self.session.get_inputs()[0].name

    @staticmethod
    def _providers() -> list[str]:
        import onnxruntime as ort

        # CoreML on Apple Silicon, CUDA on NVIDIA, CPU always last as fallback.
        avail = ort.get_available_providers()
        return [p for p in ("CUDAExecutionProvider", "CoreMLExecutionProvider") if p in avail] \
            + ["CPUExecutionProvider"]

    def predict(self, bgr: np.ndarray, conf: float, iou: float = DEFAULT_IOU) -> list[Det]:
        canvas, scale, left, top = letterbox(bgr, self.imgsz)
        rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
        blob = (rgb.astype(np.float32) / 255.0).transpose(2, 0, 1)[None]
        raw = self.session.run(None, {self.input_name: blob})[0]

        boxes, scores, classes = decode(raw, conf, iou, len(self.classes))
        orig_h, orig_w = bgr.shape[:2]
        if boxes.shape[0]:
            boxes = boxes.copy()
            boxes[:, [0, 2]] = np.clip((boxes[:, [0, 2]] - left) / scale, 0, orig_w)
            boxes[:, [1, 3]] = np.clip((boxes[:, [1, 3]] - top) / scale, 0, orig_h)
        return [
            Det(cls=int(c), label=self.classes.get(int(c), str(int(c))), score=float(s),
                xyxy=[round(float(v), 1) for v in b])
            for b, s, c in zip(boxes, scores, classes)
        ]


_cache: dict[int, OnnxDetector] = {}
_lock = threading.Lock()


def get_detector(model_id: int, model_dir: Path) -> OnnxDetector:
    """Cache by model id — see class docstring for why this matters per frame/request."""
    with _lock:
        det = _cache.get(model_id)
        if det is None:
            det = OnnxDetector(model_dir)
            _cache[model_id] = det
        return det


def invalidate(model_id: int) -> None:
    """Call after a re-export: the cached session still points at the old model.onnx
    bytes on the OS's file cache, and a new export can also change the class list."""
    with _lock:
        _cache.pop(model_id, None)


def draw_detections(bgr: np.ndarray, dets: list[Det]) -> None:
    """Consistent per-class colours, a 2px box outlined in 1px dark so it stays visible
    against both a white wall and a dark floor, and a label chip reading e.g.
    'person 0.87' (step 06 §5). Mutates bgr in place."""
    for d in dets:
        x1, y1, x2, y2 = (int(round(v)) for v in d.xyxy)
        color = color_for_class(d.cls)
        cv2.rectangle(bgr, (x1, y1), (x2, y2), (0, 0, 0), 4)
        cv2.rectangle(bgr, (x1, y1), (x2, y2), color, 2)

        label = f"{d.label} {d.score:.2f}"
        (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        chip_top = max(0, y1 - th - baseline - 4)
        cv2.rectangle(bgr, (x1, chip_top), (x1 + tw + 6, chip_top + th + baseline + 4),
                      color, -1)
        cv2.putText(bgr, label, (x1 + 3, chip_top + th + 1), cv2.FONT_HERSHEY_SIMPLEX,
                   0.5, (255, 255, 255), 1, cv2.LINE_AA)
