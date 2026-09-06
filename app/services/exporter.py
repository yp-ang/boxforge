"""ONNX export and parity check (step 05). Together with trainer.py, one of the two
modules that import ultralytics; onnx/onnxruntime imports also stay inside functions so
the web process — which reads metadata built here — never pays for them at boot.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Protocol

import cv2
import numpy as np

# The one place this value is written down (step 05 §2): every consumer of model.onnx
# needs to letterbox-pad with this exact value or its predictions silently drift.
LETTERBOX_PAD_VALUE = 114


def num_anchors(imgsz: int) -> int:
    """YOLOv8/11's three detection strides (8/16/32) each contribute (imgsz/stride)^2
    anchors; e.g. 640 -> 8400. Used only to document the output shape in metadata.json."""
    return sum((imgsz // stride) ** 2 for stride in (8, 16, 32))


def export_onnx(weights: Path, out_dir: Path, imgsz: int = 640, opset: int = 12,
                dynamic: bool = False, nms: bool = False) -> Path:
    """Step 05 §1. Fixed batch+size and no NMS in the graph are the defaults every
    downstream runtime (TensorRT, OpenVINO, DeepStream) expects — see the doc for why."""
    import shutil

    from ultralytics import YOLO

    model = YOLO(str(weights))
    produced = model.export(format="onnx", imgsz=imgsz, opset=opset, simplify=True,
                            dynamic=dynamic, nms=nms)
    out_dir.mkdir(parents=True, exist_ok=True)
    dest = out_dir / "model.onnx"
    shutil.move(produced, dest)
    return dest


def check_onnx_model(onnx_path: Path) -> None:
    """Raises onnx.onnx_cpp2py_export.checker.ValidationError on a malformed graph."""
    import onnx

    onnx.checker.check_model(str(onnx_path))


# --- output decoding (step 05 §3) --------------------------------------------------

def cxcywh_to_xyxy(boxes: np.ndarray) -> np.ndarray:
    cx, cy, w, h = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    return np.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], axis=1)


def _iou_xyxy(box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    ix1, iy1 = np.maximum(box[0], boxes[:, 0]), np.maximum(box[1], boxes[:, 1])
    ix2, iy2 = np.minimum(box[2], boxes[:, 2]), np.minimum(box[3], boxes[:, 3])
    inter = np.clip(ix2 - ix1, 0, None) * np.clip(iy2 - iy1, 0, None)
    area_a = (box[2] - box[0]) * (box[3] - box[1])
    area_b = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    return inter / np.clip(area_a + area_b - inter, 1e-9, None)


def nms_per_class(boxes: np.ndarray, scores: np.ndarray, classes: np.ndarray,
                  iou_thresh: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Greedy NMS run independently per class, so two different classes never suppress
    one another."""
    keep: list[int] = []
    for c in np.unique(classes):
        order = np.nonzero(classes == c)[0]
        order = order[np.argsort(-scores[order])]
        while order.size:
            i = order[0]
            keep.append(int(i))
            if order.size == 1:
                break
            rest = order[1:]
            order = rest[_iou_xyxy(boxes[i], boxes[rest]) <= iou_thresh]
    keep.sort()
    return boxes[keep], scores[keep], classes[keep]


def decode(raw: np.ndarray, conf: float, iou: float, nc: int
          ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """YOLOv8/11 head: [1, 4+nc, N] -> (boxes_xyxy, scores, classes), still in
    letterboxed input-pixel space. No objectness score — class score *is* the
    confidence; treating it as `obj * cls` is the classic v5 porting bug."""
    p = raw[0].T                                   # (N, 4+nc)
    boxes_cxcywh, scores_all = p[:, :4], p[:, 4:4 + nc]
    classes = scores_all.argmax(1)
    scores = scores_all.max(1)
    keep = scores > conf
    boxes_cxcywh, scores, classes = boxes_cxcywh[keep], scores[keep], classes[keep]
    if boxes_cxcywh.shape[0] == 0:
        return np.zeros((0, 4)), scores, classes
    boxes = cxcywh_to_xyxy(boxes_cxcywh)
    return nms_per_class(boxes, scores, classes, iou)


def letterbox(img: np.ndarray, imgsz: int, pad_value: int = LETTERBOX_PAD_VALUE
             ) -> tuple[np.ndarray, float, int, int]:
    """Resize keeping aspect ratio, pad to an imgsz x imgsz square. Returns the canvas
    plus what is needed to map boxes back to the original image: the scale and the
    (left, top) padding.

    This has to match ultralytics' own LetterBox bit-for-bit — cv2's resize kernel, and
    its -0.1/+0.1 rounding split for odd padding — or the parity check in §4 fails on
    resize-kernel drift that has nothing to do with the ONNX export itself."""
    h, w = img.shape[:2]
    scale = min(imgsz / h, imgsz / w)
    nw, nh = round(w * scale), round(h * scale)
    dw, dh = (imgsz - nw) / 2, (imgsz - nh) / 2
    left, right = round(dw - 0.1), round(dw + 0.1)
    top, bottom = round(dh - 0.1), round(dh + 0.1)

    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    canvas = cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_CONSTANT,
                                value=(pad_value, pad_value, pad_value))
    return canvas, scale, left, top


# --- detectors: real (ONNX Runtime) and the torch side of the parity check ---------

@dataclass
class Detection:
    cls: int
    score: float
    xyxy: np.ndarray       # x1,y1,x2,y2 in original-image pixel space


class Detector(Protocol):
    def predict(self, image: Path, conf: float, iou: float) -> list[Detection]: ...


class OnnxDetector:
    """The reference implementation of ARCHITECTURE's `runtime.py`: onnxruntime, a
    letterbox, and the decode() above. No torch import — this is what proves a model
    folder is portable to a process that doesn't have torch installed."""

    def __init__(self, onnx_path: Path):
        import onnxruntime as ort

        self.session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name
        self.imgsz = int(self.session.get_inputs()[0].shape[-1])
        self.nc = int(self.session.get_outputs()[0].shape[1]) - 4

    def predict(self, image: Path, conf: float = 0.25, iou: float = 0.5) -> list[Detection]:
        bgr = cv2.imread(str(image))
        orig_h, orig_w = bgr.shape[:2]

        canvas, scale, left, top = letterbox(bgr, self.imgsz)
        rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
        blob = (rgb.astype(np.float32) / 255.0).transpose(2, 0, 1)[None]
        raw = self.session.run(None, {self.input_name: blob})[0]

        boxes, scores, classes = decode(raw, conf, iou, self.nc)
        if boxes.shape[0]:
            boxes = boxes.copy()
            boxes[:, [0, 2]] = np.clip((boxes[:, [0, 2]] - left) / scale, 0, orig_w)
            boxes[:, [1, 3]] = np.clip((boxes[:, [1, 3]] - top) / scale, 0, orig_h)
        return [Detection(cls=int(c), score=float(s), xyxy=b)
                for b, s, c in zip(boxes, scores, classes)]


class TorchDetector:
    """The PyTorch side of the parity check — ultralytics' own pre/post-processing,
    used as the ground truth the ONNX export must reproduce."""

    def __init__(self, weights: Path, imgsz: int):
        from ultralytics import YOLO

        self._model = YOLO(str(weights))
        self._imgsz = imgsz

    def predict(self, image: Path, conf: float = 0.25, iou: float = 0.5) -> list[Detection]:
        # rect=False: ultralytics' default single-image inference pads only to the
        # nearest stride multiple (rectangular, usually smaller than imgsz on one side),
        # not to a square imgsz x imgsz — but that is exactly the fixed shape the ONNX
        # graph was exported with. Without this the two are letterboxed differently and
        # parity fails on a preprocessing difference that was never a real bug.
        result = self._model.predict(str(image), conf=conf, iou=iou, imgsz=self._imgsz,
                                     rect=False, verbose=False)[0]
        boxes = result.boxes
        return [Detection(cls=int(c), score=float(s), xyxy=np.asarray(b)) for b, s, c in
                zip(boxes.xyxy.cpu().numpy(), boxes.conf.cpu().numpy(), boxes.cls.cpu().numpy())]


# --- parity check (step 05 §4) ------------------------------------------------------

@dataclass
class ParityImageResult:
    image: str
    passed: bool
    detail: str


@dataclass
class ParityReport:
    passed: bool = True
    conf: float = 0.25
    tol_box: float = 2.0
    tol_score: float = 0.02
    images: list[ParityImageResult] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {**asdict(self), "images": [asdict(i) for i in self.images]}


def _sorted_by_score(dets: list[Detection]) -> list[Detection]:
    return sorted(dets, key=lambda d: -d.score)


def parity_check(torch_detector: Detector, onnx_detector: Detector, samples: list[Path],
                 conf: float = 0.25, iou: float = 0.5, tol_box: float = 2.0,
                 tol_score: float = 0.02) -> ParityReport:
    """Run the same image through both detectors and assert they agree (step 05 §4).
    Tolerances are non-zero because ONNX Runtime and PyTorch use different kernels; a
    *different number of detections* is not drift — that is a real bug, usually a
    preprocessing or opset mismatch, and is what this is here to catch."""
    report = ParityReport(conf=conf, tol_box=tol_box, tol_score=tol_score)
    for sample in samples:
        try:
            torch_dets = _sorted_by_score(torch_detector.predict(sample, conf=conf, iou=iou))
            onnx_dets = _sorted_by_score(onnx_detector.predict(sample, conf=conf, iou=iou))

            if len(torch_dets) != len(onnx_dets):
                raise AssertionError(
                    f"detection count differs: torch={len(torch_dets)} onnx={len(onnx_dets)}"
                )
            for t, o in zip(torch_dets, onnx_dets):
                if t.cls != o.cls:
                    raise AssertionError(f"class differs: torch={t.cls} onnx={o.cls}")
                if abs(t.score - o.score) >= tol_score:
                    raise AssertionError(
                        f"score differs by {abs(t.score - o.score):.4f} (tol {tol_score})"
                    )
                if not np.allclose(t.xyxy, o.xyxy, atol=tol_box):
                    raise AssertionError(
                        f"box differs beyond {tol_box}px: torch={t.xyxy.tolist()} "
                        f"onnx={o.xyxy.tolist()}"
                    )
        except AssertionError as exc:
            report.passed = False
            report.images.append(ParityImageResult(str(sample), False, str(exc)))
        else:
            report.images.append(
                ParityImageResult(str(sample), True, f"{len(torch_dets)} detections match")
            )
    return report


# --- metadata (step 05 §2) ----------------------------------------------------------

def build_export_metadata(
    *, name: str, task: str, imgsz: int, opset: int, nms: bool,
    classes: dict[str, str], metrics: dict[str, float], trained_at: str,
    dataset: str, framework_version: str, parity: ParityReport | None,
) -> dict[str, Any]:
    """Everything a consumer needs to run model.onnx correctly without reading our
    code — see step 05 §2 for why the letterbox padding value and colour order matter
    more than anything else in here."""
    return {
        "name": name,
        "task": task,
        "input": {
            "name": "images", "shape": [1, 3, imgsz, imgsz], "layout": "NCHW",
            "dtype": "float32", "range": "0-1", "color": "RGB",
            "preprocess": f"letterbox, pad value {LETTERBOX_PAD_VALUE}",
        },
        "output": {
            "name": "output0", "shape": [1, 4 + len(classes), num_anchors(imgsz)],
            "format": "cxcywh + class scores, no objectness (YOLOv8/11 head)",
        },
        "classes": classes,
        "imgsz": imgsz,
        "opset": opset,
        "nms_in_graph": nms,
        "metrics": metrics,
        "trained_at": trained_at,
        "dataset": dataset,
        "framework_version": framework_version,
        "parity": parity.to_dict() if parity else None,
    }
