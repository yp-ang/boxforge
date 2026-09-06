# Step 05 — ONNX export and parity check

**Goal:** turn a trained checkpoint into a self-describing model folder that other
software can consume, and prove the conversion did not change the predictions.

**Depends on:** step 04.

This is the step that makes the model portable. See ARCHITECTURE §2 for why ONNX is the
contract.

## 1. Export

```python
# app/services/exporter.py
def export_onnx(weights: Path, out_dir: Path, imgsz: int = 640, opset: int = 12) -> Path:
    from ultralytics import YOLO
    model = YOLO(str(weights))
    produced = model.export(
        format="onnx",
        imgsz=imgsz,
        opset=opset,      # 12 = widest compatibility (TensorRT, OpenVINO, older ORT)
        simplify=True,    # onnx-simplifier: fewer nodes, fewer downstream surprises
        dynamic=False,    # FIXED batch+size. See below.
        nms=False,        # NMS in Python, not in the graph. See below.
    )
    return shutil.move(produced, out_dir / "model.onnx")
```

Three flags carry real consequences:

**`dynamic=False`.** A fixed `1×3×640×640` input is what TensorRT, OpenVINO and most VMS
integrations expect. Dynamic axes are more flexible and are also the single most common
cause of "works in Python, fails in DeepStream". Export fixed by default; offer dynamic as
an advanced option for consumers that want it.

**`nms=False`.** Keeping NMS out of the graph means the raw output is a plain tensor that
every runtime handles identically, and you control thresholds at inference time rather
than baking them in. The cost is ~20 lines of Python NMS, which you need anyway. Some
platforms prefer end-to-end NMS — make it a flag, default off.

**`opset=12`.** Newer opsets support more operators and fewer consumers. 12 is the
compatibility sweet spot; bump only when something needs it.

## 2. The model folder

```
data/models/<model_id>/
├── model.onnx
├── classes.json      {"0": "forklift", "1": "person", "2": "crate"}
├── metadata.json
└── best.pt           kept for retraining; not for deployment
```

`metadata.json` is what makes the folder self-describing — everything a consumer needs to
run it correctly without reading your code:

```json
{
  "name": "warehouse-v3",
  "task": "detect",
  "input": {"name": "images", "shape": [1, 3, 640, 640], "layout": "NCHW",
            "dtype": "float32", "range": "0-1", "color": "RGB",
            "preprocess": "letterbox, pad value 114"},
  "output": {"name": "output0", "shape": [1, 84, 8400],
             "format": "cxcywh + class scores, no objectness (YOLOv8/11 head)"},
  "classes": {"0": "forklift", "1": "person", "2": "crate"},
  "imgsz": 640, "opset": 12, "nms_in_graph": false,
  "metrics": {"mAP50": 0.81, "mAP50-95": 0.57},
  "trained_at": "2026-09-06T14:12:30Z",
  "dataset": "warehouse/20260906-141230",
  "framework_version": "ultralytics 8.x"
}
```

Write down the letterbox padding value and the colour order. Those two details are
responsible for most "the model works but detects nothing in the other system" reports.

## 3. Output decoding

YOLOv8/11 output is `[1, 4 + num_classes, 8400]` — transpose it, then threshold:

```python
def decode(raw: np.ndarray, conf: float, iou: float, nc: int):
    p = raw[0].T                       # (8400, 4+nc)
    boxes, scores = p[:, :4], p[:, 4:4+nc]
    cls  = scores.argmax(1)
    best = scores.max(1)
    keep = best > conf
    boxes, best, cls = boxes[keep], best[keep], cls[keep]
    boxes = cxcywh_to_xyxy(boxes)      # still in letterboxed 640-space
    return nms_per_class(boxes, best, cls, iou)
```

Note there is **no objectness score** in v8/v11 — a common porting bug from v5, where
confidence was `obj * cls`. Here class score *is* the confidence.

## 4. Parity check — do not skip this

Run the same image through the PyTorch model and the ONNX model and assert they agree.
This is the test that catches a wrong opset, a bad simplify pass, or a preprocessing
mismatch immediately, instead of after you've deployed to a VMS.

```python
def parity_check(weights: Path, onnx_path: Path, sample: Path,
                 conf=0.25, tol_box=2.0, tol_score=0.02) -> ParityReport:
    torch_dets = YOLO(str(weights)).predict(sample, conf=conf, verbose=False)[0]
    onnx_dets  = OnnxDetector(onnx_path).predict(sample, conf=conf)

    assert len(torch_dets) == len(onnx_dets), "detection count differs"
    for t, o in zip(sorted_by_score(torch_dets), sorted_by_score(onnx_dets)):
        assert t.cls == o.cls
        assert abs(t.score - o.score) < tol_score          # px tolerance in 640-space
        assert np.allclose(t.xyxy, o.xyxy, atol=tol_box)
```

Tolerances are non-zero because ONNX Runtime and PyTorch use different kernels; ~2px and
~0.02 confidence is normal float drift. A *different number of detections* is not drift —
that is a real bug, usually preprocessing.

Run parity automatically after every export and store the result in `metadata.json`. A
model that fails parity should be visibly flagged in the UI, not silently listed.

## 5. Optional downstream conversions

All derive from the same checkpoint; add them as buttons once ONNX works:

| Target | Call | Use case |
|---|---|---|
| OpenVINO IR | `model.export(format="openvino")` | Intel CPU/iGPU; big speedup on NUCs and most NVR boxes |
| CoreML | `model.export(format="coreml")` | macOS/iOS native, Neural Engine |
| TFLite | `model.export(format="tflite", int8=True)` | Coral TPU, mobile |
| TensorRT | build on the target machine from `model.onnx` | NVIDIA; engines are **not** portable across GPU/driver — build where you deploy |

Do not attempt to build TensorRT engines in this app. They are hardware-specific; ship
ONNX and let the target build its own.

## Acceptance criteria

- Export produces the four-file folder above.
- `python -c "import onnx; onnx.checker.check_model('model.onnx')"` passes.
- Parity check passes on 5 sample images; result recorded in metadata.
- Deliberately export with the wrong `imgsz` → parity fails loudly. (Verify the test works.)
- `onnxruntime` loads the model with no torch installed in the environment.
