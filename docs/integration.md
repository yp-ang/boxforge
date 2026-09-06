# Integrating the exported model

The app publishes a self-describing folder (ARCHITECTURE §2, step 05):

```
model.onnx  classes.json  metadata.json  best.pt
```

Anything that reads ONNX can consume it. This document is the reference for the targets
worth caring about.

## The contract

| Property | Value |
|---|---|
| Input | `1×3×640×640` float32, NCHW |
| Colour | **RGB** (OpenCV gives you BGR — convert) |
| Range | 0.0–1.0 (divide by 255) |
| Resize | Letterbox, aspect preserved, **centred** pad, value 114 |
| Output | `1×(4+nc)×8400`, `cxcywh` + per-class scores in letterboxed 640-space |
| Objectness | **None.** v8/v11 confidence is the class score itself |
| NMS | Not in the graph — apply it yourself |

Nearly every integration failure is one of the first four rows. When a model "detects
nothing" in another system, check colour order and padding before you suspect the model.

## Plain Python — the reference implementation

Any consumer can be checked against this:

```python
import cv2, json, numpy as np, onnxruntime as ort

meta    = json.load(open("metadata.json"))
classes = json.load(open("classes.json"))
sess    = ort.InferenceSession("model.onnx", providers=["CPUExecutionProvider"])
name    = sess.get_inputs()[0].name
S       = meta["imgsz"]

img = cv2.imread("test.jpg")
h, w = img.shape[:2]
r = min(S/h, S/w)
nh, nw = round(h*r), round(w*r)
canvas = np.full((S, S, 3), 114, np.uint8)
top, left = (S-nh)//2, (S-nw)//2
canvas[top:top+nh, left:left+nw] = cv2.resize(img, (nw, nh))
x = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
x = x.transpose(2, 0, 1)[None]

p = sess.run(None, {name: x})[0][0].T          # (8400, 4+nc)
scores = p[:, 4:].max(1); cls = p[:, 4:].argmax(1)
keep = scores > 0.25
boxes = p[keep, :4]; scores = scores[keep]; cls = cls[keep]

xyxy = np.stack([boxes[:,0]-boxes[:,2]/2, boxes[:,1]-boxes[:,3]/2,
                 boxes[:,0]+boxes[:,2]/2, boxes[:,1]+boxes[:,3]/2], 1)
idx = cv2.dnn.NMSBoxes(xyxy.tolist(), scores.tolist(), 0.25, 0.45)
xyxy = (xyxy[idx] - [left, top, left, top]) / r   # back to original pixels
```

`cv2.dnn.NMSBoxes` saves writing NMS. Note it is class-agnostic — for per-class NMS, offset
each class's boxes by `cls * 10000` before the call, a standard trick.

## Frigate (NVR — easiest real target)

Frigate is the most likely place a personal detector earns its keep. It supports ONNX
directly, and YOLO-family models specifically.

```yaml
model:
  path: /config/model_cache/model.onnx
  input_tensor: nchw
  input_pixel_format: rgb
  width: 640
  height: 640
  labelmap_path: /config/labelmap.txt
detectors:
  onnx:
    type: onnx
```

`labelmap.txt` is one class name per line, in class-index order — generate it from
`classes.json`.

Frigate's detector plugin set moves between releases; check its docs for your version.
The model artifact does not change, only the config keys.

## OpenVINO (Intel CPU / iGPU)

The highest-value conversion if you deploy to a NUC, a mini-PC or most NVR hardware —
commonly 2–4× over generic CPU inference.

```bash
ovc model.onnx --output_model model_openvino   # or export directly from step 05
```

```python
from openvino.runtime import Core
compiled = Core().compile_model("model_openvino/model.xml", "CPU")   # or "GPU"
out = compiled([x])[compiled.output(0)]
```

Same pre/post-processing, verbatim.

## NVIDIA DeepStream / TensorRT

For multi-stream RTSP at scale. Build the engine **on the deployment machine** — TensorRT
engines are tied to GPU architecture and driver version and are not portable (step 05 §5).

```bash
trtexec --onnx=model.onnx --saveEngine=model.engine --fp16
```

DeepStream needs a `config_infer_primary.txt` and a custom output parser for the YOLO head.
The community `nvdsinfer_custom_impl_Yolo` parsers cover v8/v11 layout. This is the most
involved integration on the list; do it only if you actually need many concurrent streams.
Fixed input shape (step 05 §1) matters most here.

## VMS platforms (Milestone, Genetec, Avigilon)

These consume analytics through vendor SDKs rather than by loading an ONNX file. The
practical pattern:

```
VMS ──RTSP──► your inference service ──events/metadata──► VMS
```

Wrap the runtime from step 06 in a small service that pulls RTSP with OpenCV or FFmpeg and
pushes detections back via the vendor's SDK or a generic protocol (ONVIF metadata, MQTT,
HTTP webhook). Milestone's MIP SDK and Genetec's SDK are both .NET-first; the usual
approach is a thin C# bridge calling your HTTP endpoint rather than reimplementing
inference.

Budget real time for this: the model is the easy part, the VMS SDK is not. Prove value
with Frigate or a standalone service first.

## Browser (ONNX Runtime Web)

Runs client-side, no server:

```html
<script src="https://cdn.jsdelivr.net/npm/onnxruntime-web/dist/ort.min.js"></script>
<script>
  const sess = await ort.InferenceSession.create("model.onnx", {
    executionProviders: ["wasm"],
  });
</script>
```

Viable for a demo with `yolo11n`. Expect a few FPS on WASM; WebGPU is faster where
available.

## Home Assistant, MQTT, webhooks

Via Frigate, detections arrive on MQTT for free. Standalone, publish them yourself:

```python
mqtt.publish("cameras/warehouse/detections",
             json.dumps({"ts": ts, "objects": [d.as_dict() for d in dets]}))
```

## Choosing a target

| Situation | Start with |
|---|---|
| Home cameras, recording, notifications | **Frigate** |
| Intel box, custom app | **OpenVINO + Python** |
| Many RTSP streams, NVIDIA hardware | **DeepStream** |
| Existing enterprise VMS | **RTSP-in / events-out service** |
| Demo or sharing | **Browser (ORT Web)** |
| Batch analysis of stored media | **Plain Python, step 06's runtime** |
