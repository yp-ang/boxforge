# Step 06 — Verify playground

**Goal:** throw an image, a video, or your webcam at a trained model and see boxes with
confidences, with a threshold slider you can drag while it runs.

**Depends on:** step 05.

**Load the ONNX, not the `.pt`.** This module exercises the artifact you hand to other
software. If it works here, it works there.

## 1. Runtime

```python
# app/services/runtime.py
class OnnxDetector:
    def __init__(self, model_dir: Path):
        meta = json.loads((model_dir / "metadata.json").read_text())
        self.classes = {int(k): v for k, v in
                        json.loads((model_dir / "classes.json").read_text()).items()}
        self.imgsz = meta["imgsz"]
        self.session = ort.InferenceSession(
            str(model_dir / "model.onnx"),
            providers=self._providers(),
        )
        self.inp = self.session.get_inputs()[0].name

    @staticmethod
    def _providers():
        avail = ort.get_available_providers()
        # CoreML on Apple Silicon, CUDA on NVIDIA, CPU always last as fallback
        return [p for p in ("CUDAExecutionProvider", "CoreMLExecutionProvider")
                if p in avail] + ["CPUExecutionProvider"]

    def predict(self, bgr: np.ndarray, conf: float, iou: float = 0.45) -> list[Det]:
        tensor, ratio, pad = letterbox(bgr, self.imgsz)
        raw = self.session.run(None, {self.inp: tensor})[0]
        dets = decode(raw, conf, iou, len(self.classes))
        return unletterbox(dets, ratio, pad)     # back to original image pixels
```

Cache detector instances by model id — constructing an `InferenceSession` takes a second
or two and you do not want that per frame.

## 2. Letterbox — get this exactly right

Preprocessing mismatch is the single most common source of "the model detects nothing".
Resize preserving aspect ratio, pad to square with 114 grey, convert BGR→RGB, scale to
0–1, transpose to NCHW:

```python
def letterbox(bgr, size=640, pad_value=114):
    h, w = bgr.shape[:2]
    r = min(size / h, size / w)
    nh, nw = round(h * r), round(w * r)
    resized = cv2.resize(bgr, (nw, nh), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((size, size, 3), pad_value, np.uint8)
    top, left = (size - nh) // 2, (size - nw) // 2
    canvas[top:top+nh, left:left+nw] = resized
    rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
    x = rgb.astype(np.float32) / 255.0
    return x.transpose(2, 0, 1)[None], r, (left, top)

def unletterbox(dets, r, pad):
    left, top = pad
    for d in dets:
        d.xyxy = ((d.xyxy - np.array([left, top, left, top])) / r)
    return dets
```

Centred padding, not top-left. Ultralytics centres, and if you pad differently your boxes
land offset by half the padding — subtly wrong in a way that looks like a bad model.

## 3. Three input modes

### Image upload
`POST /api/models/{id}/predict/image` (multipart) → annotated JPEG + JSON detections.
Return both: the picture for your eyes, the JSON for debugging.

### Video upload
`POST /api/models/{id}/predict/video` → a job (reuse step 04's runner). Write an annotated
MP4 and a per-frame JSON. Add a `stride` parameter — inferring every 3rd frame is 3× faster
and visually identical for most review purposes.

```python
writer = cv2.VideoWriter(out, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
```

`mp4v` is what OpenCV ships with everywhere. It is not the most efficient codec; it is the
one that will not make you debug an FFmpeg build.

### Webcam — browser capture
Per ARCHITECTURE §5.2, capture in the **browser**, not the server. Works in conda and in
Docker, on macOS and Linux, with no device mapping.

```js
const stream = await navigator.mediaDevices.getUserMedia({ video: { width: 1280 } });
video.srcObject = stream;

const ws = new WebSocket(`ws://${location.host}/ws/predict/${modelId}`);
let inFlight = false;

async function pump() {
  if (ws.readyState === 1 && !inFlight) {
    inFlight = true;
    ctx.drawImage(video, 0, 0, 640, 640 * video.videoHeight / video.videoWidth);
    const blob = await new Promise(r => cap.toBlob(r, "image/jpeg", 0.8));
    ws.send(await blob.arrayBuffer());
  }
  requestAnimationFrame(pump);
}
ws.onmessage = e => { inFlight = false; drawBoxes(JSON.parse(e.data)); };
```

The `inFlight` latch is the important part: send the next frame only after the previous
result arrives. Without it, a slow model builds an unbounded queue and you end up watching
detections from ten seconds ago. This backpressure is one boolean and it is the difference
between "live" and "broken".

Server side, note that `getUserMedia` requires a secure context — `localhost` counts, so
local dev is fine. If you ever expose this on a LAN IP you will need HTTPS.

```python
@app.websocket("/ws/predict/{model_id}")
async def ws_predict(ws: WebSocket, model_id: int):
    await ws.accept()
    det = get_detector(model_id)
    try:
        while True:
            data = await ws.receive_bytes()
            frame = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
            conf = current_threshold(ws)          # updated by control messages
            dets = await run_in_threadpool(det.predict, frame, conf)
            await ws.send_json([d.as_dict() for d in dets])
    except WebSocketDisconnect:
        pass
```

`run_in_threadpool` matters — onnxruntime is blocking, and calling it directly in the
async handler stalls the entire event loop including the UI.

### Server-side webcam (`cv2.VideoCapture(0)`)
Add it, gated on a capability check, for future Linux deployment. Expect it to fail in
Docker on macOS and say so in the UI rather than spinning.

## 4. Threshold control

The slider must change results **without re-running inference** wherever possible. For a
single image, run once at `conf=0.01`, keep the full detection list client-side, and filter
in JavaScript as the slider moves. Instant feedback, and it makes choosing a threshold an
actual visual decision rather than a guess-and-wait loop.

```js
slider.oninput = () => {
  const t = +slider.value;
  render(allDets.filter(d => d.score >= t));
  countEl.textContent = `${allDets.filter(d => d.score >= t).length} detections @ ${t}`;
};
```

For live video, send the threshold as a WebSocket control message so the server filters
before sending — no point shipping 300 low-confidence boxes per frame over the socket.

Separate **confidence** from **NMS IoU** in the UI. They are different knobs and conflating
them is a classic confusion: confidence decides *whether* a detection counts, IoU decides
whether two overlapping detections are the same object.

## 5. Drawing

Consistent per-class colours (hash the class index into a fixed palette so the same class
is the same colour across images), 2px boxes, label chip reading `person 0.87`. Draw box
outlines with a 1px dark outline beneath the coloured stroke so they stay visible against
both a white wall and a dark floor.

## Acceptance criteria

- Upload an image → boxes with confidences, sensible positions.
- Drag the threshold from 0.05 → 0.9 → boxes disappear progressively, no server round-trip.
- Upload a 30s video → annotated MP4 plays back with boxes.
- Webcam runs at ≥5 FPS on CPU with `yolo11n` at 640, and the video does not lag behind.
- Kill the network mid-stream → UI shows disconnected, no hang.
- Confidence and IoU are separate controls.
