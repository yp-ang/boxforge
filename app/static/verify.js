const projectId = Number(document.body.dataset.projectId);

const modelSelect = document.getElementById("model-select");
const modelWarning = document.getElementById("model-warning");
const confSlider = document.getElementById("conf-slider");
const confValue = document.getElementById("conf-value");
const iouSlider = document.getElementById("iou-slider");
const iouValue = document.getElementById("iou-value");
const detCount = document.getElementById("det-count");

// --- shared: per-class colour + box drawing (step 06 §5) --------------------

// Fixed palette, hashed by class index, so the same class is the same colour across
// images — matches app/services/runtime.py's PALETTE for visual consistency.
const PALETTE = [
  "#4285f4", "#db4437", "#f4b400", "#0f9d58", "#ab47bc", "#ff7043",
  "#00acc1", "#9ccc65", "#f06292", "#795548", "#9e9e9e", "#3949ab",
];
const colorForClass = (cls) => PALETTE[cls % PALETTE.length];

function drawBoxes(ctx, dets, conf) {
  for (const d of dets) {
    if (d.score < conf) continue;
    const [x1, y1, x2, y2] = d.xyxy;
    const color = colorForClass(d.cls);
    const w = x2 - x1, h = y2 - y1;

    // 1px dark outline beneath the coloured stroke so boxes stay visible against a
    // white wall and a dark floor alike.
    ctx.strokeStyle = "#000";
    ctx.lineWidth = 4;
    ctx.strokeRect(x1, y1, w, h);
    ctx.strokeStyle = color;
    ctx.lineWidth = 2;
    ctx.strokeRect(x1, y1, w, h);

    const label = `${d.label} ${d.score.toFixed(2)}`;
    ctx.font = "13px system-ui, sans-serif";
    const tw = ctx.measureText(label).width;
    ctx.fillStyle = color;
    ctx.fillRect(x1, Math.max(0, y1 - 18), tw + 8, 18);
    ctx.fillStyle = "#fff";
    ctx.fillText(label, x1 + 4, Math.max(12, y1 - 5));
  }
}

// --- model selection ----------------------------------------------------------

let currentModelId = null;

async function loadModels() {
  const models = await (await fetch(`/api/projects/${projectId}/models`)).json();
  const usable = models.filter((m) => m.onnx_path);

  modelSelect.innerHTML = "";
  if (!usable.length) {
    modelWarning.hidden = false;
    modelWarning.textContent = "no exported model yet — export to ONNX on the train page first";
    return;
  }
  modelWarning.hidden = true;
  for (const m of usable) {
    const label = `${m.name}${m.is_active ? " (active)" : ""}${
      m.parity_status === "failed" ? " — parity FAILED" : ""
    }`;
    modelSelect.appendChild(new Option(label, m.id));
  }
  const active = usable.find((m) => m.is_active) || usable[0];
  modelSelect.value = active.id;
  currentModelId = active.id;
}

modelSelect.addEventListener("change", () => {
  currentModelId = Number(modelSelect.value);
});

// --- tabs -----------------------------------------------------------------------

for (const btn of document.querySelectorAll(".tab-btn")) {
  btn.addEventListener("click", () => {
    for (const b of document.querySelectorAll(".tab-btn")) b.classList.remove("active");
    for (const p of document.querySelectorAll(".tab-panel")) p.hidden = true;
    btn.classList.add("active");
    document.getElementById(`tab-${btn.dataset.tab}`).hidden = false;
    if (btn.dataset.tab !== "webcam") stopWebcam();
  });
}

// --- confidence / IoU controls ---------------------------------------------------

let allDets = [];   // full, low-threshold detection list for the current image

function renderCount(dets, conf) {
  const shown = dets.filter((d) => d.score >= conf).length;
  detCount.textContent = `${shown} detection${shown === 1 ? "" : "s"} @ conf ${conf.toFixed(2)}`;
}

confSlider.addEventListener("input", () => {
  const conf = +confSlider.value;
  confValue.textContent = conf.toFixed(2);
  redrawImage();
  renderCount(allDets, conf);
});

iouSlider.addEventListener("input", () => {
  iouValue.textContent = (+iouSlider.value).toFixed(2);
  // NMS already ran server-side (step 06 §4 separates the two knobs, but merging boxes
  // is not a client-side operation) — re-run once, cheaply, for a single image.
  if (currentImageFile) predictImage(currentImageFile);
});

// --- image tab --------------------------------------------------------------------

const imageFile = document.getElementById("image-file");
const imageError = document.getElementById("image-error");
const imageCanvas = document.getElementById("image-canvas");
const imageCtx = imageCanvas.getContext("2d");
const imageJson = document.getElementById("image-json");

let currentImageFile = null;
let currentImageBitmap = null;

function showImageError(message) {
  imageError.textContent = message;
  imageError.hidden = !message;
}

async function predictImage(file) {
  if (!currentModelId) return showImageError("no exported model selected");
  showImageError("");
  const form = new FormData();
  form.append("file", file);
  const res = await fetch(
    `/api/models/${currentModelId}/predict/image?iou=${iouSlider.value}`,
    { method: "POST", body: form }
  );
  if (!res.ok) {
    const body = await res.json().catch(() => ({}));
    return showImageError(body.detail || "prediction failed");
  }
  const body = await res.json();
  allDets = body.detections;
  imageJson.textContent = JSON.stringify(body.detections, null, 2);

  const bitmap = new window.Image();
  bitmap.onload = () => {
    currentImageBitmap = bitmap;
    imageCanvas.width = body.width;
    imageCanvas.height = body.height;
    redrawImage();
    renderCount(allDets, +confSlider.value);
  };
  bitmap.src = body.image;
}

function redrawImage() {
  if (!currentImageBitmap) return;
  imageCtx.drawImage(currentImageBitmap, 0, 0);
  drawBoxes(imageCtx, allDets, +confSlider.value);
}

imageFile.addEventListener("change", () => {
  if (!imageFile.files.length) return;
  currentImageFile = imageFile.files[0];
  predictImage(currentImageFile);
});

// --- video tab --------------------------------------------------------------------

const videoForm = document.getElementById("video-form");
const videoFileInput = document.getElementById("video-file");
const videoStride = document.getElementById("video-stride");
const videoError = document.getElementById("video-error");
const videoJobSection = document.getElementById("video-job");
const videoJobId = document.getElementById("video-job-id");
const videoJobStatus = document.getElementById("video-job-status");
const videoLog = document.getElementById("video-log");
const videoResult = document.getElementById("video-result");

function showVideoError(message) {
  videoError.textContent = message;
  videoError.hidden = !message;
}

videoForm.addEventListener("submit", async (e) => {
  e.preventDefault();
  showVideoError("");
  if (!currentModelId) return showVideoError("no exported model selected");
  if (!videoFileInput.files.length) return showVideoError("choose a video file");

  const form = new FormData();
  form.append("file", videoFileInput.files[0]);
  const params = new URLSearchParams({
    stride: videoStride.value, conf: confSlider.value, iou: iouSlider.value,
  });
  const res = await fetch(
    `/api/models/${currentModelId}/predict/video?${params}`, { method: "POST", body: form }
  );
  if (!res.ok) {
    const body = await res.json().catch(() => ({}));
    return showVideoError(body.detail || "could not start prediction");
  }
  const job = await res.json();
  attachToVideoJob(job.id);
});

function attachToVideoJob(jobId) {
  videoJobSection.hidden = false;
  videoJobId.textContent = `#${jobId}`;
  videoJobStatus.textContent = "running";
  videoJobStatus.className = "status-running";
  videoLog.textContent = "";
  videoResult.hidden = true;

  const source = new EventSource(`/api/jobs/${jobId}/logs`);
  source.onmessage = (e) => {
    const atBottom = videoLog.scrollHeight - videoLog.scrollTop - videoLog.clientHeight < 40;
    videoLog.textContent += e.data + "\n";
    if (atBottom) videoLog.scrollTop = videoLog.scrollHeight;
  };
  source.addEventListener("done", async () => {
    source.close();
    const job = await (await fetch(`/api/jobs/${jobId}`)).json();
    videoJobStatus.textContent = job.status;
    videoJobStatus.className = `status-${job.status}`;
    if (job.status === "done") {
      videoResult.src = `/api/jobs/${jobId}/video`;
      videoResult.hidden = false;
    } else if (job.result_json) {
      const parsed = JSON.parse(job.result_json);
      if (parsed.error) showVideoError(`job ${jobId} failed: ${parsed.error}`);
    }
  });
}

// --- webcam tab -------------------------------------------------------------------

const webcamStart = document.getElementById("webcam-start");
const webcamStop = document.getElementById("webcam-stop");
const webcamStatus = document.getElementById("webcam-status");
const webcamVideo = document.getElementById("webcam-video");
const webcamCanvas = document.getElementById("webcam-canvas");
const webcamCtx = webcamCanvas.getContext("2d");

let webcamStream = null;
let webcamSocket = null;
let webcamRunning = false;
let webcamInFlight = false;
let webcamLastDets = [];

function setWebcamStatus(text, cls) {
  webcamStatus.textContent = text;
  webcamStatus.className = cls ? `hint status-${cls}` : "hint";
}

async function startWebcam() {
  if (!currentModelId) return setWebcamStatus("no exported model selected", "disconnected");
  if (!window.isSecureContext) {
    return setWebcamStatus("requires a secure context (localhost is fine)", "disconnected");
  }

  webcamStart.disabled = true;
  try {
    webcamStream = await navigator.mediaDevices.getUserMedia({ video: { width: 640 } });
  } catch (err) {
    webcamStart.disabled = false;
    return setWebcamStatus(`camera error: ${err.message}`, "disconnected");
  }
  webcamVideo.srcObject = webcamStream;
  await webcamVideo.play();

  const w = 640;
  const h = Math.round(640 * webcamVideo.videoHeight / webcamVideo.videoWidth) || 480;
  webcamCanvas.width = w;
  webcamCanvas.height = h;

  const proto = location.protocol === "https:" ? "wss" : "ws";
  webcamSocket = new WebSocket(`${proto}://${location.host}/ws/predict/${currentModelId}`);
  webcamSocket.binaryType = "arraybuffer";

  webcamSocket.onopen = () => {
    webcamRunning = true;
    webcamStop.disabled = false;
    setWebcamStatus("live", "done");
    sendThreshold();
    requestAnimationFrame(pump);
  };
  webcamSocket.onmessage = (e) => {
    webcamLastDets = JSON.parse(e.data);
    webcamInFlight = false;
  };
  webcamSocket.onerror = () => setWebcamStatus("connection error", "disconnected");
  webcamSocket.onclose = () => {
    // The important part per step 06 §3: a dead connection stops the pump loop and
    // resets the UI rather than leaving the page waiting on a response that will
    // never arrive.
    webcamRunning = false;
    webcamInFlight = false;
    setWebcamStatus("disconnected", "disconnected");
    stopWebcam();
  };
}

function sendThreshold() {
  if (webcamSocket && webcamSocket.readyState === WebSocket.OPEN) {
    webcamSocket.send(JSON.stringify({ conf: +confSlider.value, iou: +iouSlider.value }));
  }
}
confSlider.addEventListener("change", sendThreshold);
iouSlider.addEventListener("change", sendThreshold);

function pump() {
  if (!webcamRunning) return;
  webcamCtx.drawImage(webcamVideo, 0, 0, webcamCanvas.width, webcamCanvas.height);
  drawBoxes(webcamCtx, webcamLastDets, 0);   // server already filtered by confidence

  if (webcamSocket.readyState === WebSocket.OPEN && !webcamInFlight) {
    webcamInFlight = true;
    webcamCanvas.toBlob(
      (blob) => blob && webcamSocket.readyState === WebSocket.OPEN
        && blob.arrayBuffer().then((buf) => webcamSocket.send(buf)),
      "image/jpeg", 0.8
    );
  }
  requestAnimationFrame(pump);
}

function stopWebcam() {
  webcamRunning = false;
  webcamInFlight = false;
  if (webcamSocket) { webcamSocket.onclose = null; webcamSocket.close(); webcamSocket = null; }
  if (webcamStream) { for (const t of webcamStream.getTracks()) t.stop(); webcamStream = null; }
  webcamStart.disabled = false;
  webcamStop.disabled = true;
  webcamLastDets = [];
  webcamCtx.clearRect(0, 0, webcamCanvas.width, webcamCanvas.height);
}

webcamStart.addEventListener("click", startWebcam);
webcamStop.addEventListener("click", () => { stopWebcam(); setWebcamStatus("stopped"); });

// --- server-side webcam capability check (nice-to-have; step 06 §3) --------------

document.getElementById("server-webcam-check").addEventListener("click", async () => {
  const resultEl = document.getElementById("server-webcam-result");
  resultEl.textContent = "checking…";
  const body = await (await fetch("/api/capabilities/webcam")).json();
  resultEl.textContent = body.server_webcam ? "available" : body.note;
});

// --- boot ---------------------------------------------------------------------

(async function init() {
  await loadModels();
})();
