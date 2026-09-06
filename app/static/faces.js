// --- models status ------------------------------------------------------------

const modelsStatus = document.getElementById("models-status");
const downloadBtn = document.getElementById("download-btn");
const downloadError = document.getElementById("download-error");

async function refreshModelsStatus() {
  const body = await (await fetch("/api/faces/models/status")).json();
  if (body.ready) {
    modelsStatus.textContent = "SCRFD + ArcFace ready.";
    downloadBtn.hidden = true;
  } else {
    modelsStatus.textContent = "SCRFD + ArcFace not downloaded yet — required before adding " +
      "identities or testing.";
    downloadBtn.hidden = false;
  }
  return body.ready;
}

downloadBtn.addEventListener("click", async () => {
  downloadBtn.disabled = true;
  downloadError.hidden = true;
  modelsStatus.textContent = "downloading (~190MB, this can take a minute)…";
  try {
    const res = await fetch("/api/faces/models/download", { method: "POST" });
    if (!res.ok) throw new Error((await res.json()).detail || "download failed");
  } catch (err) {
    downloadError.hidden = false;
    downloadError.textContent = err.message;
  }
  downloadBtn.disabled = false;
  await refreshModelsStatus();
});

// --- identities -----------------------------------------------------------------

const identityForm = document.getElementById("identity-form");
const identityRows = document.getElementById("identity-rows");
const detailSection = document.getElementById("identity-detail");
const detailName = document.getElementById("detail-name");
const photoFiles = document.getElementById("photo-files");
const uploadBtn = document.getElementById("upload-btn");
const uploadReport = document.getElementById("upload-report");
const photoGrid = document.getElementById("photo-grid");

let currentIdentityId = null;

async function loadIdentities() {
  const identities = await (await fetch("/api/faces/identities")).json();
  identityRows.innerHTML = "";
  for (const identity of identities) {
    const tr = document.createElement("tr");
    const nameTd = document.createElement("td");
    const nameLink = document.createElement("a");
    nameLink.href = "#";
    nameLink.textContent = identity.name;
    nameLink.addEventListener("click", (e) => { e.preventDefault(); openDetail(identity); });
    nameTd.appendChild(nameLink);

    const notesTd = document.createElement("td");
    notesTd.textContent = identity.notes || "";
    const countTd = document.createElement("td");
    countTd.textContent = identity.embedding_count;

    const actionsTd = document.createElement("td");
    const delBtn = document.createElement("button");
    delBtn.type = "button";
    delBtn.textContent = "delete";
    delBtn.addEventListener("click", async () => {
      if (!confirm(`Delete identity "${identity.name}" and all its embeddings?`)) return;
      await fetch(`/api/faces/identities/${identity.id}`, { method: "DELETE" });
      if (currentIdentityId === identity.id) detailSection.hidden = true;
      await loadIdentities();
    });
    actionsTd.appendChild(delBtn);

    tr.append(nameTd, notesTd, countTd, actionsTd);
    identityRows.appendChild(tr);
  }
}

identityForm.addEventListener("submit", async (e) => {
  e.preventDefault();
  const name = document.getElementById("identity-name").value.trim();
  const notes = document.getElementById("identity-notes").value.trim();
  if (!name) return;
  const res = await fetch("/api/faces/identities", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ name, notes: notes || null }),
  });
  if (res.ok) {
    identityForm.reset();
    await loadIdentities();
  }
});

async function openDetail(identity) {
  currentIdentityId = identity.id;
  detailName.textContent = identity.name;
  detailSection.hidden = false;
  uploadReport.hidden = true;
  await loadPhotos();
  detailSection.scrollIntoView({ behavior: "smooth" });
}

async function loadPhotos() {
  const body = await (await fetch(`/api/faces/identities/${currentIdentityId}`)).json();
  photoGrid.innerHTML = "";
  for (const emb of body.embeddings) {
    const fig = document.createElement("figure");
    const img = document.createElement("img");
    img.src = `/api/faces/embeddings/${emb.id}/photo`;
    const del = document.createElement("button");
    del.type = "button";
    del.textContent = "✕";
    del.addEventListener("click", async () => {
      await fetch(`/api/faces/embeddings/${emb.id}`, { method: "DELETE" });
      await loadPhotos();
      await loadIdentities();
    });
    fig.append(img, del);
    photoGrid.appendChild(fig);
  }
}

uploadBtn.addEventListener("click", async () => {
  if (!currentIdentityId || !photoFiles.files.length) return;
  uploadBtn.disabled = true;
  const form = new FormData();
  for (const f of photoFiles.files) form.append("files", f);
  const res = await fetch(`/api/faces/identities/${currentIdentityId}/photos`, {
    method: "POST", body: form,
  });
  uploadBtn.disabled = false;
  const body = await res.json();
  uploadReport.hidden = false;
  uploadReport.textContent = body.items
    .map((i) => `${i.ok ? "✓" : "✗"} ${i.filename}: ${i.detail}`)
    .join("\n");
  photoFiles.value = "";
  await loadPhotos();
  await loadIdentities();
});

// --- test controls ----------------------------------------------------------------

const detConfSlider = document.getElementById("det-conf-slider");
const detConfValue = document.getElementById("det-conf-value");
const matchSlider = document.getElementById("match-slider");
const matchValue = document.getElementById("match-value");

detConfSlider.addEventListener("input", () => {
  detConfValue.textContent = (+detConfSlider.value).toFixed(2);
});
matchSlider.addEventListener("input", () => {
  matchValue.textContent = (+matchSlider.value).toFixed(2);
});

for (const btn of document.querySelectorAll(".tab-btn")) {
  btn.addEventListener("click", () => {
    for (const b of document.querySelectorAll(".tab-btn")) b.classList.remove("active");
    for (const p of document.querySelectorAll(".tab-panel")) p.hidden = true;
    btn.classList.add("active");
    document.getElementById(`tab-${btn.dataset.tab}`).hidden = false;
    if (btn.dataset.tab !== "webcam") stopWebcam();
  });
}

function drawFaces(ctx, faces) {
  for (const f of faces) {
    const [x1, y1, x2, y2] = f.bbox;
    const color = f.identity_id != null ? "#4285f4" : "#999";
    ctx.strokeStyle = "#000";
    ctx.lineWidth = 4;
    ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);
    ctx.strokeStyle = color;
    ctx.lineWidth = 2;
    ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);

    const label = `${f.name} ${f.similarity.toFixed(2)}`;
    ctx.font = "13px system-ui, sans-serif";
    const tw = ctx.measureText(label).width;
    ctx.fillStyle = color;
    ctx.fillRect(x1, Math.max(0, y1 - 18), tw + 8, 18);
    ctx.fillStyle = "#fff";
    ctx.fillText(label, x1 + 4, Math.max(12, y1 - 5));
  }
}

// --- image tab ---------------------------------------------------------------------

const imageFile = document.getElementById("image-file");
const imageError = document.getElementById("image-error");
const imageCanvas = document.getElementById("image-canvas");
const imageCtx = imageCanvas.getContext("2d");

function showImageError(message) {
  imageError.textContent = message;
  imageError.hidden = !message;
}

imageFile.addEventListener("change", async () => {
  if (!imageFile.files.length) return;
  showImageError("");
  const form = new FormData();
  form.append("file", imageFile.files[0]);
  const params = new URLSearchParams({ det_conf: detConfSlider.value, match_threshold: matchSlider.value });
  const res = await fetch(`/api/faces/test/image?${params}`, { method: "POST", body: form });
  if (!res.ok) return showImageError((await res.json()).detail || "prediction failed");
  const body = await res.json();

  const bitmap = new window.Image();
  bitmap.onload = () => {
    imageCanvas.width = body.width;
    imageCanvas.height = body.height;
    imageCtx.drawImage(bitmap, 0, 0);
  };
  bitmap.src = body.image;
});

// --- webcam tab ----------------------------------------------------------------------

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
let webcamLastFaces = [];

function setWebcamStatus(text, cls) {
  webcamStatus.textContent = text;
  webcamStatus.className = cls ? `hint status-${cls}` : "hint";
}

async function startWebcam() {
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
  webcamSocket = new WebSocket(`${proto}://${location.host}/api/faces/ws/predict`);
  webcamSocket.binaryType = "arraybuffer";

  webcamSocket.onopen = () => {
    webcamRunning = true;
    webcamStop.disabled = false;
    setWebcamStatus("live", "done");
    sendThreshold();
    requestAnimationFrame(pump);
  };
  webcamSocket.onmessage = (e) => {
    webcamLastFaces = JSON.parse(e.data);
    webcamInFlight = false;
  };
  webcamSocket.onerror = () => setWebcamStatus("connection error", "disconnected");
  webcamSocket.onclose = () => {
    webcamRunning = false;
    webcamInFlight = false;
    setWebcamStatus("disconnected", "disconnected");
    stopWebcam();
  };
}

function sendThreshold() {
  if (webcamSocket && webcamSocket.readyState === WebSocket.OPEN) {
    webcamSocket.send(JSON.stringify({ det_conf: +detConfSlider.value, match_threshold: +matchSlider.value }));
  }
}
detConfSlider.addEventListener("change", sendThreshold);
matchSlider.addEventListener("change", sendThreshold);

function pump() {
  if (!webcamRunning) return;
  webcamCtx.drawImage(webcamVideo, 0, 0, webcamCanvas.width, webcamCanvas.height);
  drawFaces(webcamCtx, webcamLastFaces);

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
  webcamLastFaces = [];
  webcamCtx.clearRect(0, 0, webcamCanvas.width, webcamCanvas.height);
}

webcamStart.addEventListener("click", startWebcam);
webcamStop.addEventListener("click", () => { stopWebcam(); setWebcamStatus("stopped"); });

// --- boot ---------------------------------------------------------------------

(async function init() {
  await refreshModelsStatus();
  await loadIdentities();
})();
