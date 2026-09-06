const projectId = Number(document.body.dataset.projectId);

const canvas = document.getElementById("canvas");
const ctx = canvas.getContext("2d");
const canvasPane = document.querySelector(".canvas-pane");
const labelListEl = document.getElementById("label-list");
const boxListEl = document.getElementById("box-list");
const boxCountEl = document.getElementById("box-count");
const positionEl = document.getElementById("position");
const saveIndicatorEl = document.getElementById("save-indicator");
const reviewedIndicatorEl = document.getElementById("reviewed-indicator");
const prevBtn = document.getElementById("prev-btn");
const nextBtn = document.getElementById("next-btn");
const skipBtn = document.getElementById("skip-btn");
const deleteBtn = document.getElementById("delete-btn");
const orderSelect = document.getElementById("order-select");

let labels = [];               // [{id, name, color, class_index}]
let activeLabelId = null;      // persists across images

let current = null;            // {image, annotations, index, total}
let boxes = [];                // working set for current image
let selectedIndex = -1;
let dirty = false;
let undoStack = [];
let img = new window.Image();  // current bitmap
let imgLoaded = false;

let scale = 1, offsetX = 0, offsetY = 0;
let zoomFactor = 1;
let panning = false, spaceHeld = false;
let panStart = null;
let dragMode = null;           // "draw" | "move" | "resize" | null
let dragHandle = null;         // "nw" | "ne" | "sw" | "se" when resizing
let dragStartNorm = null;
let dragOrigBox = null;

let nextCache = null;          // {key, promise|data} — one image ahead

// ---------- data loading ----------

async function loadLabels() {
  const res = await fetch(`/api/projects/${projectId}/labels`);
  labels = await res.json();
  if (!activeLabelId && labels.length) activeLabelId = labels[0].id;
  renderLabelList();
}

function labelById(id) {
  return labels.find((l) => l.id === id);
}

async function fetchNext(after, direction) {
  const params = new URLSearchParams({ direction, order: orderSelect.value });
  if (after !== null && after !== undefined) params.set("after", after);
  const res = await fetch(`/api/projects/${projectId}/next?${params}`);
  if (!res.ok) return null;
  return res.json();
}

function cacheKeyFor(afterId) {
  return `${afterId}:next:${orderSelect.value}`;
}

function prefetchNext() {
  if (!current) return;
  const key = cacheKeyFor(current.image.id);
  const promise = fetchNext(current.image.id, "next").then((data) => {
    if (data) {
      const pre = new window.Image();
      pre.src = `/api/images/${data.image.id}/file`;
    }
    return data;
  });
  nextCache = { key, promise };
}

async function goTo(direction) {
  await saveCurrentIfDirty();
  let data;
  if (direction === "next" && nextCache && current &&
      nextCache.key === cacheKeyFor(current.image.id)) {
    data = await nextCache.promise;
  } else {
    data = await fetchNext(current ? current.image.id : null, direction);
  }
  nextCache = null;
  if (!data) {
    flashSave(direction === "next" ? "no more images" : "start of project");
    return;
  }
  await loadCurrent(data);
}

async function loadCurrent(data) {
  current = data;
  boxes = data.annotations.map((a) => ({ ...a, dirty: false }));
  selectedIndex = -1;
  undoStack = [];
  dirty = false;

  imgLoaded = false;
  img = new window.Image();
  img.onload = () => {
    imgLoaded = true;
    fitToContainer();
    draw();
  };
  img.src = `/api/images/${data.image.id}/file`;

  positionEl.textContent = `${data.index} / ${data.total}`;
  reviewedIndicatorEl.textContent = data.image.reviewed_at ? "yes" : "no";
  renderBoxList();
  draw();
  prefetchNext();
}

function annotationsPayload() {
  return {
    boxes: boxes.map((b) => ({
      label_id: b.label_id, x1: b.x1, y1: b.y1, x2: b.x2, y2: b.y2,
    })),
  };
}

async function saveCurrentIfDirty() {
  if (!current || !dirty) return;
  await fetch(`/api/images/${current.image.id}/annotations`, {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(annotationsPayload()),
  });
  dirty = false;
  flashSave("saved");
}

window.addEventListener("beforeunload", () => {
  if (!current || !dirty) return;
  fetch(`/api/images/${current.image.id}/annotations`, {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(annotationsPayload()),
    keepalive: true,
  });
});

function flashSave(msg) {
  saveIndicatorEl.textContent = msg;
  setTimeout(() => { if (saveIndicatorEl.textContent === msg) saveIndicatorEl.textContent = ""; }, 1200);
}

async function skipCurrent() {
  if (!current) return;
  await fetch(`/api/images/${current.image.id}/skip`, { method: "POST" });
  dirty = false;
  await goTo("next");
}

async function markReviewed() {
  if (!current) return;
  const res = await fetch(`/api/images/${current.image.id}/review`, { method: "POST" });
  const img2 = await res.json();
  current.image = img2;
  reviewedIndicatorEl.textContent = "yes";
  flashSave("reviewed");
}

// ---------- geometry ----------

function imgNatural() {
  return { w: img.naturalWidth || 1, h: img.naturalHeight || 1 };
}

function toCanvas(nx, ny) {
  const { w, h } = imgNatural();
  return [nx * w * scale + offsetX, ny * h * scale + offsetY];
}

function toNorm(cx, cy) {
  const { w, h } = imgNatural();
  return [clamp01((cx - offsetX) / (w * scale)), clamp01((cy - offsetY) / (h * scale))];
}

function clamp01(v) { return Math.max(0, Math.min(1, v)); }

function fitToContainer() {
  const rect = canvasPane.getBoundingClientRect();
  const { w, h } = imgNatural();
  const base = Math.min(rect.width / w, rect.height / h);
  scale = base * zoomFactor;
  offsetX = (rect.width - w * scale) / 2;
  offsetY = (rect.height - h * scale) / 2;
  sizeCanvas(rect.width, rect.height);
}

function sizeCanvas(cssW, cssH) {
  const dpr = window.devicePixelRatio || 1;
  canvas.style.width = `${cssW}px`;
  canvas.style.height = `${cssH}px`;
  canvas.width = cssW * dpr;
  canvas.height = cssH * dpr;
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
}

window.addEventListener("resize", () => { if (imgLoaded) { fitToContainer(); draw(); } });

function commitBox(a, b, labelId) {
  const { w, h } = imgNatural();
  const [x1, x2] = [Math.min(a[0], b[0]), Math.max(a[0], b[0])];
  const [y1, y2] = [Math.min(a[1], b[1]), Math.max(a[1], b[1])];
  if ((x2 - x1) * w < 4 || (y2 - y1) * h < 4) return null;
  return { label_id: labelId, x1, y1, x2, y2, source: "human" };
}

function hitTest(px, py, box) {
  const [x1, y1] = toCanvas(box.x1, box.y1);
  const [x2, y2] = toCanvas(box.x2, box.y2);
  const T = 8;
  for (const [name, hx, hy] of [["nw", x1, y1], ["ne", x2, y1], ["sw", x1, y2], ["se", x2, y2]]) {
    if (Math.abs(px - hx) < T && Math.abs(py - hy) < T) return name;
  }
  if (px > x1 && px < x2 && py > y1 && py < y2) return "move";
  return null;
}

// ---------- rendering ----------

function draw() {
  const rect = canvasPane.getBoundingClientRect();
  ctx.clearRect(0, 0, rect.width, rect.height);
  if (!imgLoaded) return;

  const { w, h } = imgNatural();
  ctx.drawImage(img, offsetX, offsetY, w * scale, h * scale);

  boxes.forEach((box, i) => drawBox(box, i === selectedIndex));

  if (selectedIndex >= 0) drawHandles(boxes[selectedIndex]);
}

function drawBox(box, selected) {
  const label = labelById(box.label_id);
  const color = label ? label.color : "#ff3b30";
  const [x1, y1] = toCanvas(box.x1, box.y1);
  const [x2, y2] = toCanvas(box.x2, box.y2);

  ctx.fillStyle = hexAlpha(color, 0.12);
  ctx.fillRect(x1, y1, x2 - x1, y2 - y1);
  ctx.strokeStyle = color;
  ctx.lineWidth = 2;
  // Dashed = drafted by the model, not yet reviewed (step 07 §2). Saving the image
  // flips every box to source="human" server-side, so this is purely a "have I looked
  // at this yet" signal, not a permanent style.
  if (box.source === "model") ctx.setLineDash([5, 4]);
  ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);
  ctx.setLineDash([]);

  if (selected) {
    ctx.save();
    ctx.setLineDash([6, 4]);
    ctx.strokeStyle = "#fff";
    ctx.lineWidth = 1.5;
    ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);
    ctx.restore();
  }

  const text = label ? label.name : `#${box.label_id}`;
  ctx.font = "12px system-ui, sans-serif";
  const padding = 4;
  const metrics = ctx.measureText(text);
  const chipH = 16;
  const chipY = y1 - chipH >= 0 ? y1 - chipH : y1;
  ctx.fillStyle = color;
  ctx.fillRect(x1, chipY, metrics.width + padding * 2, chipH);
  ctx.fillStyle = "#fff";
  ctx.fillText(text, x1 + padding, chipY + chipH - 4);
}

function drawHandles(box) {
  const [x1, y1] = toCanvas(box.x1, box.y1);
  const [x2, y2] = toCanvas(box.x2, box.y2);
  ctx.fillStyle = "#fff";
  for (const [hx, hy] of [[x1, y1], [x2, y1], [x1, y2], [x2, y2]]) {
    ctx.fillRect(hx - 4, hy - 4, 8, 8);
  }
}

function hexAlpha(hex, alpha) {
  const m = hex.replace("#", "");
  const r = parseInt(m.substring(0, 2), 16) || 0;
  const g = parseInt(m.substring(2, 4), 16) || 0;
  const b = parseInt(m.substring(4, 6), 16) || 0;
  return `rgba(${r},${g},${b},${alpha})`;
}

// ---------- undo ----------

function snapshot() {
  undoStack.push(boxes.map((b) => ({ ...b })));
  if (undoStack.length > 100) undoStack.shift();
}

function undo() {
  if (!undoStack.length) return;
  boxes = undoStack.pop();
  selectedIndex = -1;
  dirty = true;
  renderBoxList();
  draw();
}

// ---------- mutations ----------

function addBox(box) {
  snapshot();
  boxes.push(box);
  selectedIndex = boxes.length - 1;
  dirty = true;
  renderBoxList();
  draw();
}

function deleteSelected() {
  if (selectedIndex < 0) return;
  snapshot();
  boxes.splice(selectedIndex, 1);
  selectedIndex = -1;
  dirty = true;
  renderBoxList();
  draw();
}

function retagSelected(labelId) {
  if (selectedIndex < 0) return;
  snapshot();
  boxes[selectedIndex] = { ...boxes[selectedIndex], label_id: labelId };
  dirty = true;
  renderBoxList();
  draw();
}

function selectBox(i) {
  selectedIndex = i;
  renderBoxList();
  draw();
}

// ---------- side panel ----------

function renderLabelList() {
  labelListEl.innerHTML = "";
  labels.forEach((label, i) => {
    const li = document.createElement("li");
    li.className = label.id === activeLabelId ? "active" : "";
    const swatch = document.createElement("span");
    swatch.className = "swatch";
    swatch.style.background = label.color;
    const text = document.createElement("span");
    text.textContent = label.name;
    const hint = document.createElement("span");
    hint.className = "key-hint";
    hint.textContent = i < 9 ? String(i + 1) : "";
    li.append(swatch, text, hint);
    li.addEventListener("click", () => setActiveLabel(label.id));
    labelListEl.appendChild(li);
  });
}

function setActiveLabel(labelId) {
  activeLabelId = labelId;
  renderLabelList();
  if (selectedIndex >= 0) retagSelected(labelId);
}

function renderBoxList() {
  boxListEl.innerHTML = "";
  boxCountEl.textContent = boxes.length;
  boxes.forEach((box, i) => {
    const label = labelById(box.label_id);
    const li = document.createElement("li");
    li.className = i === selectedIndex ? "selected" : "";
    const swatch = document.createElement("span");
    swatch.className = "swatch";
    swatch.style.background = label ? label.color : "#999";
    const text = document.createElement("span");
    text.textContent = (label ? label.name : `#${box.label_id}`) + (box.source === "model" ? " (draft)" : "");
    const del = document.createElement("button");
    del.className = "del";
    del.textContent = "✕";
    del.addEventListener("click", (e) => {
      e.stopPropagation();
      selectedIndex = i;
      deleteSelected();
    });
    li.append(swatch, text, del);
    li.addEventListener("click", () => selectBox(i));
    boxListEl.appendChild(li);
  });
}

// ---------- mouse ----------

function canvasPoint(e) {
  const rect = canvas.getBoundingClientRect();
  return [e.clientX - rect.left, e.clientY - rect.top];
}

canvas.addEventListener("mousedown", (e) => {
  if (!imgLoaded) return;
  const [px, py] = canvasPoint(e);

  if (spaceHeld) {
    panning = true;
    panStart = { x: e.clientX, y: e.clientY, offsetX, offsetY };
    return;
  }

  if (selectedIndex >= 0) {
    const handle = hitTest(px, py, boxes[selectedIndex]);
    if (handle && handle !== "move") {
      dragMode = "resize";
      dragHandle = handle;
      dragOrigBox = { ...boxes[selectedIndex] };
      return;
    }
  }

  for (let i = boxes.length - 1; i >= 0; i--) {
    if (hitTest(px, py, boxes[i]) === "move") {
      selectedIndex = i;
      dragMode = "move";
      dragOrigBox = { ...boxes[i] };
      dragStartNorm = toNorm(px, py);
      renderBoxList();
      draw();
      return;
    }
  }

  if (!activeLabelId) return;
  dragMode = "draw";
  dragStartNorm = toNorm(px, py);
  selectedIndex = -1;
  renderBoxList();
});

canvas.addEventListener("mousemove", (e) => {
  if (panning) {
    offsetX = panStart.offsetX + (e.clientX - panStart.x);
    offsetY = panStart.offsetY + (e.clientY - panStart.y);
    draw();
    return;
  }
  if (!dragMode) return;
  const [px, py] = canvasPoint(e);
  const norm = toNorm(px, py);

  if (dragMode === "draw") {
    draw();
    drawPreview(dragStartNorm, norm);
  } else if (dragMode === "move") {
    const dx = norm[0] - dragStartNorm[0];
    const dy = norm[1] - dragStartNorm[1];
    const w = dragOrigBox.x2 - dragOrigBox.x1;
    const h = dragOrigBox.y2 - dragOrigBox.y1;
    let x1 = clamp01(dragOrigBox.x1 + dx);
    let y1 = clamp01(dragOrigBox.y1 + dy);
    x1 = Math.min(x1, 1 - w);
    y1 = Math.min(y1, 1 - h);
    boxes[selectedIndex] = { ...boxes[selectedIndex], x1, y1, x2: x1 + w, y2: y1 + h };
    draw();
  } else if (dragMode === "resize") {
    const box = { ...boxes[selectedIndex] };
    if (dragHandle.includes("w")) box.x1 = Math.min(norm[0], box.x2 - 0.001);
    if (dragHandle.includes("e")) box.x2 = Math.max(norm[0], box.x1 + 0.001);
    if (dragHandle.includes("n")) box.y1 = Math.min(norm[1], box.y2 - 0.001);
    if (dragHandle.includes("s")) box.y2 = Math.max(norm[1], box.y1 + 0.001);
    boxes[selectedIndex] = box;
    draw();
  }
});

function drawPreview(a, b) {
  const [x1, y1] = toCanvas(Math.min(a[0], b[0]), Math.min(a[1], b[1]));
  const [x2, y2] = toCanvas(Math.max(a[0], b[0]), Math.max(a[1], b[1]));
  const color = labelById(activeLabelId)?.color || "#ff3b30";
  ctx.fillStyle = hexAlpha(color, 0.12);
  ctx.fillRect(x1, y1, x2 - x1, y2 - y1);
  ctx.strokeStyle = color;
  ctx.lineWidth = 2;
  ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);
}

window.addEventListener("mouseup", (e) => {
  if (panning) { panning = false; return; }
  if (!dragMode) return;
  const [px, py] = canvasPoint(e);
  const norm = toNorm(px, py);

  if (dragMode === "draw") {
    const box = commitBox(dragStartNorm, norm, activeLabelId);
    if (box) addBox(box);
  } else if (dragMode === "move" || dragMode === "resize") {
    const b = boxes[selectedIndex];
    const changed = !dragOrigBox || b.x1 !== dragOrigBox.x1 || b.y1 !== dragOrigBox.y1 ||
      b.x2 !== dragOrigBox.x2 || b.y2 !== dragOrigBox.y2;
    if (changed) {
      undoStack.push(withReplacedIndex(boxes, selectedIndex, dragOrigBox));
      dirty = true;
    }
    renderBoxList();
    draw();
  }

  dragMode = null;
  dragHandle = null;
  dragOrigBox = null;
  dragStartNorm = null;
});

function withReplacedIndex(arr, i, item) {
  const copy = arr.map((b) => ({ ...b }));
  copy[i] = { ...item };
  return copy;
}

// ---------- keyboard ----------

window.addEventListener("keydown", (e) => {
  if (e.code === "Space") { spaceHeld = true; e.preventDefault(); return; }

  if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === "z") {
    e.preventDefault();
    undo();
    return;
  }

  if (e.key >= "1" && e.key <= "9") {
    const label = labels[Number(e.key) - 1];
    if (label) setActiveLabel(label.id);
    return;
  }

  switch (e.key) {
    case "Backspace":
    case "Delete":
      e.preventDefault();
      deleteSelected();
      break;
    case "d":
    case "D":
    case "ArrowRight":
      e.preventDefault();
      goTo("next");
      break;
    case "a":
    case "A":
    case "ArrowLeft":
      e.preventDefault();
      goTo("prev");
      break;
    case "s":
    case "S":
      e.preventDefault();
      skipCurrent();
      break;
    case "r":
    case "R":
      e.preventDefault();
      markReviewed();
      break;
    case "Escape":
      selectedIndex = -1;
      renderBoxList();
      draw();
      break;
    case "+":
    case "=":
      zoomFactor = Math.min(zoomFactor * 1.2, 8);
      fitToContainer();
      draw();
      break;
    case "-":
    case "_":
      zoomFactor = Math.max(zoomFactor / 1.2, 0.2);
      fitToContainer();
      draw();
      break;
  }
});

window.addEventListener("keyup", (e) => {
  if (e.code === "Space") spaceHeld = false;
});

orderSelect.addEventListener("change", () => {
  nextCache = null;
  goTo("next");
});

prevBtn.addEventListener("click", () => goTo("prev"));
nextBtn.addEventListener("click", () => goTo("next"));
skipBtn.addEventListener("click", () => skipCurrent());
deleteBtn.addEventListener("click", () => deleteSelected());

// ---------- boot ----------

(async function init() {
  await loadLabels();
  await goTo("next");
})();
