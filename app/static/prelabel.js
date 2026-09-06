const projectId = Number(document.body.dataset.projectId);

const modelSelect = document.getElementById("pl-model");
const confSlider = document.getElementById("pl-conf");
const confValue = document.getElementById("pl-conf-value");
const limitInput = document.getElementById("pl-limit");
const previewBtn = document.getElementById("pl-preview-btn");
const previewEl = document.getElementById("pl-preview");
const errorEl = document.getElementById("pl-error");
const form = document.getElementById("prelabel-form");

const jobSection = document.getElementById("job-section");
const jobIdEl = document.getElementById("job-id");
const jobStatusEl = document.getElementById("job-status");
const cancelBtn = document.getElementById("cancel-btn");
const logEl = document.getElementById("log");

const batchRows = document.getElementById("batch-rows");
const gridSection = document.getElementById("grid-section");
const gridJobIdEl = document.getElementById("grid-job-id");
const gridEl = document.getElementById("grid");
const gridEmptyEl = document.getElementById("grid-empty");

let eventSource = null;

function showError(message) {
  errorEl.textContent = message;
  errorEl.hidden = !message;
}

confSlider.addEventListener("input", () => {
  confValue.textContent = (+confSlider.value).toFixed(2);
});

// --- model list ---------------------------------------------------------------

async function loadModels() {
  const models = await (await fetch(`/api/projects/${projectId}/models`)).json();
  const usable = models.filter((m) => m.onnx_path);
  modelSelect.innerHTML = "";
  if (!usable.length) {
    modelSelect.appendChild(new Option("no exported model yet", ""));
    return;
  }
  for (const m of usable) {
    modelSelect.appendChild(new Option(`${m.name}${m.is_active ? " (active)" : ""}`, m.id));
  }
  const active = usable.find((m) => m.is_active) || usable[0];
  modelSelect.value = active.id;
}

// --- preview count (doc §5 guardrail) -------------------------------------------

previewBtn.addEventListener("click", async () => {
  showError("");
  if (!modelSelect.value) return showError("no exported model available");
  const params = new URLSearchParams({
    model_id: modelSelect.value, conf: confSlider.value, limit: limitInput.value,
  });
  const res = await fetch(`/api/projects/${projectId}/prelabel-preview?${params}`);
  if (!res.ok) return showError((await res.json()).detail || "could not preview");
  const body = await res.json();
  previewEl.textContent = `will draft boxes on ${body.eligible} pending image` +
    `${body.eligible === 1 ? "" : "s"} @ conf ${body.conf.toFixed(2)}`;
});

// --- start a batch ----------------------------------------------------------------

form.addEventListener("submit", async (e) => {
  e.preventDefault();
  showError("");
  if (!modelSelect.value) return showError("no exported model available");

  const res = await fetch(`/api/projects/${projectId}/prelabel`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      model_id: Number(modelSelect.value), conf: +confSlider.value, limit: Number(limitInput.value),
    }),
  });
  if (!res.ok) return showError((await res.json()).detail || "could not start batch");
  attachToJob((await res.json()).id);
});

function appendLine(text) {
  const atBottom = logEl.scrollHeight - logEl.scrollTop - logEl.clientHeight < 40;
  logEl.textContent += text + "\n";
  if (atBottom) logEl.scrollTop = logEl.scrollHeight;
}

function setStatus(status) {
  jobStatusEl.textContent = status;
  jobStatusEl.className = `status-${status}`;
  cancelBtn.disabled = !["queued", "running"].includes(status);
}

function attachToJob(jobId) {
  if (eventSource) eventSource.close();
  jobSection.hidden = false;
  jobIdEl.textContent = `#${jobId}`;
  logEl.textContent = "";
  setStatus("running");

  eventSource = new EventSource(`/api/jobs/${jobId}/logs`);
  eventSource.onmessage = (e) => appendLine(e.data);
  eventSource.addEventListener("done", async () => {
    eventSource.close();
    eventSource = null;
    await refreshJobStatus(jobId);
    await loadBatches();
  });
  eventSource.onerror = () => refreshJobStatus(jobId);
  refreshJobStatus(jobId);
}

async function refreshJobStatus(jobId) {
  const res = await fetch(`/api/jobs/${jobId}`);
  if (!res.ok) return;
  const job = await res.json();
  setStatus(job.status);
  if (job.status === "failed" && job.result_json) {
    const parsed = JSON.parse(job.result_json);
    if (parsed.error) showError(`job ${jobId} failed: ${parsed.error}`);
  }
}

cancelBtn.addEventListener("click", async () => {
  if (!jobIdEl.textContent) return;
  const jobId = jobIdEl.textContent.replace("#", "");
  cancelBtn.disabled = true;
  await fetch(`/api/jobs/${jobId}/cancel`, { method: "POST" });
  await refreshJobStatus(jobId);
});

// --- batch list ---------------------------------------------------------------

async function loadBatches() {
  const jobs = await (await fetch(`/api/projects/${projectId}/jobs?limit=50`)).json();
  const batches = jobs.filter((j) => j.type === "prelabel");
  batchRows.innerHTML = "";
  for (const job of batches) {
    const params = job.params_json ? JSON.parse(job.params_json) : {};
    const result = job.result_json ? JSON.parse(job.result_json) : null;

    const tr = document.createElement("tr");
    const cells = [
      `#${job.id}`,
      result ? `#${result.model_id}` : `#${params.model_id ?? "?"}`,
      (result ? result.conf : params.conf)?.toFixed?.(2) ?? "—",
      result ? result.n_images : "—",
      result ? result.n_boxes : "—",
      job.status,
    ];
    for (const cell of cells) {
      const td = document.createElement("td");
      td.textContent = cell;
      tr.appendChild(td);
    }

    const actionsTd = document.createElement("td");
    if (job.status === "done") {
      const reviewBtn = document.createElement("button");
      reviewBtn.type = "button";
      reviewBtn.textContent = "review";
      reviewBtn.addEventListener("click", () => openGrid(job.id));
      actionsTd.appendChild(reviewBtn);

      const undoBtn = document.createElement("button");
      undoBtn.type = "button";
      undoBtn.textContent = "undo";
      undoBtn.addEventListener("click", async () => {
        if (!confirm(`Undo batch #${job.id}? This deletes its unreviewed draft boxes.`)) return;
        await fetch(`/api/jobs/${job.id}/prelabel/undo`, { method: "POST" });
        await loadBatches();
        if (gridJobIdEl.textContent === `#${job.id}`) gridSection.hidden = true;
      });
      actionsTd.appendChild(undoBtn);
    }
    tr.appendChild(actionsTd);
    batchRows.appendChild(tr);
  }
}

// --- grid review (doc §6) -------------------------------------------------------

let gridItems = [];
let focusedIndex = -1;

async function openGrid(jobId) {
  gridJobIdEl.textContent = `#${jobId}`;
  gridSection.hidden = false;
  gridEl.innerHTML = "";
  const res = await fetch(`/api/projects/${projectId}/prelabel-batch/${jobId}`);
  if (!res.ok) {
    gridEmptyEl.hidden = false;
    gridEmptyEl.textContent = (await res.json()).detail || "could not load batch";
    return;
  }
  const body = await res.json();
  gridItems = body.images;
  gridEmptyEl.hidden = gridItems.length > 0;
  focusedIndex = gridItems.length ? 0 : -1;
  for (const [i, item] of gridItems.entries()) renderTile(i, item);
  updateFocusRing();
  gridSection.scrollIntoView({ behavior: "smooth", block: "start" });
}

function renderTile(index, item) {
  const tile = document.createElement("div");
  tile.className = "pl-tile";
  tile.dataset.index = String(index);

  const canvas = document.createElement("canvas");
  tile.appendChild(canvas);

  const actions = document.createElement("div");
  actions.className = "pl-actions";
  const acceptBtn = document.createElement("button");
  acceptBtn.className = "pl-accept";
  acceptBtn.textContent = "✓ accept";
  acceptBtn.addEventListener("click", (e) => { e.stopPropagation(); acceptTile(index); });
  const rejectBtn = document.createElement("button");
  rejectBtn.className = "pl-reject";
  rejectBtn.textContent = "✕ reject";
  rejectBtn.addEventListener("click", (e) => { e.stopPropagation(); rejectTile(index); });
  actions.append(acceptBtn, rejectBtn);
  tile.appendChild(actions);

  tile.addEventListener("click", () => { focusedIndex = index; updateFocusRing(); });
  gridEl.appendChild(tile);

  const img = new window.Image();
  img.onload = () => {
    const scale = Math.min(1, 400 / img.naturalWidth);
    canvas.width = img.naturalWidth * scale;
    canvas.height = img.naturalHeight * scale;
    const ctx = canvas.getContext("2d");
    ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
    for (const a of item.annotations) {
      ctx.strokeStyle = "#4285f4";
      ctx.lineWidth = 2;
      ctx.setLineDash([5, 4]);
      ctx.strokeRect(a.x1 * canvas.width, a.y1 * canvas.height,
        (a.x2 - a.x1) * canvas.width, (a.y2 - a.y1) * canvas.height);
    }
    ctx.setLineDash([]);
  };
  img.src = `/api/images/${item.image.id}/file`;
}

function updateFocusRing() {
  for (const tile of gridEl.children) {
    tile.classList.toggle("focused", Number(tile.dataset.index) === focusedIndex);
  }
}

function tileAt(index) {
  return gridEl.querySelector(`.pl-tile[data-index="${index}"]`);
}

async function acceptTile(index) {
  const item = gridItems[index];
  if (!item) return;
  await fetch(`/api/images/${item.image.id}/accept`, { method: "POST" });
  removeTile(index);
}

async function rejectTile(index) {
  const item = gridItems[index];
  if (!item) return;
  await fetch(`/api/images/${item.image.id}/reject`, { method: "POST" });
  removeTile(index);
}

function removeTile(index) {
  const tile = tileAt(index);
  if (tile) { tile.classList.add("rejecting"); setTimeout(() => tile.remove(), 150); }
  gridItems[index] = null;
  if (gridItems.every((it) => it === null)) {
    setTimeout(() => { gridEmptyEl.hidden = false; }, 200);
  } else if (focusedIndex === index) {
    const next = gridItems.findIndex((it, i) => it && i > index);
    focusedIndex = next >= 0 ? next : gridItems.findIndex((it) => it);
    updateFocusRing();
  }
}

window.addEventListener("keydown", (e) => {
  if (gridSection.hidden || focusedIndex < 0 || !gridItems[focusedIndex]) return;
  if (["INPUT", "SELECT", "TEXTAREA"].includes(document.activeElement.tagName)) return;

  if (e.code === "Space") {
    e.preventDefault();
    acceptTile(focusedIndex);
  } else if (e.key === "x" || e.key === "X") {
    e.preventDefault();
    rejectTile(focusedIndex);
  } else if (e.key === "Enter") {
    window.open(`/annotate/${projectId}`, "_blank");
  } else if (e.key === "ArrowRight" || e.key === "ArrowLeft" ||
            e.key === "ArrowDown" || e.key === "ArrowUp") {
    e.preventDefault();
    const dir = (e.key === "ArrowRight" || e.key === "ArrowDown") ? 1 : -1;
    let i = focusedIndex;
    do {
      i = (i + dir + gridItems.length) % gridItems.length;
    } while (!gridItems[i] && i !== focusedIndex);
    focusedIndex = i;
    updateFocusRing();
    tileAt(i)?.scrollIntoView({ block: "nearest" });
  }
});

// --- boot ---------------------------------------------------------------------

(async function init() {
  await loadModels();
  await loadBatches();
})();
