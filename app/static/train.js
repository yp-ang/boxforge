const projectId = Number(document.body.dataset.projectId);

const datasetSelect = document.getElementById("dataset-select");
const exportBtn = document.getElementById("export-btn");
const exportReport = document.getElementById("export-report");
const trainForm = document.getElementById("train-form");
const trainBtn = document.getElementById("train-btn");
const trainError = document.getElementById("train-error");
const modelSelect = document.getElementById("cfg-model");
const previewBtn = document.getElementById("preview-btn");
const previewGrid = document.getElementById("preview-grid");
const previewNote = document.getElementById("preview-note");
const jobSection = document.getElementById("job-section");
const jobIdEl = document.getElementById("job-id");
const jobStatusEl = document.getElementById("job-status");
const cancelBtn = document.getElementById("cancel-btn");
const logEl = document.getElementById("log");
const modelRows = document.getElementById("model-rows");
const modelDetail = document.getElementById("model-detail");

let eventSource = null;
let currentJobId = null;

const num = (id) => Number(document.getElementById(id).value);

function readConfig() {
  return {
    model: modelSelect.value,
    epochs: num("cfg-epochs"),
    imgsz: num("cfg-imgsz"),
    batch: num("cfg-batch"),
    patience: num("cfg-patience"),
    device: document.getElementById("cfg-device").value,
    seed: num("cfg-seed"),
    fliplr: document.getElementById("cfg-fliplr").checked ? 0.5 : 0.0,
  };
}

function showError(message) {
  trainError.textContent = message;
  trainError.hidden = !message;
}

// --- datasets and model list ------------------------------------------------

async function loadDatasets() {
  const datasets = await (await fetch(`/api/projects/${projectId}/datasets`)).json();
  datasetSelect.innerHTML = "";
  if (!datasets.length) {
    datasetSelect.appendChild(new Option("no dataset exported yet", ""));
    return;
  }
  for (const d of datasets) {
    const label = `#${d.id} — ${d.n_train} train / ${d.n_val} val — ${d.created_at.slice(0, 16).replace("T", " ")}`;
    datasetSelect.appendChild(new Option(label, d.id));
  }
}

async function loadCheckpoints() {
  const models = await (await fetch("/api/train/models")).json();
  modelSelect.innerHTML = "";
  for (const m of models) {
    modelSelect.appendChild(new Option(m.cached ? m.name : `${m.name} (will download)`, m.name));
  }
}

exportBtn.addEventListener("click", async () => {
  exportBtn.disabled = true;
  exportReport.hidden = false;
  exportReport.textContent = "exporting…";
  const res = await fetch(`/api/projects/${projectId}/datasets`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ val_pct: 20, force: true }),
  });
  const body = await res.json();
  exportReport.textContent = JSON.stringify(body, null, 2);
  exportBtn.disabled = false;
  await loadDatasets();
});

// --- starting a run ---------------------------------------------------------

trainForm.addEventListener("submit", async (e) => {
  e.preventDefault();
  showError("");
  const datasetId = datasetSelect.value ? Number(datasetSelect.value) : null;
  if (!datasetId) return showError("export a dataset first");

  trainBtn.disabled = true;
  const res = await fetch(`/api/projects/${projectId}/train`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ dataset_id: datasetId, config: readConfig() }),
  });
  trainBtn.disabled = false;
  if (!res.ok) return showError((await res.json()).detail || "could not start training");

  attachToJob((await res.json()).id);
});

// --- live logs --------------------------------------------------------------

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
  currentJobId = jobId;
  // Survives a reload: the stream replays the log from the top, so reattaching shows the
  // whole run rather than whatever happened to arrive after the page came back.
  sessionStorage.setItem(`job:${projectId}`, String(jobId));

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
    await loadModels();
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
  if (!currentJobId) return;
  cancelBtn.disabled = true;
  await fetch(`/api/jobs/${currentJobId}/cancel`, { method: "POST" });
  await refreshJobStatus(currentJobId);
});

// --- model registry ---------------------------------------------------------

function metric(model, key) {
  if (!model.metrics_json) return "—";
  const value = JSON.parse(model.metrics_json)[key];
  return value === undefined ? "—" : value.toFixed(3);
}

async function loadModels() {
  const models = await (await fetch(`/api/projects/${projectId}/models`)).json();
  const jobs = await (await fetch(`/api/projects/${projectId}/jobs?limit=100`)).json();
  const configByJob = Object.fromEntries(
    jobs.map((j) => [j.id, j.params_json ? JSON.parse(j.params_json).config || {} : {}])
  );

  modelRows.innerHTML = "";
  for (const model of models) {
    const cfg = configByJob[model.job_id] || {};
    const tr = document.createElement("tr");
    if (model.is_active) tr.className = "active";

    const radio = document.createElement("input");
    radio.type = "radio";
    radio.name = "active-model";
    radio.checked = model.is_active;
    radio.title = "make this the model steps 06/07 use";
    radio.addEventListener("change", async () => {
      await fetch(`/api/models/${model.id}/activate`, { method: "POST" });
      await loadModels();
    });

    const cells = [
      radio, model.name,
      metric(model, "mAP50-95"), metric(model, "mAP50"),
      metric(model, "precision"), metric(model, "recall"),
      model.dataset_id ? `#${model.dataset_id}` : "—",
      cfg.epochs ?? "—", cfg.imgsz ?? "—",
      model.created_at.slice(0, 16).replace("T", " "),
    ];
    for (const cell of cells) {
      const td = document.createElement("td");
      if (cell instanceof Node) td.appendChild(cell);
      else td.textContent = cell;
      tr.appendChild(td);
    }

    const plotsTd = document.createElement("td");
    const plotsBtn = document.createElement("button");
    plotsBtn.textContent = "plots";
    plotsBtn.addEventListener("click", () => showPlots(model));
    plotsTd.appendChild(plotsBtn);
    tr.appendChild(plotsTd);

    modelRows.appendChild(tr);
  }
}

function showPlots(model) {
  modelDetail.hidden = false;
  modelDetail.innerHTML = `<h3>${model.name}</h3>`;
  for (const name of ["results.png", "confusion_matrix.png"]) {
    const img = document.createElement("img");
    img.src = `/api/models/${model.id}/plots/${name}`;
    img.alt = name;
    img.onerror = () => img.remove();
    modelDetail.appendChild(img);
  }
}

// --- augmentation preview ---------------------------------------------------

previewBtn.addEventListener("click", async () => {
  previewBtn.disabled = true;
  previewGrid.textContent = "rendering…";
  const cfg = readConfig();
  const res = await fetch(
    `/api/projects/${projectId}/augment-preview?n=6&imgsz=${cfg.imgsz}&fliplr=${cfg.fliplr}`
  );
  previewBtn.disabled = false;
  previewGrid.textContent = "";
  if (!res.ok) {
    previewGrid.textContent = (await res.json()).detail || "preview failed";
    return;
  }
  const body = await res.json();
  previewNote.hidden = !body.note;
  previewNote.textContent = body.note || "";
  for (const item of body.items) {
    const fig = document.createElement("figure");
    fig.className = "preview-pair";
    for (const [src, caption] of [[item.original, "original"], [item.augmented, "augmented"]]) {
      const img = document.createElement("img");
      img.src = src;
      img.alt = caption;
      fig.appendChild(img);
    }
    const cap = document.createElement("figcaption");
    cap.textContent = `image ${item.image_id} — original / augmented`;
    fig.appendChild(cap);
    previewGrid.appendChild(fig);
  }
});

// --- boot -------------------------------------------------------------------

(async function init() {
  await Promise.all([loadDatasets(), loadCheckpoints(), loadModels()]);

  // Reattach to whatever is running: the job this browser started, or one started by
  // another tab entirely.
  const jobs = await (await fetch(`/api/projects/${projectId}/jobs?limit=20`)).json();
  const active = jobs.find((j) => j.status === "running" || j.status === "queued");
  const remembered = Number(sessionStorage.getItem(`job:${projectId}`)) || null;
  const target = active || jobs.find((j) => j.id === remembered);
  if (target) attachToJob(target.id);
})();
