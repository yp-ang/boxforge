const projectSelect = document.getElementById("project-select");
const annotateLink = document.getElementById("annotate-link");
const trainLink = document.getElementById("train-link");
const projectForm = document.getElementById("project-form");
const labelList = document.getElementById("label-list");
const labelForm = document.getElementById("label-form");
const ingestForm = document.getElementById("ingest-form");
const ingestReport = document.getElementById("ingest-report");
const statsEl = document.getElementById("stats");

let projects = [];

function currentProjectId() {
  return projectSelect.value ? Number(projectSelect.value) : null;
}

async function loadProjects() {
  const res = await fetch("/api/projects");
  projects = await res.json();
  const prev = currentProjectId();
  projectSelect.innerHTML = "";
  for (const p of projects) {
    const opt = document.createElement("option");
    opt.value = p.id;
    opt.textContent = `${p.name} (${p.slug})`;
    projectSelect.appendChild(opt);
  }
  if (prev && projects.some((p) => p.id === prev)) {
    projectSelect.value = prev;
  }
  await refreshProject();
}

async function refreshProject() {
  const id = currentProjectId();
  labelList.innerHTML = "";
  statsEl.textContent = "";
  annotateLink.href = id ? `/annotate/${id}` : "#";
  trainLink.href = id ? `/train/${id}` : "#";
  if (!id) return;

  const stats = await (await fetch(`/api/projects/${id}/stats`)).json();
  statsEl.textContent = JSON.stringify(stats, null, 2);

  const labels = await (await fetch(`/api/projects/${id}/labels`)).json();
  for (const label of labels) addLabelRow(label);
}

projectSelect.addEventListener("change", refreshProject);

projectForm.addEventListener("submit", async (e) => {
  e.preventDefault();
  const name = document.getElementById("project-name").value.trim();
  if (!name) return;
  const res = await fetch("/api/projects", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ name }),
  });
  if (res.ok) {
    document.getElementById("project-name").value = "";
    await loadProjects();
  } else {
    alert((await res.json()).detail);
  }
});

labelForm.addEventListener("submit", async (e) => {
  e.preventDefault();
  const id = currentProjectId();
  if (!id) return alert("create a project first");
  const name = document.getElementById("label-name").value.trim();
  const color = document.getElementById("label-color").value;
  if (!name) return;
  const res = await fetch(`/api/projects/${id}/labels`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ name, color }),
  });
  if (res.ok) {
    const label = await res.json();
    addLabelRow(label);
    document.getElementById("label-name").value = "";
    await refreshProject();
  } else {
    alert((await res.json()).detail);
  }
});

function addLabelRow(label) {
  const li = document.createElement("li");
  li.dataset.labelId = label.id;
  const swatch = document.createElement("span");
  swatch.className = "swatch";
  swatch.style.background = label.color;
  const text = document.createElement("span");
  text.textContent = `${label.name} (class ${label.class_index})`;
  const del = document.createElement("button");
  del.textContent = "Delete";
  del.addEventListener("click", async () => {
    await fetch(`/api/labels/${label.id}`, { method: "DELETE" });
    li.remove();
  });
  li.append(swatch, text, del);
  labelList.appendChild(li);
}

ingestForm.addEventListener("submit", async (e) => {
  e.preventDefault();
  const id = currentProjectId();
  if (!id) return alert("create a project first");
  const source_dir = document.getElementById("ingest-dir").value.trim();
  const copy = document.getElementById("ingest-copy").checked;
  ingestReport.textContent = "ingesting...";
  const res = await fetch(`/api/projects/${id}/ingest`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ source_dir, copy }),
  });
  const body = await res.json();
  ingestReport.textContent = JSON.stringify(body, null, 2);
  if (res.ok) await refreshProject();
});

loadProjects();
