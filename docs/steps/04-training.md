# Step 04 — Training

**Goal:** start a training run from the UI, watch its logs live, and end up with a
checkpoint recorded in the database. Without blocking the web server, and without adding
a broker.

**Depends on:** step 03.

## 1. Job runner

Run training in a **subprocess**, not a thread. Training holds the GIL, allocates
gigabytes, and occasionally dies in native code — none of which you want inside your web
process. A subprocess can be killed cleanly and cannot take the UI down with it.

```python
# app/services/jobs.py
def start_job(db, job_type: str, project_id: int, params: dict) -> Job:
    job = Job(type=job_type, project_id=project_id, status="queued",
              params_json=json.dumps(params))
    db.add(job); db.commit()

    log_path = settings.runs_dir / str(job.id) / "job.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    job.log_path = str(log_path)

    proc = subprocess.Popen(
        [sys.executable, "-m", "app.worker", str(job.id)],
        stdout=open(log_path, "w", buffering=1),   # line buffered
        stderr=subprocess.STDOUT,
        env={**os.environ, "PYTHONUNBUFFERED": "1"},
    )
    job.status, job.pid = "running", proc.pid
    db.commit()
    return job
```

`app/worker.py` is a standalone entry point: it opens its own DB session, reads the job
row, dispatches on `job.type`, writes `result_json`, sets the terminal status, and exits.
Everything it prints lands in the log file.

**One job at a time.** Refuse a new training job while one is running — two runs on one
GPU (or one laptop) is slower than running them in sequence, and the failure mode is an
OOM you have to diagnose. A single `WHERE status='running'` check is the whole feature.

## 2. Trainer behind an interface

Only this module imports `ultralytics`. See ARCHITECTURE §5.1 for why that isolation is
worth the indirection.

```python
# app/services/trainer.py
@dataclass
class TrainConfig:
    model: str = "yolo11n.pt"     # n=nano (fast, weak) … s, m, l, x
    epochs: int = 100
    imgsz: int = 640
    batch: int = -1               # -1 = auto-fit to available memory
    patience: int = 30            # early stop
    device: str = "auto"
    seed: int = 0                 # reproducibility

class UltralyticsTrainer:
    def train(self, dataset_yaml: Path, cfg: TrainConfig, run_dir: Path) -> TrainResult:
        from ultralytics import YOLO
        model = YOLO(cfg.model)
        results = model.train(
            data=str(dataset_yaml), epochs=cfg.epochs, imgsz=cfg.imgsz,
            batch=cfg.batch, patience=cfg.patience, device=resolve_device(cfg.device),
            seed=cfg.seed, project=str(run_dir.parent), name=run_dir.name,
            exist_ok=True, plots=True,
        )
        best = run_dir / "weights" / "best.pt"
        return TrainResult(weights=best, metrics=extract_metrics(results))
```

Device resolution, given ARCHITECTURE §5.3:

```python
def resolve_device(pref: str) -> str:
    if pref != "auto": return pref
    import torch
    if torch.cuda.is_available(): return "0"
    if torch.backends.mps.is_available(): return "mps"    # Apple Silicon, conda only
    return "cpu"
```

## 3. Sensible defaults for a personal dataset

Starting points, not laws:

| Situation | Model | Epochs | imgsz |
|---|---|---|---|
| First run, <500 images | `yolo11n.pt` | 100 | 640 |
| Serious run, 2k+ images | `yolo11s.pt` | 200–300 | 640 |
| Small distant objects | `yolo11s.pt` | 300 | 960 or 1280 |
| CPU-only, just proving the pipeline | `yolo11n.pt` | 20 | 416 |

Always start from a **pretrained** checkpoint (`.pt`, not `.yaml`). Fine-tuning COCO
weights on 300 images works; training from scratch on 300 images does not, and the
difference is not subtle.

Leave the Ultralytics default augmentations alone at first — mosaic, HSV jitter and
flips are already tuned. One exception worth setting deliberately: if your objects have a
meaningful orientation (text, gauges, signage), set `fliplr=0.0`, because a mirrored
forklift is still a forklift but a mirrored "7" is not.

## 4. Live logs via SSE

Server-Sent Events, not WebSockets — one-directional, reconnects automatically, and it is
about fifteen lines:

```python
@router.get("/api/jobs/{job_id}/logs")
async def stream_logs(job_id: int, db=Depends(get_db)):
    job = db.get(Job, job_id) or abort(404)

    async def gen():
        with open(job.log_path) as f:
            while True:
                line = f.readline()
                if line:
                    yield f"data: {line.rstrip()}\n\n"
                    continue
                if job_finished(job_id):          # re-read status
                    yield "event: done\ndata: {}\n\n"
                    return
                await asyncio.sleep(0.4)

    return StreamingResponse(gen(), media_type="text/event-stream")
```

```js
const es = new EventSource(`/api/jobs/${jobId}/logs`);
es.onmessage = e => appendLine(e.data);
es.addEventListener("done", () => { es.close(); refreshModels(); });
```

## 5. Recording the result

On success the worker writes a `Model` row: the run directory, `classes_json` copied from
the dataset manifest, and `metrics_json` with mAP50, mAP50-95, precision and recall.

Ultralytics also writes `results.png`, `confusion_matrix.png` and `results.csv` into the
run directory. Surface those in the model detail page — the confusion matrix in particular
tells you *which* class is failing, which is the question you actually have.

## 6. What the numbers mean

- **mAP50-95** — the headline. >0.5 is strong for a custom dataset, 0.3–0.5 is usable,
  <0.2 means a data problem rather than a training problem.
- **mAP50** — looser IoU. If mAP50 is high and mAP50-95 is low, the model finds objects
  but its boxes are sloppy — usually inconsistent annotation tightness.
- **Precision high, recall low** — misses objects. More data, or more varied data.
- **Recall high, precision low** — false positives. Add background images (annotated,
  zero boxes) showing the scenes where it fires wrongly.

Write these four lines into the UI next to the metrics. Future-you will not remember them.

## 7. Augmentation preview — sanity-check before you spend an hour training

Roboflow shows you what its augmentation pipeline actually does to your images before you
commit to a training run. Ultralytics' defaults (mosaic, HSV jitter, flips, translation)
are well-tuned, but "well-tuned in general" isn't the same as "correct for your dataset" —
the one case worth catching before training rather than after is orientation-sensitive
objects (gauges, signage, text) getting mirrored by the default `fliplr=0.5`.

This doesn't need real training code — Ultralytics' augmentation transforms are usable
standalone against a couple of sample images:

```
GET /api/projects/{id}/augment-preview?n=6   → 6 images, each shown once original,
                                                once with the configured augmentations applied
```

```python
from ultralytics.data.augment import Compose, build_transforms   # exact call per Ultralytics version
def preview(sample_images: list[np.ndarray], cfg: TrainConfig) -> list[np.ndarray]:
    transforms = build_transforms(cfg_to_hyp(cfg))
    return [transforms(im) for im in sample_images]
```

Render as a small grid on the "start training" screen, next to the `fliplr` checkbox
mentioned in §3. Seeing a mirrored "7" next to the flip toggle is a much faster way to
catch that setting than reading the mAP confusion matrix after the fact.

## 8. A lightweight model registry

Labelbox and Roboflow both let you compare model versions side by side and mark one as
the version actually in use. You already have every trained `Model` row with its metrics
(§5) — the registry is a query and a flag, not new infrastructure:

```
GET  /api/projects/{id}/models                    → all Model rows, sorted by created_at,
                                                     with mAP50-95, dataset id, and job params
POST /api/models/{id}/activate                     → sets Model.is_active=True,
                                                     clears the flag on any other model
                                                     in the same project
```

```python
class Model(Base):
    ...
    is_active: Mapped[bool] = mapped_column(default=False)
```

Two things this buys you, cheaply:

- **A comparison table** — mAP50-95, precision/recall, dataset version, epochs, imgsz —
  across every run in a project, so "is v3 actually better than v2" is a glance instead of
  a memory exercise.
- **A single answer to "which model is current"** for step 06 (verify) and step 07
  (pre-labelling) to default to, instead of always picking "most recent" — useful once
  you've trained a worse run by accident and don't want it silently picked up next.

Not building: full experiment tracking (MLflow/W&B-style metric curves, artifact diffing).
The `results.png`/`results.csv` Ultralytics already writes per run (§5) covers that need at
this scale; a registry table is just the index over runs you already have.

## Acceptance criteria

- Start a 5-epoch run on a small dataset from the UI; the browser stays responsive.
- Logs stream live; closing and reopening the page reattaches.
- Cancel mid-run → process dies, status `cancelled`, no orphan.
- Kill the training process externally → job reconciles to `failed`, app survives.
- On success: a `Model` row exists, `best.pt` is on disk, metrics are displayed.
- Starting a second job while one runs → refused with a clear message.
- (If §7 built) Preview grid renders augmented variants that visibly differ from
  originals (jittered colour, at least one flipped/mosaiced sample).
- (If §8 built) Training 3 models in one project → all 3 appear in the comparison table;
  activating one clears the flag on the others.
