# Architecture

## 1. Shape of the system

One FastAPI process serves everything: the annotation UI, the training job runner, the
inference playground, and a small JSON API. There is no separate frontend build, no
message broker, no external database.

That is a deliberate choice for a personal project. Every component you add is a
component you have to keep alive at 11pm on a Sunday. The design below has exactly one
process, one file-backed database, and one directory of data.

```
┌─────────────────────────────────────────────────────────────┐
│  FastAPI (uvicorn)                                          │
│                                                             │
│  routers/           services/              static/          │
│   projects           ingest                 annotate.html   │
│   labels             dataset_export         verify.html     │
│   images             trainer  ──┐           app.js          │
│   annotations        exporter   │                           │
│   datasets           runtime    │ spawns                    │
│   training           faces      │                           │
│   inference          jobs ──────┘                           │
└───────┬─────────────────────────────┬───────────────────────┘
        │                             │
   ┌────▼─────┐                 ┌─────▼──────────────────────┐
   │ app.db   │                 │ worker subprocess          │
   │ (SQLite) │                 │  ultralytics train / export│
   └──────────┘                 │  writes logs + weights     │
                                └────────────────────────────┘
        data/
          images/<project>/…        original files (or symlinks)
          datasets/<project>/<ts>/  generated YOLO layout
          runs/<job_id>/            training output, logs
          models/<model_id>/        best.pt, model.onnx, classes.json
```

## 2. Stack, and why

| Concern | Choice | Why this and not the alternative |
|---|---|---|
| Language | **Python 3.11** | Not 3.12+, not your system 3.14. The CV/torch ecosystem lags; 3.11 is the version where every wheel you need exists prebuilt. Pin it. |
| Web framework | **FastAPI + Uvicorn** | Async, typed, auto OpenAPI docs at `/docs` (free API explorer while you build), serves static files fine. Flask would also work; FastAPI's pydantic validation earns its keep on the annotation endpoints. |
| Frontend | **Vanilla JS + HTML `<canvas>`**, served static | Zero build step. No npm, no bundler, no lockfile rot. The annotation UI is ~500 lines of canvas drag-handling; a React toolchain would be more code than the feature. Add [Alpine.js](https://alpinejs.dev) (one `<script>` tag) if you want reactive bindings. |
| Database | **SQLite + SQLAlchemy 2.0** | One file, atomic writes, real queries ("show me every image with zero annotations"), trivially backed up. JSON-on-disk seems simpler until two requests write at once. Postgres is ops you don't need. |
| Detector | **Ultralytics YOLO11n/s** | Best training-quality-per-line-of-code available. Exports to ONNX in one call. **See licensing warning below.** |
| Inference runtime | **onnxruntime** | Deliberately *not* torch. Runs the exported artifact, proving the thing you hand to other software actually works. Also 10× lighter in the container. |
| Face embedding | **InsightFace ArcFace (ONNX)** | The de-facto open face-recognition model. Detection via SCRFD or your own YOLO face class. |
| Vector search | **numpy cosine similarity**, FAISS only if >50k faces | For a personal gallery, a `(N, 512)` matrix and one `@` is faster than the FAISS import. |
| Jobs | **subprocess + SQLite job table + SSE log stream** | Celery/Redis is two more services. A `Popen` writing to a log file, plus Server-Sent Events tailing it, gives you live training logs in the browser with no broker. |
| Config | **pydantic-settings + `.env`** | Same config object in conda and Docker; env vars override file. |
| Packaging | **`environment.yml`** (conda) and **multi-stage Dockerfile + compose profiles** | Both paths, one codebase, as you asked. |

### Interchange format: ONNX

This is the load-bearing decision for "integrates with widely used visual analytic
software". ONNX is the format that DeepStream, OpenVINO, TensorRT, Frigate, Milestone
and Genetec integrations, and every Python/C++/C# runtime can consume. Everything else
(CoreML, TFLite, TensorRT engine) is a downstream conversion *from* ONNX.

So the contract this app publishes is a folder:

```
models/<model_id>/
  model.onnx        # NCHW float32, input 1×3×640×640, normalised 0-1
  classes.json      # {"0": "forklift", "1": "pallet"}  — index matches training
  metadata.json     # imgsz, task, mAP50-95, trained_at, source dataset hash
  best.pt           # kept for retraining, NOT for deployment
```

Anything that can read those four files can use your model. See
[docs/integration.md](docs/integration.md).

## 3. Data model

SQLAlchemy tables. Small enough to hold in your head, which is the point.

```python
Project(id, name, slug, task, created_at)
    task ∈ {"detect"}                      # room for "segment" later

Label(id, project_id, name, color, class_index)
    UNIQUE(project_id, class_index)        # class_index is the YOLO id — never reuse
    UNIQUE(project_id, name)

Image(id, project_id, rel_path, sha256, width, height, status, ingested_at)
    status ∈ {"pending", "annotated", "skipped"}
    UNIQUE(project_id, sha256)             # dedupe on re-ingest

Annotation(id, image_id, label_id, x1, y1, x2, y2, source, created_at)
    coords NORMALISED 0-1, top-left origin
    source ∈ {"human", "model"}            # so pre-labels are auditable

Job(id, type, status, project_id, params_json, log_path, result_json, started_at, ended_at)
    type ∈ {"train", "export", "prelabel"}
    status ∈ {"queued", "running", "done", "failed", "cancelled"}

Model(id, project_id, job_id, name, dir_path, classes_json, metrics_json, created_at)

FaceIdentity(id, name, notes, created_at)
FaceEmbedding(id, identity_id, vector_blob, source_image, created_at)
```

Two details worth defending:

**Store bounding boxes as normalised `x1,y1,x2,y2`.** The UI thinks in corners; YOLO
thinks in `cx,cy,w,h`. Convert in exactly one function, at export time. Every bbox bug
I have ever seen came from doing that conversion in three places with two conventions.

**`class_index` is assigned once and never reused.** If you delete a label, its index is
retired. Otherwise a model trained last month silently means something different than its
`classes.json` claims.

## 4. Train/val split

Assigned at export time, deterministically, from the image hash:

```python
bucket = int(sha256_hex[:8], 16) % 100
split  = "val" if bucket < 20 else "train"     # 80/20
```

Consequence: re-exporting after adding images keeps every existing image in the split it
was already in. Your validation metrics stay comparable across runs, and no image ever
leaks from train to val between experiments. A random shuffle would break both.

## 5. Known constraints — read before you start

### 5.1 Ultralytics is AGPL-3.0

YOLO11 via the `ultralytics` package is **AGPL-3.0**. For a personal project that you run
yourself and don't distribute, this is fine and costs you nothing. It becomes a real
problem the moment you ship the software to someone else or run it as a network service
for others — AGPL then requires you to release your source, and it is viral across the
linked application.

Mitigation, cheap to do now and expensive to retrofit: keep the trainer behind an
interface.

```python
# services/trainer.py
class Trainer(Protocol):
    def train(self, dataset_yaml: Path, cfg: TrainConfig, log: TextIO) -> Path: ...
    def export_onnx(self, weights: Path, out: Path, imgsz: int) -> Path: ...

class UltralyticsTrainer(Trainer): ...   # today
# swappable later for YOLOX (Apache-2.0), RF-DETR, or torchvision's detectors
```

Nothing else in the codebase imports `ultralytics`. If licensing ever matters, you
replace one file rather than the project.

### 5.2 Webcam inside Docker does not work on macOS

Docker Desktop on macOS runs a Linux VM with no host USB/AVFoundation passthrough.
`cv2.VideoCapture(0)` inside the container will fail, and no `--device` flag fixes it.
This is a platform limitation, not a config error.

Design around it rather than fighting it — the verify module supports three sources:

| Source | conda (local) | Docker on macOS | Docker on Linux |
|---|---|---|---|
| Uploaded image | ✅ | ✅ | ✅ |
| Uploaded video | ✅ | ✅ | ✅ |
| Server-side webcam (`cv2`) | ✅ | ❌ | ✅ with `--device /dev/video0` |
| **Browser webcam** (`getUserMedia` → frames over WebSocket) | ✅ | ✅ | ✅ |

Build the browser-webcam path (step 06). It works everywhere, needs no device mapping,
and gives you the same live bounding boxes. The server-side `cv2` path is a nice-to-have
for a future Linux box.

### 5.3 GPU

- **Apple Silicon, conda:** PyTorch MPS backend works. Set `device="mps"`. Expect roughly
  3–5× over CPU on a YOLO11n — good enough to iterate on a few thousand images.
- **Apple Silicon, Docker:** CPU only. MPS is not available in the Linux VM. Train locally
  in conda; use the container for serving and reproducibility.
- **NVIDIA, Docker:** works via `nvidia-container-toolkit`; the compose file gets a `gpu`
  profile. Leave the hook in, don't build it until you have the hardware.

Practical read: **conda is your training path on this Mac, Docker is your deployment
path.** Both run the same code; the compose file just won't be the fast one here.

### 5.4 Faces are a legal question, not only a technical one

Face *recognition* (matching identity against a gallery) is regulated in a way object
detection is not — BIPA in Illinois, GDPR Art. 9 in the EU, and comparable rules
elsewhere treat face templates as biometric data with consent requirements. For a
personal project on your own photos this is your call to make; I'm flagging it because
it's the kind of thing that's invisible until it isn't. Keep the face gallery in a
separate SQLite table (it already is) so it can be dropped wholesale.

Also note: InsightFace's pretrained model weights are licensed for **non-commercial
research use**. Fine for personal use; not fine to ship in a product.

### 5.5 Scale expectations

This design is comfortable to roughly **50k images and 500k annotations** on SQLite.
Past that, you'd want Postgres and a real object store. You will not get there by hand-
annotating, which is the honest bound on a personal project — expect 200–600 boxes per
hour of human effort, which is exactly why step 07 (pre-labelling) matters more than it
looks.

## 6. Repo layout

```
machine-id/
├── app/
│   ├── main.py                 # FastAPI app, mounts routers + static
│   ├── config.py               # pydantic-settings
│   ├── db.py                   # engine, session, create_all
│   ├── models.py               # SQLAlchemy tables
│   ├── schemas.py              # pydantic request/response
│   ├── routers/
│   │   ├── projects.py  labels.py   images.py    annotations.py
│   │   └── datasets.py  training.py inference.py faces.py
│   ├── services/
│   │   ├── ingest.py           # scan folder → Image rows
│   │   ├── dataset_export.py   # DB → YOLO layout + data.yaml
│   │   ├── trainer.py          # Trainer protocol + Ultralytics impl
│   │   ├── exporter.py         # → ONNX, parity check
│   │   ├── runtime.py          # onnxruntime session, letterbox, NMS
│   │   ├── prelabel.py         # model → draft annotations
│   │   ├── face_engine.py      # detect → align → embed → match
│   │   └── jobs.py             # subprocess runner + SSE log tail
│   ├── static/                 # app.js, annotate.js, verify.js, style.css
│   └── templates/              # index, annotate, verify, models (Jinja2)
├── data/                       # gitignored; the only writable path
│   ├── images/  datasets/  runs/  models/  app.db
├── docker/
│   ├── Dockerfile  entrypoint.sh
├── tests/
├── docs/
├── environment.yml
├── pyproject.toml
├── docker-compose.yml
└── .env.example
```

## 7. Build order

| Step | Doc | Outcome |
|---|---|---|
| 00 | [scaffold](docs/steps/00-scaffold.md) | App boots, DB exists, `/health` returns 200 |
| 01 | [projects, labels, ingest](docs/steps/01-projects-labels-ingest.md) | Point at a folder, images appear in DB |
| 02 | [annotation UI](docs/steps/02-annotation-ui.md) | Draw boxes with the keyboard, saved to DB |
| 03 | [dataset export](docs/steps/03-dataset-export.md) | Valid YOLO dataset on disk |
| 04 | [training](docs/steps/04-training.md) | Background training with live logs |
| 05 | [ONNX export](docs/steps/05-export-onnx.md) | Portable model folder + parity check |
| 06 | [verify playground](docs/steps/06-inference-playground.md) | Image/video/webcam with a threshold slider |
| 07 | [pre-labelling loop](docs/steps/07-prelabel-loop.md) | Model drafts boxes, you correct them |
| 08 | [faces](docs/steps/08-faces.md) | Detection + identity gallery |
| 09 | [Docker](docs/steps/09-docker.md) | `docker compose up` reproduces everything |
| 10 | [integration](docs/integration.md) | Model running inside other software |

Steps 00–06 are the spine. 07 is the highest-leverage addition. 08 and 09 are independent
of each other and can be done in either order.
