# Machine ID — annotate → train → export → verify

A single, self-hosted Python app for building object-detection models from your own
photos, and for putting the resulting model to work in other systems. One FastAPI
process serves the whole thing: annotation UI, training job runner, inference
playground, and a small JSON API — see [ARCHITECTURE.md](ARCHITECTURE.md) for why.

The loop it implements:

```
folder of images
      │
      ▼
[1] Ingest ──► [2] Annotate (bounding box + preset label, keyboard driven)
                          │
                          ▼
                  [3] Export dataset (YOLO layout, deterministic splits)
                          │
                          ▼
                  [4] Train (background job, live logs)
                          │
                          ▼
                  [5] Export ONNX (+ OpenVINO / CoreML)  ──► other software
                          │
                          ▼
                  [6] Verify (image / video / webcam, adjustable confidence)
                          │
                          └──► [7] Pre-label the next batch, correct, retrain
```

Two extra modules sit on the same spine:

- **[8] Faces** — face *detection* is just another class; face *recognition* (who is this)
  is a separate detect → align → embed → nearest-neighbour pipeline with its own gallery.
- **[9] Packaging** — one `conda activate` path and one `docker compose` path, same code.

This document is about running it. For *why* it's built this way, read
[ARCHITECTURE.md](ARCHITECTURE.md); for how each step was implemented, see
[docs/steps/](docs/steps/).

## Which path to use

| Task | conda | Docker |
|---|---|---|
| Annotating, browsing models | ✅ | ✅ |
| Training on Apple Silicon | ✅ MPS, 3–5× faster | ⚠️ CPU only |
| Training on an NVIDIA box | ✅ | ✅ `--profile gpu` |
| Deploying to a Linux server | — | ✅ |
| Reproducible, pinned deps | ⚠️ drifts | ✅ |

Practical read: **conda is the training path, Docker is the deployment path.** Both run
the same code against the same `data/` directory — just not both against the same copy
of it at the same time (see [ARCHITECTURE §5.6](ARCHITECTURE.md#56-never-run-conda-and-docker-against-the-same-data-at-the-same-time)).

## 1. Run with conda

Requires [Miniconda/Anaconda](https://docs.conda.io/en/latest/miniconda.html) and Python
3.11 (pinned in `environment.yml` — the CV/torch ecosystem lags, and 3.11 is the version
every wheel you need exists prebuilt for).

```bash
conda env create -f environment.yml
conda activate boxforge
cp .env.example .env          # optional — the defaults already match
uvicorn app.main:app --reload
```

Open **http://localhost:8000**. The database and `data/` layout are created
automatically on first boot — nothing to migrate or seed by hand.

**macOS only, once, before your first training run:** conda's OpenCV and pip's torch
each ship their own copy of the OpenMP runtime, and loading both crashes training with
either a loud `OMP: Error #15` or a silent `SIGSEGV`. Fix (re-run after any
`pip install`/upgrade of torch, which restores its bundled copy):

```bash
ln -sf "$CONDA_PREFIX/lib/libomp.dylib" \
       "$CONDA_PREFIX/lib/python3.11/site-packages/torch/lib/libomp.dylib"
```

See `environment.yml` for the full explanation.

## 2. Run with Docker

Requires Docker Desktop (or the Docker Engine + Compose plugin on Linux).

```bash
docker compose up                    # serve only — small image, CPU inference
docker compose --profile train up    # + a training container on :8001
docker compose --profile gpu up      # + an NVIDIA-backed trainer on :8002
```

Open **http://localhost:8000** (the `serve` app) — or **:8001** / **:8002** for a
training container's own UI, if you started one. Everything under `./data` on the host
is bind-mounted into the container, so it's the same projects/models/images either way.

The default image builds from `requirements-serve.txt` (fastapi, onnxruntime, opencv,
sqlalchemy — no torch) and comes in under 1GB. `--profile train`/`--profile gpu` build a
second image on top with `requirements-train.txt` (adds `ultralytics`, several GB
larger) — you don't pay for that unless you ask for it.

```bash
docker compose down          # stop everything; ./data is untouched
docker image ls              # confirm the serve image's size
```

## 3. Using the app

Start at **http://localhost:8000**.

1. **Home (`/`)** — create a project, add labels (each gets a colour and a fixed
   `class_index` — never reused, even if you delete the label later), and ingest a
   folder of images by typing its path. Copies by default (bakes in EXIF rotation and
   dedupes by content hash); uncheck "copy" to symlink instead if the folder is huge
   and you don't need that normalisation.
2. **Annotate (`/annotate/{project}`)** — draw boxes with the mouse, `1`–`9` to switch
   labels, `A`/`D` or arrow keys to move between images, `S` to skip, `⌫` to delete,
   `Ctrl/Cmd+Z` to undo. Saves automatically when you move on. The side panel has a
   review-order dropdown (ingest order / most uncertain / zero detections / most boxes)
   — that only does something once step 07 below has drafted some boxes for you.
3. **Train (`/train/{project}`)** — export a YOLO-format dataset (this runs a
   preflight check: class balance, box sizes, empty validation split — read the
   warnings before forcing an export through them), preview augmentations, then start a
   training run. Logs stream live; cancel is always available. Once a run finishes,
   pick a model as "active" and export it to ONNX — that `model.onnx` + `classes.json`
   + `metadata.json` folder under `data/models/<id>/` is the portable artifact other
   software can load (see [docs/integration.md](docs/integration.md)).
4. **Verify (`/verify/{project}`)** — run the *exported ONNX* (never the raw
   checkpoint — this is what proves the artifact you'd hand to another system actually
   works) against an uploaded image, an uploaded video, or your browser's webcam, with
   live confidence/IoU sliders.
5. **Pre-label (`/prelabel/{project}`)** — once you have a decent model, draft boxes on
   the next batch of pending images instead of hand-drawing them. Preview the count
   first, run the batch, then either correct drafts one-by-one in the annotator (dashed
   = unreviewed) or sweep through the grid view (space = accept, `x` = reject) for
   images the model already got right. `undo` removes exactly one batch's drafts.
6. **Faces (`/faces`)** — a separate pipeline from the rest: add an identity, upload
   5–10 reference photos (each must show exactly one face), then test against an image
   or webcam. The first visit needs a one-time ~190MB download of the detection/
   embedding weights (button on the page). Read
   [ARCHITECTURE §5.4](ARCHITECTURE.md#54-faces-are-a-legal-question-not-only-a-technical-one)
   before pointing this at anyone but yourself.

## 4. Tests

```bash
pytest
```

A few tests exercise the real pipeline end to end (downloading a pretrained checkpoint
or the face models, running actual inference) and skip themselves automatically if
you're offline. Everything else is fast and hermetic — each test gets an isolated
`data/` and SQLite file via `tests/conftest.py`, so this never touches your real data.

## Configuration

Environment variables (`MID_` prefix), read from `.env` or the shell — see
`app/config.py`:

| Variable | Default | Meaning |
|---|---|---|
| `MID_DATA_DIR` | `data` | Where everything lives: images, datasets, runs, models, `app.db`. The one directory Docker's volume mount needs to cover. |
| `MID_DEVICE` | `auto` | Training device: `auto` picks CUDA → MPS → CPU; override with `cpu`, `mps`, or `0` (cuda:0). |
| `MID_DEFAULT_CONF` | `0.25` | Default confidence threshold shown in the verify playground. |

## Where things live

```
data/
  app.db                        SQLite — projects, labels, images, annotations, jobs, models, faces
  images/<project>/             ingested originals (or hashed copies)
  datasets/<project>/<ts>/      exported YOLO layout: images/, labels/, data.yaml, manifest.json
  runs/<job_id>/                training/export/prelabel job logs and Ultralytics output
  models/<model_id>/            best.pt, model.onnx, classes.json, metadata.json — the portable artifact
  pretrained/                   downloaded checkpoints (yolo11n.pt, SCRFD/ArcFace) — fetched on demand
  faces/<identity_id>/          reference photos for the face gallery
```

Nothing is ever written outside `data/` — that's what makes the Docker bind mount
trivial, and what makes it safe to `rm -rf data/` for a clean slate.

## Where to go deeper

| Doc | What it is |
|---|---|
| [ARCHITECTURE.md](ARCHITECTURE.md) | The stack, why each piece, the data model, the risks to know up front |
| [docs/steps/](docs/steps/) | The 10 build steps this app was implemented in, each with acceptance criteria |
| [docs/integration.md](docs/integration.md) | How the exported model plugs into Frigate, DeepStream, OpenVINO, VMS platforms, plain Python |
| [docs/comparison.md](docs/comparison.md) | How this stacks up against CVAT, Label Studio, Roboflow, Labelbox and friends |

## Ground rules

1. **The verify module loads the exported ONNX, not the PyTorch checkpoint.** You test
   what you ship, not what you trained.
2. **Data lives under `data/`, code never writes outside it.**
3. **A draft box is never training data on its own say-so.** Pre-labelled images stay
   `pending` until a human looks at them (step 07).
4. **Don't run conda and Docker against the same `data/` at the same time** — see
   ARCHITECTURE §5.6.
