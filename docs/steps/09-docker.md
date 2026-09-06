# Step 09 — Packaging

**Goal:** the same codebase runs two ways — `conda activate` for development and training
on your Mac, `docker compose up` for a reproducible deployment.

**Depends on:** step 06 working locally. Independent of step 08.

Read ARCHITECTURE §5.2 and §5.3 before deciding what to run where. Short version on this
machine: **train in conda (MPS), serve in Docker (CPU).**

## 1. Dockerfile

Multi-stage, so the runtime image doesn't carry build tools:

```dockerfile
# docker/Dockerfile
FROM python:3.11-slim AS base
ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
RUN apt-get update && apt-get install -y --no-install-recommends \
      libgl1 libglib2.0-0 ffmpeg \
    && rm -rf /var/lib/apt/lists/*
# libgl1 + libglib2.0-0: OpenCV's runtime deps. Without them, `import cv2` fails
# inside slim images with a confusing ImportError about libGL.

# ---------- serve: onnxruntime only, no torch ----------
FROM base AS serve
WORKDIR /app
COPY requirements-serve.txt .
RUN pip install -r requirements-serve.txt
COPY app ./app
EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]

# ---------- train: adds torch + ultralytics ----------
FROM serve AS train
COPY requirements-train.txt .
RUN pip install -r requirements-train.txt
```

Two requirement files is the point. `requirements-serve.txt` (fastapi, onnxruntime, opencv,
sqlalchemy) yields roughly a 700MB image. Adding torch + ultralytics takes it past 3GB.
Most of the time you only need to serve, so don't pay for training in the default image.

Use `python:3.11-slim`, not `alpine` — musl breaks numpy/opencv wheels and you'll end up
compiling from source for no benefit.

## 2. Compose with profiles

```yaml
# docker-compose.yml
services:
  app:
    build: {context: ., dockerfile: docker/Dockerfile, target: serve}
    ports: ["8000:8000"]
    volumes:
      - ./data:/app/data          # THE mount. All state lives here.
    environment:
      MID_DATA_DIR: /app/data
      MID_DEVICE: cpu
    restart: unless-stopped

  trainer:
    profiles: ["train"]
    build: {context: ., dockerfile: docker/Dockerfile, target: train}
    volumes: ["./data:/app/data"]
    environment: {MID_DATA_DIR: /app/data, MID_DEVICE: cpu}
    command: ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
    ports: ["8001:8000"]

  trainer-gpu:
    profiles: ["gpu"]             # NVIDIA hosts only; no-op on macOS
    build: {context: ., dockerfile: docker/Dockerfile, target: train}
    volumes: ["./data:/app/data"]
    environment: {MID_DATA_DIR: /app/data, MID_DEVICE: "cuda:0"}
    deploy:
      resources:
        reservations:
          devices: [{driver: nvidia, count: 1, capabilities: [gpu]}]
```

```bash
docker compose up                        # serve only, small image
docker compose --profile train up        # + training container
docker compose --profile gpu up          # NVIDIA host
```

Because everything writes under `data/` (ARCHITECTURE §1), one bind mount is the entire
persistence story: models trained in conda are visible to the container and vice versa.

## 3. Paths across the boundary

The one thing that breaks: absolute paths stored in the database. `data.yaml` from step 03
contains an absolute `path:`, and `/Users/yp/Projects/...` does not exist in the container.

Store paths **relative to `data_dir`** in the DB, and resolve through `settings` at use
time. Regenerate `data.yaml` at training start rather than reusing the exported one:

```python
def materialise_data_yaml(dataset_rel: str) -> Path:
    root = (settings.datasets_dir / dataset_rel).resolve()
    ...  # write data.yaml with `path: {root}` computed now, in this environment
```

This is also why step 03 copies images instead of symlinking — a symlink to
`/Users/yp/...` is dead inside the container.

## 4. Non-root user

```dockerfile
RUN useradd -m -u 1000 app && chown -R app:app /app
USER app
```

Bind-mounted `data/` must be writable by uid 1000. On macOS Docker Desktop handles the
mapping; on Linux, `chown -R 1000:1000 data/` once.

## 5. Health check

```yaml
healthcheck:
  test: ["CMD", "python", "-c",
         "import urllib.request;urllib.request.urlopen('http://localhost:8000/health')"]
  interval: 30s
  timeout: 3s
  retries: 3
  start_period: 20s
```

`curl` isn't in slim images; using Python avoids installing it just for this.

## 6. `.dockerignore`

```
data/
.git/
**/__pycache__/
*.pt
*.onnx
.venv/
```

Without this, the build context includes your entire dataset and every build crawls.

## 7. Which path for what

| Task | conda | Docker |
|---|---|---|
| Annotating | ✅ either | ✅ either |
| Training on this Mac | ✅ MPS, 3–5× faster | ⚠️ CPU only |
| Webcam (browser capture) | ✅ | ✅ |
| Webcam (server `cv2`) | ✅ | ❌ on macOS |
| Deploying to a Linux box | — | ✅ |
| Reproducibility | ⚠️ drifts | ✅ pinned |

## Acceptance criteria

- `docker compose up` → `/health` responds, UI loads, existing models are listed.
- A model trained in conda loads and runs in the container (no absolute-path failures).
- Browser webcam works against the containerised app.
- `docker compose down && up` → all projects, images and annotations survive.
- Serve image is under ~1GB; `docker image ls` confirms.
- `import cv2` works in the container (this is the libGL check).
