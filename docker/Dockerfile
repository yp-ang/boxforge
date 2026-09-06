# Multi-stage: the runtime image never carries build tools, and by default nobody pays
# for torch just to serve inference (step 09 §1).
FROM python:3.11-slim AS base
ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
RUN apt-get update && apt-get install -y --no-install-recommends \
      libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*
# libgl1 + libglib2.0-0: OpenCV's runtime deps. Without them, `import cv2` fails
# inside slim images with a confusing ImportError about libGL. Deliberately NOT the
# system ffmpeg package: opencv-python-headless already bundles its own ffmpeg plugin
# for cv2.VideoCapture/VideoWriter (step 06's video predict), and the apt package pulls
# in a large transitive dependency tree (~400MB) for codecs this app never calls
# directly — it blew the serve image well past the ~1GB target for zero benefit.

# ---------- serve: onnxruntime only, no torch ----------
FROM base AS serve
WORKDIR /app
RUN useradd -m -u 1000 app
COPY requirements-serve.txt .
RUN pip install -r requirements-serve.txt
COPY app ./app
RUN mkdir -p /app/data && chown -R app:app /app
USER app
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=3s --start-period=20s --retries=3 \
  CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"]
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]

# ---------- train: adds torch + ultralytics ----------
FROM serve AS train
USER root
COPY requirements-train.txt .
RUN pip install -r requirements-train.txt
USER app
