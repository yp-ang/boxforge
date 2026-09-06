"""Standalone job entry point: `python -m app.worker <job_id>`.

Opens its own DB session, reads the job row, dispatches on job.type, writes result_json,
sets a terminal status and exits. Everything it prints lands in the job's log file, which
is what the SSE stream tails. It never imports the FastAPI app.
"""
from __future__ import annotations

import json
import os
import sys
import traceback
from datetime import datetime
from pathlib import Path

import cv2

from app.config import settings
from app.db import SessionLocal, init_db
from app.models import Dataset, Job, Model, Project
from app.services.dataset_export import materialise_data_yaml
from app.services.exporter import (
    OnnxDetector,
    TorchDetector,
    build_export_metadata,
    check_onnx_model,
    export_onnx,
    parity_check,
)
from app.services.prelabel import DEFAULT_PRELABEL_CONF, prelabel
from app.services.runtime import OnnxDetector as RuntimeOnnxDetector
from app.services.runtime import draw_detections
from app.services.trainer import TrainConfig, UltralyticsTrainer, write_model_folder

PARITY_SAMPLE_IMAGES = 5


def log(message: str) -> None:
    print(f"[{datetime.now():%H:%M:%S}] {message}", flush=True)


def run_train(db, job: Job) -> dict:
    params = json.loads(job.params_json or "{}")
    project = db.get(Project, job.project_id)
    if project is None:
        raise RuntimeError(f"project {job.project_id} no longer exists")

    dataset = db.get(Dataset, params["dataset_id"])
    if dataset is None or dataset.project_id != project.id:
        raise RuntimeError(f"dataset {params.get('dataset_id')} not found in this project")

    dataset_dir = settings.data_path(dataset.dir_path)
    if not dataset_dir.is_dir():
        raise RuntimeError(f"dataset directory missing: {dataset_dir}")
    # Step 09 §3: never trust the path: key baked into data.yaml at export time — this
    # process may be running in a different environment (conda vs. the Docker
    # container's /app/data) than the one that exported it.
    dataset_yaml = materialise_data_yaml(dataset_dir)

    cfg = TrainConfig(**params["config"])
    run_dir = settings.runs_dir / str(job.id) / "train"

    manifest = json.loads(dataset.manifest_json)
    names: dict[str, str] = {str(k): v for k, v in manifest["names"].items()}

    log(f"dataset {dataset.id}: {dataset.n_train} train / {dataset.n_val} val, "
        f"classes {list(names.values())}")
    log(f"run dir {run_dir}")

    result = UltralyticsTrainer().train(dataset_yaml, cfg, run_dir)

    # Training just held this session's read transaction open for the length of the run.
    # Drop it before writing anything: it pins a WAL snapshot, and every row read through
    # it from here on would be the hour-old copy rather than what the web process wrote.
    db.rollback()

    if not result.weights.is_file():
        raise RuntimeError(f"training finished but produced no weights at {result.weights}")

    model = Model(
        project_id=project.id,
        job_id=job.id,
        dataset_id=dataset.id,
        name=f"{Path(cfg.model).stem}-job{job.id}",
        dir_path="",                       # needs the row id; filled in below
        run_dir=settings.rel_data_path(result.run_dir),
        classes_json=json.dumps(names),
        metrics_json=json.dumps(result.metrics),
    )
    db.add(model)
    db.commit()
    db.refresh(model)

    model_dir = settings.models_dir / str(model.id)
    write_model_folder(
        model_dir,
        result.weights,
        names,
        {
            "model_id": model.id,
            "job_id": job.id,
            "project": project.slug,
            "dataset_id": dataset.id,
            "imgsz": cfg.imgsz,
            "task": project.task,
            "trained_at": datetime.utcnow().isoformat(),
            "config": cfg.to_dict(),
            "metrics": result.metrics,
        },
    )
    model.dir_path = settings.rel_data_path(model_dir)
    db.commit()

    log(f"model {model.id} written to {model_dir}")
    for key in ("mAP50-95", "mAP50", "precision", "recall"):
        if key in result.metrics:
            log(f"  {key:>9} = {result.metrics[key]:.4f}")

    return {"model_id": model.id, "run_dir": str(result.run_dir),
            "weights": str(model_dir / "best.pt"), "metrics": result.metrics}


def _pick_parity_samples(dataset: Dataset | None, n: int) -> list[Path]:
    """Prefer val images — they were never trained on, so a pass there is the stronger
    claim — and fall back to train images for a dataset too small to have a val split."""
    if dataset is None:
        return []
    dataset_dir = settings.data_path(dataset.dir_path)
    for split in ("val", "train"):
        images = sorted((dataset_dir / "images" / split).glob("*"))
        if images:
            return images[:n]
    return []


def run_export(db, job: Job) -> dict:
    params = json.loads(job.params_json or "{}")
    model = db.get(Model, params["model_id"])
    if model is None:
        raise RuntimeError(f"model {params.get('model_id')} no longer exists")
    project = db.get(Project, model.project_id)

    model_dir = settings.data_path(model.dir_path)
    weights = model_dir / "best.pt"
    if not weights.is_file():
        raise RuntimeError(f"model {model.id} has no weights on disk at {weights}")

    metadata_path = model_dir / "metadata.json"
    existing = json.loads(metadata_path.read_text()) if metadata_path.is_file() else {}

    imgsz = params.get("imgsz") or existing.get("imgsz") or settings.default_imgsz
    opset = params.get("opset", 12)
    dynamic = params.get("dynamic", False)
    nms = params.get("nms", False)

    log(f"exporting model {model.id}: weights={weights} imgsz={imgsz} opset={opset} "
        f"dynamic={dynamic} nms={nms}")
    onnx_path = export_onnx(weights, model_dir, imgsz=imgsz, opset=opset,
                            dynamic=dynamic, nms=nms)
    check_onnx_model(onnx_path)
    log(f"model.onnx written to {onnx_path}, onnx.checker passed")

    dataset = db.get(Dataset, model.dataset_id) if model.dataset_id else None
    samples = _pick_parity_samples(dataset, PARITY_SAMPLE_IMAGES)
    parity = None
    if samples:
        log(f"running parity check on {len(samples)} sample image(s)")
        parity = parity_check(TorchDetector(weights, imgsz), OnnxDetector(onnx_path), samples)
        log(f"parity check {'PASSED' if parity.passed else 'FAILED'}")
        if not parity.passed:
            for img_result in parity.images:
                if not img_result.passed:
                    log(f"  {img_result.image}: {img_result.detail}")
    else:
        log("no dataset images available for a parity check — skipping")

    classes = json.loads(model.classes_json)
    metrics = json.loads(model.metrics_json or "{}")
    metadata = {
        **existing,
        **build_export_metadata(
            name=model.name,
            task=project.task if project else "detect",
            imgsz=imgsz, opset=opset, nms=nms,
            classes=classes, metrics=metrics,
            trained_at=existing.get("trained_at", datetime.utcnow().isoformat()),
            dataset=f"{project.slug}/{Path(dataset.dir_path).name}" if project and dataset
                    else existing.get("dataset"),
            framework_version=existing.get("framework_version", "ultralytics 8.x"),
            parity=parity,
        ),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2))

    # Same reasoning as run_train above: drop the read transaction the export just held
    # open before writing, and re-fetch rather than trust the identity-mapped copy.
    db.rollback()
    db.expire_all()
    model = db.get(Model, model.id)
    model.onnx_path = settings.rel_data_path(onnx_path)
    model.parity_status = None if parity is None else ("passed" if parity.passed else "failed")
    model.parity_json = json.dumps(parity.to_dict()) if parity else None
    db.commit()

    return {"model_id": model.id, "onnx_path": str(onnx_path),
            "parity_status": model.parity_status}


def run_predict_video(db, job: Job) -> dict:
    """Step 06: annotate an uploaded video against the exported ONNX model. Runs as a
    job for the same reason training does — a 30s clip at full resolution takes long
    enough that the request would otherwise time out, and this reuses the SSE log
    stream and cancel button step 04 already built."""
    params = json.loads(job.params_json or "{}")
    model = db.get(Model, params["model_id"])
    if model is None:
        raise RuntimeError(f"model {params.get('model_id')} no longer exists")
    if not model.onnx_path or not settings.data_path(model.onnx_path).is_file():
        raise RuntimeError(f"model {model.id} has no exported ONNX")

    src_path = Path(params["src_path"])
    stride = max(1, int(params.get("stride", 3)))
    conf = float(params.get("conf", 0.25))
    iou = float(params.get("iou", 0.45))

    out_dir = settings.runs_dir / str(job.id) / "predict"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_video = out_dir / "annotated.mp4"
    out_json = out_dir / "detections.json"

    cap = cv2.VideoCapture(str(src_path))
    if not cap.isOpened():
        raise RuntimeError(f"could not open uploaded video: {src_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    detector = RuntimeOnnxDetector(settings.data_path(model.dir_path))
    # mp4v is what OpenCV ships with everywhere (step 06 §3) — not the most efficient
    # codec, but the one that will not make you debug an FFmpeg build.
    writer = cv2.VideoWriter(str(out_video), cv2.VideoWriter_fourcc(*"mp4v"), fps,
                             (width, height))

    log(f"predicting {src_path.name}: {total} frames @ {fps:.1f}fps, stride={stride}, "
        f"conf={conf}, iou={iou}")

    per_frame: list[dict] = []
    last_dets: list = []
    idx = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            # Infer every stride-th frame and reuse the last boxes in between — 3x
            # faster and visually identical for review purposes (step 06 §3).
            if idx % stride == 0:
                last_dets = detector.predict(frame, conf=conf, iou=iou)
                per_frame.append({"frame": idx, "detections": [d.as_dict() for d in last_dets]})
            draw_detections(frame, last_dets)
            writer.write(frame)
            idx += 1
            if idx % 100 == 0:
                log(f"  frame {idx}/{total}")
    finally:
        cap.release()
        writer.release()

    out_json.write_text(json.dumps(
        {"fps": fps, "width": width, "height": height, "stride": stride, "frames": per_frame},
        indent=2,
    ))
    src_path.unlink(missing_ok=True)   # scratch upload, not part of the model folder

    log(f"annotated video written to {out_video} ({idx} frames)")
    return {"model_id": model.id, "video": str(out_video), "frames": idx}


def run_prelabel(db, job: Job) -> dict:
    """Step 07: draft boxes on pending, un-annotated images with the active/selected
    model. Runs as a job for the same reason predict_video does — a few hundred images
    through onnxruntime takes long enough to want the SSE log stream and cancel button
    step 04 already built, rather than tying up a request."""
    params = json.loads(job.params_json or "{}")
    project = db.get(Project, job.project_id)
    if project is None:
        raise RuntimeError(f"project {job.project_id} no longer exists")

    model = db.get(Model, params["model_id"])
    if model is None:
        raise RuntimeError(f"model {params.get('model_id')} no longer exists")
    if not model.onnx_path or not Path(model.onnx_path).is_file():
        raise RuntimeError(f"model {model.id} has no exported ONNX — run the step 05 export first")

    conf = float(params.get("conf", DEFAULT_PRELABEL_CONF))
    limit = int(params.get("limit", 500))

    detector = RuntimeOnnxDetector(settings.data_path(model.dir_path))

    def read_image(rel_path: str):
        path = settings.images_dir / rel_path
        frame = cv2.imread(str(path))
        if frame is None:
            raise RuntimeError(f"could not read image {path}")
        return frame

    log(f"pre-labelling project {project.id} with model {model.id} @ conf={conf}, "
        f"up to {limit} images")
    result = prelabel(db, project, detector, conf, limit, read_image)
    log(f"drafted {result['n_boxes']} boxes across {result['n_images']} images")
    if result["unmapped_classes"]:
        log(f"skipped detections for classes not in this project's labels: "
            f"{result['unmapped_classes']}")

    return {"model_id": model.id, "conf": conf, **result}


HANDLERS = {"train": run_train, "export": run_export, "predict_video": run_predict_video,
            "prelabel": run_prelabel}


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print("usage: python -m app.worker <job_id>", file=sys.stderr)
        return 2

    job_id = int(argv[1])
    init_db()

    # Ultralytics writes stray files (downloaded checkpoints, its settings file) relative
    # to the process CWD. Everything this worker uses is an absolute path under data/, so
    # parking the CWD there keeps those strays inside the one directory Docker mounts
    # (README ground rule 4).
    settings.pretrained_dir.mkdir(parents=True, exist_ok=True)
    os.chdir(settings.pretrained_dir)

    with SessionLocal() as db:
        job = db.get(Job, job_id)
        if job is None:
            print(f"job {job_id} not found", file=sys.stderr)
            return 2

        handler = HANDLERS.get(job.type)
        if handler is None:
            job.status, job.ended_at = "failed", datetime.utcnow()
            job.result_json = json.dumps({"error": f"unknown job type {job.type!r}"})
            db.commit()
            return 2

        log(f"job {job.id} ({job.type}) starting, pid {os.getpid()}")
        try:
            result = handler(db, job)
        except BaseException as exc:                    # SystemExit/KeyboardInterrupt too
            traceback.print_exc()
            sys.stderr.flush()
            db.rollback()
            db.expire_all()
            job = db.get(Job, job_id)
            # A cancel writes 'cancelled' from the web process and then signals us; don't
            # overwrite that verdict with 'failed' on the way out.
            if job is not None and job.status not in ("cancelled", "done"):
                job.status, job.ended_at = "failed", datetime.utcnow()
                job.result_json = json.dumps({"error": f"{type(exc).__name__}: {exc}"})
                db.commit()
            log("job failed")
            return 1

        # Re-read across a transaction boundary. Session.get() would otherwise hand back
        # the identity-mapped row loaded before training started, and a cancel that landed
        # while we were finishing would be silently overwritten with 'done'.
        db.rollback()
        db.expire_all()
        job = db.get(Job, job_id)
        if job is not None and job.status == "cancelled":
            log("job was cancelled after the work finished; leaving status as cancelled")
            return 0
        job.status, job.ended_at = "done", datetime.utcnow()
        job.result_json = json.dumps(result)
        db.commit()
        log(f"job {job.id} done")
        return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
