import asyncio
import json
import re
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from sqlalchemy import select, update
from sqlalchemy.orm import Session

from app.config import settings
from app.db import get_db
from app.models import Dataset, Job, Model, Project
from app.schemas import AugmentPreviewOut, ExportOnnxRequest, JobOut, ModelOut, TrainRequest
from app.services.augment_preview import build_augment_preview
from app.services.jobs import JobConflict, cancel_job, job_status, reconcile_jobs, start_job
from app.services.trainer import KNOWN_MODELS

router = APIRouter(tags=["training"])

LOG_POLL_SECONDS = 0.4
ANSI_ESCAPE = re.compile(r"\x1b\[[0-9;?]*[a-zA-Z]")


def clean_log_line(line: str) -> str:
    """Ultralytics draws progress bars with carriage returns and colours them with ANSI
    escapes. Rendered verbatim in a <pre> that is unreadable, so keep only what a terminal
    would have left visible: the last segment after the final \r, minus the escapes."""
    return ANSI_ESCAPE.sub("", line.rstrip("\n").split("\r")[-1]).rstrip()


PLOT_FILES = {"results.png", "confusion_matrix.png", "confusion_matrix_normalized.png",
              "results.csv", "labels.jpg", "train_batch0.jpg", "val_batch0_pred.jpg"}


def _project_or_404(db: Session, project_id: int) -> Project:
    project = db.get(Project, project_id)
    if not project:
        raise HTTPException(404, "project not found")
    return project


@router.get("/api/train/models")
def available_models():
    """What the 'model' dropdown offers, and which are already on disk."""
    return [
        {"name": name, "cached": (settings.pretrained_dir / name).is_file()}
        for name in KNOWN_MODELS
    ]


@router.post("/api/projects/{project_id}/train", response_model=JobOut)
def start_training(project_id: int, payload: TrainRequest, db: Session = Depends(get_db)):
    project = _project_or_404(db, project_id)

    if payload.dataset_id is None:
        dataset = db.scalars(
            select(Dataset).where(Dataset.project_id == project.id).order_by(Dataset.id.desc())
        ).first()
        if dataset is None:
            raise HTTPException(400, "no dataset exported yet — run a dataset export first")
    else:
        dataset = db.get(Dataset, payload.dataset_id)
        if dataset is None or dataset.project_id != project.id:
            raise HTTPException(404, "dataset not found in this project")

    if not (settings.data_path(dataset.dir_path) / "data.yaml").is_file():
        raise HTTPException(400, f"dataset {dataset.id} is missing data.yaml on disk")

    params = {"dataset_id": dataset.id, "config": payload.config.model_dump()}
    try:
        return start_job(db, "train", project.id, params)
    except JobConflict as exc:
        raise HTTPException(409, str(exc)) from exc


@router.get("/api/projects/{project_id}/jobs", response_model=list[JobOut])
def list_jobs(project_id: int, limit: int = 20, db: Session = Depends(get_db)):
    reconcile_jobs(db)
    return db.scalars(
        select(Job).where(Job.project_id == project_id).order_by(Job.id.desc()).limit(limit)
    ).all()


@router.get("/api/jobs/{job_id}", response_model=JobOut)
def get_job(job_id: int, db: Session = Depends(get_db)):
    reconcile_jobs(db)
    job = db.get(Job, job_id)
    if not job:
        raise HTTPException(404, "job not found")
    db.refresh(job)
    return job


@router.post("/api/jobs/{job_id}/cancel", response_model=JobOut)
def cancel(job_id: int, db: Session = Depends(get_db)):
    job = db.get(Job, job_id)
    if not job:
        raise HTTPException(404, "job not found")
    return cancel_job(db, job)


@router.get("/api/jobs/{job_id}/logs")
async def stream_logs(job_id: int, db: Session = Depends(get_db)):
    """Server-Sent Events, not WebSockets — one-directional, and the browser reconnects
    on its own. Always replayed from the top of the file, which is what makes closing and
    reopening the page reattach instead of showing a log that starts mid-epoch."""
    job = db.get(Job, job_id)
    if not job:
        raise HTTPException(404, "job not found")
    log_path = Path(job.log_path) if job.log_path else None
    # This response stays open for the length of the run. Close the read transaction now
    # rather than pinning a WAL snapshot for an hour — the generator below re-reads status
    # on its own short-lived sessions anyway.
    db.rollback()

    async def gen():
        if log_path is None or not log_path.is_file():
            yield "event: done\ndata: {}\n\n"
            return
        with open(log_path, errors="replace") as f:
            # readline() at EOF hands back whatever has been written so far, newline or
            # not. Emitting that immediately would split one log line across two events,
            # so hold an unterminated fragment until its newline arrives.
            pending = ""
            while True:
                chunk = f.readline()
                if chunk:
                    pending += chunk
                    if pending.endswith("\n"):
                        yield f"data: {clean_log_line(pending)}\n\n"
                        pending = ""
                    continue
                status = job_status(job_id)              # fresh session; see services/jobs
                if status is None or status in ("done", "failed", "cancelled"):
                    # One last pass: the worker's final lines can land after its DB commit.
                    for tail in f.readlines():
                        pending += tail
                        if pending.endswith("\n"):
                            yield f"data: {clean_log_line(pending)}\n\n"
                            pending = ""
                    if pending:                          # last line, never newline-terminated
                        yield f"data: {clean_log_line(pending)}\n\n"
                    yield f"event: done\ndata: {json.dumps({'status': status})}\n\n"
                    return
                await asyncio.sleep(LOG_POLL_SECONDS)

    return StreamingResponse(
        gen(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# --- model registry (§8) ---------------------------------------------------------

@router.get("/api/projects/{project_id}/models", response_model=list[ModelOut])
def list_models(project_id: int, db: Session = Depends(get_db)):
    return db.scalars(
        select(Model).where(Model.project_id == project_id).order_by(Model.created_at.desc(),
                                                                     Model.id.desc())
    ).all()


@router.get("/api/models/{model_id}", response_model=ModelOut)
def get_model(model_id: int, db: Session = Depends(get_db)):
    model = db.get(Model, model_id)
    if not model:
        raise HTTPException(404, "model not found")
    return model


@router.post("/api/models/{model_id}/activate", response_model=ModelOut)
def activate_model(model_id: int, db: Session = Depends(get_db)):
    """One active model per project — the single answer to 'which model is current' that
    steps 06 and 07 default to, instead of silently picking the most recent run."""
    model = db.get(Model, model_id)
    if not model:
        raise HTTPException(404, "model not found")
    db.execute(
        update(Model).where(Model.project_id == model.project_id).values(is_active=False)
    )
    model.is_active = True
    db.commit()
    db.refresh(model)
    return model


@router.post("/api/models/{model_id}/export-onnx", response_model=JobOut)
def export_onnx_endpoint(model_id: int, payload: ExportOnnxRequest, db: Session = Depends(get_db)):
    """Step 05: turn best.pt into model.onnx + parity-checked metadata.json, in the same
    model folder training already wrote. Runs as a job so its progress streams like a
    training run does."""
    model = db.get(Model, model_id)
    if not model:
        raise HTTPException(404, "model not found")
    if not (settings.data_path(model.dir_path) / "best.pt").is_file():
        raise HTTPException(400, f"model {model_id} has no weights on disk")

    params = {"model_id": model.id, **payload.model_dump()}
    try:
        return start_job(db, "export", model.project_id, params)
    except JobConflict as exc:
        raise HTTPException(409, str(exc)) from exc


@router.get("/api/models/{model_id}/plots/{name}")
def model_plot(model_id: int, name: str, db: Session = Depends(get_db)):
    """Ultralytics writes results.png / confusion_matrix.png / results.csv into the run
    directory; the confusion matrix in particular tells you *which* class is failing."""
    if name not in PLOT_FILES:
        raise HTTPException(400, "not a plot this endpoint serves")
    model = db.get(Model, model_id)
    if not model or not model.run_dir:
        raise HTTPException(404, "model or run directory not found")
    path = (settings.data_path(model.run_dir) / name).resolve()
    if not path.is_relative_to(settings.runs_dir.resolve()):
        raise HTTPException(400, "path escape")
    if not path.is_file():
        raise HTTPException(404, f"{name} not in this run")
    return FileResponse(path)


# --- augmentation preview (§7) ---------------------------------------------------

@router.get("/api/projects/{project_id}/augment-preview", response_model=AugmentPreviewOut)
def augment_preview(
    project_id: int,
    n: int = 6,
    imgsz: int = 640,
    fliplr: float = 0.5,
    db: Session = Depends(get_db),
):
    """Sanity-check the pipeline before spending an hour training: a mirrored forklift is
    still a forklift, a mirrored '7' is not (§7)."""
    project = _project_or_404(db, project_id)
    try:
        return build_augment_preview(db, project, n=max(1, min(n, 12)),
                                     imgsz=imgsz, fliplr=fliplr)
    except RuntimeError as exc:
        raise HTTPException(400, str(exc)) from exc
