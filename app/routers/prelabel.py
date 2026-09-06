"""Step 07: pre-labelling loop — draft boxes with a trained model, review fast, retrain.
See docs/steps/07-prelabel-loop.md."""
from __future__ import annotations

import json
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.db import get_db
from app.models import Annotation, Image, Job, Model, Project
from app.schemas import (
    AnnotationOut,
    ImageOut,
    JobOut,
    PrelabelBatchImageOut,
    PrelabelBatchOut,
    PrelabelPreviewOut,
    PrelabelRequest,
    UndoPrelabelOut,
)
from app.services.jobs import JobConflict, start_job
from app.services.prelabel import count_eligible, undo_batch

router = APIRouter(tags=["prelabel"])


def _project_or_404(db: Session, project_id: int) -> Project:
    project = db.get(Project, project_id)
    if not project:
        raise HTTPException(404, "project not found")
    return project


def _exported_model_in_project(db: Session, project_id: int, model_id: int) -> Model:
    model = db.get(Model, model_id)
    if not model or model.project_id != project_id:
        raise HTTPException(404, "model not found in this project")
    if not model.onnx_path or not Path(model.onnx_path).is_file():
        raise HTTPException(400, f"model {model_id} has no exported ONNX — run the "
                                  "step 05 export first")
    return model


@router.get("/api/projects/{project_id}/prelabel-preview", response_model=PrelabelPreviewOut)
def prelabel_preview(
    project_id: int, model_id: int, conf: float = 0.4, limit: int = 500,
    db: Session = Depends(get_db),
):
    """Doc §5 guardrail: show a count before running anything."""
    _project_or_404(db, project_id)
    _exported_model_in_project(db, project_id, model_id)
    return PrelabelPreviewOut(eligible=count_eligible(db, project_id, limit),
                              model_id=model_id, conf=conf)


@router.post("/api/projects/{project_id}/prelabel", response_model=JobOut)
def start_prelabel(project_id: int, payload: PrelabelRequest, db: Session = Depends(get_db)):
    project = _project_or_404(db, project_id)
    _exported_model_in_project(db, project_id, payload.model_id)

    params = {"model_id": payload.model_id, "conf": payload.conf, "limit": payload.limit}
    try:
        return start_job(db, "prelabel", project.id, params)
    except JobConflict as exc:
        raise HTTPException(409, str(exc)) from exc


def _prelabel_job_or_404(db: Session, job_id: int) -> Job:
    job = db.get(Job, job_id)
    if not job or job.type != "prelabel":
        raise HTTPException(404, "prelabel job not found")
    return job


@router.post("/api/jobs/{job_id}/prelabel/undo", response_model=UndoPrelabelOut)
def undo_prelabel(job_id: int, db: Session = Depends(get_db)):
    """Doc §5: 'undo the drafts from v2' — scoped to exactly the image ids this job
    drafted, recorded in its result_json when the job finished."""
    job = _prelabel_job_or_404(db, job_id)
    if job.status != "done" or not job.result_json:
        raise HTTPException(400, f"job {job_id} has no completed prelabel batch to undo")
    image_ids = json.loads(job.result_json).get("image_ids", [])
    result = undo_batch(db, image_ids)
    return UndoPrelabelOut(**result)


# --- §6: bulk accept/reject grid review -------------------------------------------

@router.get("/api/projects/{project_id}/prelabel-batch/{job_id}", response_model=PrelabelBatchOut)
def prelabel_batch(project_id: int, job_id: int, db: Session = Depends(get_db)):
    _project_or_404(db, project_id)
    job = _prelabel_job_or_404(db, job_id)
    if job.project_id != project_id:
        raise HTTPException(404, "prelabel job not found in this project")
    if not job.result_json:
        raise HTTPException(400, f"job {job_id} has not finished yet")

    result = json.loads(job.result_json)
    image_ids = result.get("image_ids", [])
    # Only images still pending — one accepted/rejected out of the batch (or annotated
    # by hand since) simply drops out of the grid rather than erroring the whole batch.
    images = db.scalars(
        select(Image).where(Image.id.in_(image_ids), Image.status == "pending")
        .order_by(Image.id)
    ).all()

    items = []
    for img in images:
        annotations = db.scalars(
            select(Annotation).where(Annotation.image_id == img.id)
        ).all()
        items.append(PrelabelBatchImageOut(image=ImageOut.model_validate(img),
                                           annotations=[AnnotationOut.model_validate(a)
                                                        for a in annotations]))

    return PrelabelBatchOut(job_id=job.id, model_id=result.get("model_id"),
                            conf=result.get("conf", 0.4), images=items)


@router.post("/api/images/{image_id}/accept", response_model=ImageOut)
def accept_prelabel(image_id: int, db: Session = Depends(get_db)):
    """The model already nailed this one: promote its drafts to human-sourced boxes
    without opening the per-box annotator (doc §6)."""
    img = db.get(Image, image_id)
    if not img:
        raise HTTPException(404, "image not found")
    annotations = db.scalars(select(Annotation).where(Annotation.image_id == image_id)).all()
    for a in annotations:
        a.source = "human"
    img.status = "annotated"
    db.commit()
    db.refresh(img)
    return img


@router.post("/api/images/{image_id}/reject", response_model=ImageOut)
def reject_prelabel(image_id: int, db: Session = Depends(get_db)):
    """The drafts are wrong: clear them and leave the image pending for a from-scratch
    pass in the full annotator (doc §6)."""
    img = db.get(Image, image_id)
    if not img:
        raise HTTPException(404, "image not found")
    db.execute(
        Annotation.__table__.delete().where(Annotation.image_id == image_id,
                                            Annotation.source == "model")
    )
    img.status = "pending"
    img.uncertainty = None
    db.commit()
    db.refresh(img)
    return img
