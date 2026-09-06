from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.db import get_db
from app.models import Label, Project
from app.schemas import LabelCreate, LabelOut
from app.services.labels import next_class_index

router = APIRouter(tags=["labels"])


@router.get("/api/projects/{project_id}/labels", response_model=list[LabelOut])
def list_labels(project_id: int, db: Session = Depends(get_db)):
    return db.scalars(
        select(Label).where(Label.project_id == project_id).order_by(Label.class_index)
    ).all()


@router.post("/api/projects/{project_id}/labels", response_model=LabelOut)
def create_label(project_id: int, payload: LabelCreate, db: Session = Depends(get_db)):
    project = db.get(Project, project_id)
    if not project:
        raise HTTPException(404, "project not found")
    if db.scalar(select(Label).where(Label.project_id == project_id, Label.name == payload.name)):
        raise HTTPException(400, "a label with this name already exists")
    label = Label(
        project_id=project_id,
        name=payload.name,
        color=payload.color or "#ff3b30",
        class_index=next_class_index(db, project_id),
    )
    db.add(label)
    db.commit()
    db.refresh(label)
    return label


@router.delete("/api/labels/{label_id}", status_code=204)
def delete_label(label_id: int, db: Session = Depends(get_db)):
    label = db.get(Label, label_id)
    if not label:
        raise HTTPException(404, "label not found")
    db.delete(label)          # class_index is retired, never reused
    db.commit()
