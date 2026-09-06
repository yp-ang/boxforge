import re

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.db import get_db
from app.models import Annotation, Image, Label, Project
from app.schemas import ProjectCreate, ProjectOut, StatsOut

router = APIRouter(prefix="/api/projects", tags=["projects"])


def slugify(name: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", name.strip().lower()).strip("-")
    return slug or "project"


@router.post("", response_model=ProjectOut)
def create_project(payload: ProjectCreate, db: Session = Depends(get_db)):
    slug = slugify(payload.name)
    if db.scalar(select(Project).where(Project.slug == slug)):
        raise HTTPException(400, "a project with this name already exists")
    project = Project(name=payload.name, slug=slug)
    db.add(project)
    db.commit()
    db.refresh(project)
    return project


@router.get("", response_model=list[ProjectOut])
def list_projects(db: Session = Depends(get_db)):
    return db.scalars(select(Project).order_by(Project.id)).all()


@router.get("/{project_id}/stats", response_model=StatsOut)
def project_stats(project_id: int, db: Session = Depends(get_db)):
    project = db.get(Project, project_id)
    if not project:
        raise HTTPException(404, "project not found")

    def count(status: str) -> int:
        return db.scalar(
            select(func.count()).select_from(Image)
            .where(Image.project_id == project_id, Image.status == status)
        ) or 0

    total = db.scalar(
        select(func.count()).select_from(Image).where(Image.project_id == project_id)
    ) or 0

    reviewed = db.scalar(
        select(func.count()).select_from(Image)
        .where(Image.project_id == project_id, Image.reviewed_at.is_not(None))
    ) or 0

    rows = db.execute(
        select(Label.name, func.count(Annotation.id))
        .outerjoin(Annotation, Annotation.label_id == Label.id)
        .where(Label.project_id == project_id)
        .group_by(Label.id)
    ).all()

    return StatsOut(
        total=total,
        pending=count("pending"),
        annotated=count("annotated"),
        skipped=count("skipped"),
        reviewed=reviewed,
        boxes_per_label={name: n for name, n in rows},
    )
