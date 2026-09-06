from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import FileResponse
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.config import settings
from app.db import get_db
from app.models import Image, Project
from app.schemas import ImageOut, IngestReport, IngestRequest
from app.services.ingest import ingest_folder

router = APIRouter(tags=["images"])


@router.post("/api/projects/{project_id}/ingest", response_model=IngestReport)
def ingest(project_id: int, payload: IngestRequest, db: Session = Depends(get_db)):
    project = db.get(Project, project_id)
    if not project:
        raise HTTPException(404, "project not found")
    src = Path(payload.source_dir).expanduser()
    if not src.is_dir():
        raise HTTPException(400, f"not a directory: {src}")
    return ingest_folder(db, project, src, copy=payload.copy)


@router.get("/api/projects/{project_id}/images", response_model=list[ImageOut])
def list_images(
    project_id: int,
    status: str | None = None,
    limit: int = 50,
    offset: int = 0,
    db: Session = Depends(get_db),
):
    stmt = select(Image).where(Image.project_id == project_id)
    if status:
        stmt = stmt.where(Image.status == status)
    stmt = stmt.order_by(Image.id).offset(offset).limit(limit)
    return db.scalars(stmt).all()


@router.get("/api/images/{image_id}/file")
def image_file(image_id: int, db: Session = Depends(get_db)):
    img = db.get(Image, image_id)
    if not img:
        raise HTTPException(404, "image not found")

    resolved = (settings.images_dir / img.rel_path).resolve()
    if not resolved.is_relative_to(settings.images_dir.resolve()):
        raise HTTPException(400, "path escape")
    if not resolved.is_file():
        raise HTTPException(404, "file missing on disk")

    return FileResponse(resolved)
