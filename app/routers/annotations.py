from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import delete, func, select
from sqlalchemy.orm import Session

from app.db import get_db
from app.models import Annotation, Image, Label, Project
from app.schemas import AnnotationsIn, ImageDetailOut, ImageOut

router = APIRouter(tags=["annotations"])


@router.get("/api/projects/{project_id}/next", response_model=ImageDetailOut)
def next_image(
    project_id: int,
    after: int | None = None,
    direction: str = "next",
    status: str | None = None,
    unreviewed: bool = False,
    db: Session = Depends(get_db),
):
    if direction not in ("next", "prev"):
        raise HTTPException(400, "direction must be 'next' or 'prev'")
    project = db.get(Project, project_id)
    if not project:
        raise HTTPException(404, "project not found")

    def filtered(stmt):
        stmt = stmt.where(Image.project_id == project_id)
        if status:
            stmt = stmt.where(Image.status == status)
        if unreviewed:
            stmt = stmt.where(Image.reviewed_at.is_(None))
        return stmt

    stmt = filtered(select(Image))
    if direction == "next":
        if after is not None:
            stmt = stmt.where(Image.id > after)
        stmt = stmt.order_by(Image.id.asc())
    else:
        if after is not None:
            stmt = stmt.where(Image.id < after)
        stmt = stmt.order_by(Image.id.desc())

    img = db.scalars(stmt.limit(1)).first()
    if not img:
        raise HTTPException(404, "no more images")

    total = db.scalar(filtered(select(func.count()).select_from(Image))) or 0
    index = db.scalar(
        filtered(select(func.count()).select_from(Image)).where(Image.id <= img.id)
    ) or 0

    annotations = db.scalars(
        select(Annotation).where(Annotation.image_id == img.id)
    ).all()

    return ImageDetailOut(image=ImageOut.model_validate(img), annotations=annotations,
                           index=index, total=total)


@router.put("/api/images/{image_id}/annotations")
def put_annotations(image_id: int, payload: AnnotationsIn, db: Session = Depends(get_db)):
    img = db.get(Image, image_id)
    if not img:
        raise HTTPException(404, "image not found")
    valid = {l.id for l in db.scalars(select(Label).where(Label.project_id == img.project_id))}
    for b in payload.boxes:
        if b.label_id not in valid:
            raise HTTPException(400, f"label {b.label_id} not in project")
        if not (0 <= b.x1 < b.x2 <= 1 and 0 <= b.y1 < b.y2 <= 1):
            raise HTTPException(400, "invalid box geometry")
    db.execute(delete(Annotation).where(Annotation.image_id == image_id))
    db.add_all([Annotation(image_id=image_id, **b.model_dump()) for b in payload.boxes])
    img.status = "annotated" if payload.boxes else "pending"
    db.commit()
    return {"ok": True}


@router.post("/api/images/{image_id}/skip", response_model=ImageOut)
def skip_image(image_id: int, db: Session = Depends(get_db)):
    img = db.get(Image, image_id)
    if not img:
        raise HTTPException(404, "image not found")
    img.status = "skipped"
    db.commit()
    db.refresh(img)
    return img


@router.post("/api/images/{image_id}/review", response_model=ImageOut)
def review_image(image_id: int, db: Session = Depends(get_db)):
    img = db.get(Image, image_id)
    if not img:
        raise HTTPException(404, "image not found")
    img.reviewed_at = datetime.utcnow()
    db.commit()
    db.refresh(img)
    return img
