from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import delete, func, select
from sqlalchemy.orm import Session

from app.db import get_db
from app.models import Annotation, Image, Label, Project
from app.schemas import AnnotationsIn, ImageDetailOut, ImageOut

router = APIRouter(tags=["annotations"])

ORDERINGS = ("id", "uncertain", "empty", "dense")


def _ordered_ids(db: Session, project_id: int, status: str | None, unreviewed: bool,
                 order: str) -> list[int]:
    """The full id list for this filter, in review order (doc §4). Re-sorted on every
    call rather than kept as a cursor — simple, and cheap at the personal-project scale
    ARCHITECTURE §5.5 targets (tens of thousands of images, not millions)."""
    if order == "dense":
        # Most detections first: order by annotation count, a join+group rather than a
        # stored column since it changes every time a box is added or removed.
        stmt = (
            select(Image.id, func.count(Annotation.id))
            .outerjoin(Annotation, Annotation.image_id == Image.id)
            .where(Image.project_id == project_id)
        )
        if status:
            stmt = stmt.where(Image.status == status)
        if unreviewed:
            stmt = stmt.where(Image.reviewed_at.is_(None))
        stmt = stmt.group_by(Image.id).order_by(func.count(Annotation.id).desc(), Image.id.asc())
        return [row[0] for row in db.execute(stmt).all()]

    stmt = select(Image.id, Image.uncertainty).where(Image.project_id == project_id)
    if status:
        stmt = stmt.where(Image.status == status)
    if unreviewed:
        stmt = stmt.where(Image.reviewed_at.is_(None))
    rows = db.execute(stmt).all()

    if order == "uncertain":
        # Least mean-confidence first; never-prelabelled images (uncertainty is None)
        # sort after everything the model has actually looked at.
        rows.sort(key=lambda r: (-(r[1] if r[1] is not None else -1.0), r[0]))
    elif order == "empty":
        # Zero-detection images (the 0.9 sentinel from services/prelabel.uncertainty)
        # first, in ingest order; everything else follows in ingest order too.
        rows.sort(key=lambda r: (0 if r[1] is not None and r[1] >= 0.9 else 1, r[0]))
    else:
        rows.sort(key=lambda r: r[0])
    return [r[0] for r in rows]


@router.get("/api/projects/{project_id}/next", response_model=ImageDetailOut)
def next_image(
    project_id: int,
    after: int | None = None,
    direction: str = "next",
    status: str | None = None,
    unreviewed: bool = False,
    order: str = "id",
    db: Session = Depends(get_db),
):
    if direction not in ("next", "prev"):
        raise HTTPException(400, "direction must be 'next' or 'prev'")
    if order not in ORDERINGS:
        raise HTTPException(400, f"order must be one of {ORDERINGS}")
    project = db.get(Project, project_id)
    if not project:
        raise HTTPException(404, "project not found")

    ids = _ordered_ids(db, project_id, status, unreviewed, order)
    if not ids:
        raise HTTPException(404, "no more images")

    if after is None:
        target_pos = 0 if direction == "next" else len(ids) - 1
    else:
        try:
            anchor_pos = ids.index(after)
        except ValueError:
            anchor_pos = -1 if direction == "next" else len(ids)
        target_pos = anchor_pos + 1 if direction == "next" else anchor_pos - 1

    if target_pos < 0 or target_pos >= len(ids):
        raise HTTPException(404, "no more images")

    img = db.get(Image, ids[target_pos])
    annotations = db.scalars(
        select(Annotation).where(Annotation.image_id == img.id)
    ).all()

    return ImageDetailOut(image=ImageOut.model_validate(img), annotations=annotations,
                           index=target_pos + 1, total=len(ids))


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
