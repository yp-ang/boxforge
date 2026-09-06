"""Step 07: draft boxes on un-annotated images from a trained model, so annotation is
correction rather than from-scratch drawing (docs/steps/07-prelabel-loop.md).

Two invariants the rest of the app depends on:
  - every drafted box gets source="model" (dashed in the annotator, bulk-deletable)
  - status stays "pending" until a human looks at it — model output is never training
    data on its own say-so
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models import Annotation, Image, Label, Project

# Doc §3: lower than the ~0.5+ used at deployment. A wrong box is one keystroke to
# delete; a missing one requires actually noticing, so pre-labelling biases to recall.
DEFAULT_PRELABEL_CONF = 0.4


@dataclass
class DetLike:
    cls: int
    score: float
    xyxy: list[float]


class Detector(Protocol):
    classes: dict[int, str]

    def predict(self, image, conf: float) -> list[DetLike]: ...


def uncertainty(dets: list) -> float:
    """Doc §4: low mean confidence -> uncertain. Zero detections is treated as the most
    suspicious case of all, ranking above anything a real detection could produce."""
    if not dets:
        return 0.9
    scores = [d.score for d in dets]
    return 1.0 - sum(scores) / len(scores)


def eligible_images(db: Session, project_id: int, limit: int) -> list[Image]:
    """Pending images with no annotations at all yet — never touches an image a human
    (or an earlier prelabel batch) has already put boxes on."""
    annotated_image_ids = select(Annotation.image_id).distinct()
    stmt = (
        select(Image)
        .where(Image.project_id == project_id, Image.status == "pending",
               Image.id.not_in(annotated_image_ids))
        .order_by(Image.id)
        .limit(limit)
    )
    return list(db.scalars(stmt))


def count_eligible(db: Session, project_id: int, limit: int) -> int:
    return len(eligible_images(db, project_id, limit))


def _label_id_by_class_name(db: Session, project: Project) -> dict[str, int]:
    labels = db.scalars(select(Label).where(Label.project_id == project.id))
    return {l.name: l.id for l in labels}


def prelabel(
    db: Session, project: Project, detector: Detector, conf: float, limit: int,
    read_image,
) -> dict:
    """Draft boxes on up to `limit` pending, un-annotated images.

    `read_image(rel_path) -> np.ndarray` is injected so this stays testable without a
    real ONNX session or real files on disk. Returns a summary including every image id
    touched, which the caller stores in the job's result_json — that list is what makes
    "undo this batch" possible (doc §5).
    """
    name_to_label_id = _label_id_by_class_name(db, project)
    unmapped: set[str] = set()

    images = eligible_images(db, project.id, limit)
    touched_ids: list[int] = []
    n_boxes = 0

    for img in images:
        frame = read_image(img.rel_path)
        dets = detector.predict(frame, conf=conf)
        h, w = frame.shape[:2]

        rows = []
        for d in dets:
            name = detector.classes.get(d.cls, str(d.cls))
            label_id = name_to_label_id.get(name)
            if label_id is None:
                unmapped.add(name)
                continue
            x1, y1, x2, y2 = d.xyxy
            rows.append(Annotation(
                image_id=img.id, label_id=label_id,
                x1=max(0.0, min(1.0, x1 / w)), y1=max(0.0, min(1.0, y1 / h)),
                x2=max(0.0, min(1.0, x2 / w)), y2=max(0.0, min(1.0, y2 / h)),
                source="model",
            ))
        db.add_all(rows)
        img.uncertainty = uncertainty(dets)
        # Status stays "pending" — see module docstring. Only the human-facing PUT in
        # routers/annotations.py is allowed to flip it to "annotated".
        touched_ids.append(img.id)
        n_boxes += len(rows)

    db.commit()
    return {
        "image_ids": touched_ids,
        "n_images": len(touched_ids),
        "n_boxes": n_boxes,
        "unmapped_classes": sorted(unmapped),
    }


def undo_batch(db: Session, image_ids: list[int]) -> dict:
    """Delete exactly one batch's drafts (doc §2, §5). Only ever touches source='model'
    rows on images still 'pending' — an image a human has since annotated (status
    'annotated', boxes flipped to 'human' by the full-replace PUT) is left untouched, so
    this can never undo real work."""
    if not image_ids:
        return {"deleted_annotations": 0, "images_reset": 0}

    pending_ids = [
        img_id for img_id, in db.execute(
            select(Image.id).where(Image.id.in_(image_ids), Image.status == "pending")
        ).all()
    ]
    if not pending_ids:
        return {"deleted_annotations": 0, "images_reset": 0}

    rows = db.scalars(
        select(Annotation).where(Annotation.source == "model",
                                 Annotation.image_id.in_(pending_ids))
    ).all()
    n = len(rows)
    for row in rows:
        db.delete(row)
    for img in db.scalars(select(Image).where(Image.id.in_(pending_ids))):
        img.uncertainty = None
    db.commit()
    return {"deleted_annotations": n, "images_reset": len(pending_ids)}
