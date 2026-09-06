"""Renders §7's before/after grid. Keeps the base64/PIL plumbing out of the router and
the DB out of trainer.py, which stays the single ultralytics-facing module."""
from __future__ import annotations

import base64
import io

import numpy as np
from PIL import Image as PILImage
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.config import settings
from app.models import Image, Project
from app.schemas import AugmentPreviewItem, AugmentPreviewOut

# Ultralytics transforms are BGR-in/BGR-out (they are written against cv2), so the
# channel flips below are load-bearing, not decoration.
PREVIEW_MAX_SIDE = 640


def _to_data_uri(bgr: np.ndarray) -> str:
    rgb = np.ascontiguousarray(bgr[:, :, ::-1])
    buf = io.BytesIO()
    pil = PILImage.fromarray(rgb)
    pil.thumbnail((PREVIEW_MAX_SIDE, PREVIEW_MAX_SIDE))
    pil.save(buf, format="JPEG", quality=80)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def _load_bgr(rel_path: str, imgsz: int) -> np.ndarray:
    with PILImage.open(settings.images_dir / rel_path) as pil:
        pil = pil.convert("RGB")
        pil.thumbnail((imgsz, imgsz))
        return np.asarray(pil)[:, :, ::-1].copy()


def build_augment_preview(
    db: Session, project: Project, n: int = 6, imgsz: int = 640, fliplr: float = 0.5
) -> AugmentPreviewOut:
    from app.services.trainer import TrainConfig, augment_preview

    images = db.scalars(
        select(Image)
        .where(Image.project_id == project.id, Image.status.in_(("annotated", "pending")))
        .order_by(Image.id)
        .limit(n)
    ).all()
    if not images:
        raise RuntimeError("no images ingested in this project yet")

    loaded: list[np.ndarray] = []
    kept: list[Image] = []
    for img in images:
        try:
            loaded.append(_load_bgr(img.rel_path, imgsz))
            kept.append(img)
        except (OSError, ValueError):
            continue
    if not loaded:
        raise RuntimeError("none of this project's images could be read from disk")

    cfg = TrainConfig(imgsz=imgsz, fliplr=fliplr)
    augmented = augment_preview(loaded, cfg)

    note = None
    if len(loaded) < 4:
        note = ("mosaic needs four tiles; with fewer images than that it reuses the ones "
                "you have, so the grid repeats")

    return AugmentPreviewOut(
        items=[
            AugmentPreviewItem(image_id=img.id, original=_to_data_uri(orig),
                               augmented=_to_data_uri(aug))
            for img, orig, aug in zip(kept, loaded, augmented)
        ],
        note=note,
    )
