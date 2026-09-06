import hashlib
import tempfile
from pathlib import Path

import cv2
from PIL import Image as PILImage
from PIL import ImageOps
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.config import settings
from app.models import Image, Project
from app.schemas import IngestReport

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def sha256_file(path: Path, chunk_size: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def write_normalized(src: Path, dest: Path, *, copy: bool) -> tuple[int, int]:
    """Bake EXIF rotation into pixels and write the hashed copy. Returns (w, h) post-rotation.
    This is exactly why ingest copies instead of symlinking — a symlink can't hold pixels
    that differ from the source file."""
    if not copy:
        dest.symlink_to(src.resolve())
        with PILImage.open(src) as im:
            im = ImageOps.exif_transpose(im)
            return im.size
    with PILImage.open(src) as im:
        im = ImageOps.exif_transpose(im)
        w, h = im.size
        im.save(dest, quality=95) if dest.suffix.lower() in {".jpg", ".jpeg"} else im.save(dest)
    return w, h


def ingest_folder(db: Session, project: Project, src: Path, *, copy: bool = True) -> IngestReport:
    """Walk src, hash each image, register new ones. Idempotent: re-running after adding
    files to the folder registers only the new ones."""
    added = skipped = failed = 0
    dest_root = settings.images_dir / project.slug
    dest_root.mkdir(parents=True, exist_ok=True)

    for p in sorted(src.rglob("*")):
        if p.suffix.lower() not in IMAGE_EXTS or not p.is_file():
            continue
        try:
            digest = sha256_file(p)
            if db.scalar(select(Image).where(Image.project_id == project.id,
                                             Image.sha256 == digest)):
                skipped += 1                      # exact duplicate content
                continue
            with PILImage.open(p) as im:
                im.verify()                       # cheap corruption check
            dest = dest_root / f"{digest[:16]}{p.suffix.lower()}"
            w, h = write_normalized(p, dest, copy=copy)   # see §6 — bakes EXIF rotation in
            db.add(Image(project_id=project.id, rel_path=str(dest.relative_to(settings.images_dir)),
                         sha256=digest, width=w, height=h))
            added += 1
        except Exception:
            failed += 1
    db.commit()
    note = "symlinked (no EXIF normalization applied)" if not copy else None
    return IngestReport(added=added, skipped=skipped, failed=failed, note=note)


def ingest_video(db: Session, project: Project, video_path: Path, *, fps: float = 1.0) -> IngestReport:
    """Extract frames at `fps` and hand each one through the normal image ingest path —
    same hashing, same dedupe. cv2 frames are already upright, so no EXIF step needed."""
    cap = cv2.VideoCapture(str(video_path))
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30
    stride = max(1, round(src_fps / fps))

    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        frame_idx = written = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if frame_idx % stride == 0:
                cv2.imwrite(str(tmp_dir / f"frame_{written:06d}.jpg"), frame)
                written += 1
            frame_idx += 1
        cap.release()
        return ingest_folder(db, project, tmp_dir, copy=True)
