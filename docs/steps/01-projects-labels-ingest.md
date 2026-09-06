# Step 01 — Projects, labels, and folder ingest

**Goal:** create a project, define the labels you'll annotate with, point the app at a
folder of images, and see those images registered in the database.

**Depends on:** step 00.

## 1. Tables

`app/models.py` — the full schema from ARCHITECTURE §3. Add all tables now even though
later steps use them; `create_all` is cheap and it saves churn.

```python
class Project(Base):
    __tablename__ = "projects"
    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(unique=True)
    slug: Mapped[str] = mapped_column(unique=True)
    task: Mapped[str] = mapped_column(default="detect")
    created_at: Mapped[datetime] = mapped_column(default=datetime.utcnow)
    labels: Mapped[list["Label"]] = relationship(back_populates="project",
                                                 cascade="all, delete-orphan")

class Label(Base):
    __tablename__ = "labels"
    __table_args__ = (UniqueConstraint("project_id", "class_index"),
                      UniqueConstraint("project_id", "name"))
    id: Mapped[int] = mapped_column(primary_key=True)
    project_id: Mapped[int] = mapped_column(ForeignKey("projects.id", ondelete="CASCADE"))
    name: Mapped[str]
    color: Mapped[str] = mapped_column(default="#ff3b30")
    class_index: Mapped[int]

class Image(Base):
    __tablename__ = "images"
    __table_args__ = (UniqueConstraint("project_id", "sha256"),)
    id: Mapped[int] = mapped_column(primary_key=True)
    project_id: Mapped[int] = mapped_column(ForeignKey("projects.id", ondelete="CASCADE"))
    rel_path: Mapped[str]           # relative to settings.images_dir
    sha256: Mapped[str] = mapped_column(index=True)
    width: Mapped[int]
    height: Mapped[int]
    status: Mapped[str] = mapped_column(default="pending", index=True)
    ingested_at: Mapped[datetime] = mapped_column(default=datetime.utcnow)

class Annotation(Base):
    __tablename__ = "annotations"
    id: Mapped[int] = mapped_column(primary_key=True)
    image_id: Mapped[int] = mapped_column(ForeignKey("images.id", ondelete="CASCADE"),
                                          index=True)
    label_id: Mapped[int] = mapped_column(ForeignKey("labels.id", ondelete="CASCADE"))
    x1: Mapped[float]; y1: Mapped[float]; x2: Mapped[float]; y2: Mapped[float]  # 0-1
    source: Mapped[str] = mapped_column(default="human")
    created_at: Mapped[datetime] = mapped_column(default=datetime.utcnow)
```

Plus `Job`, `Model`, `FaceIdentity`, `FaceEmbedding` as sketched in ARCHITECTURE §3.

## 2. Class index allocation

```python
def next_class_index(db, project_id: int) -> int:
    """Monotonic. Retired indices are never reused — a model's classes.json must keep
    meaning what it meant when it was trained."""
    hi = db.scalar(select(func.max(Label.class_index)).where(Label.project_id == project_id))
    return 0 if hi is None else hi + 1
```

This means class indices can have gaps after deletions. That is correct and intentional.
Handle it at export time by remapping to a contiguous range *and* writing the remap into
`classes.json`, so the exported model is self-describing.

## 3. Ingest

`app/services/ingest.py`:

```python
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}

def ingest_folder(db, project: Project, src: Path, *, copy: bool = True) -> IngestReport:
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
    return IngestReport(added=added, skipped=skipped, failed=failed)
```

Design notes:

- **Hash-based filenames.** Two source folders with `IMG_0001.jpg` in each no longer
  collide, and re-ingesting is naturally idempotent.
- **`copy=True` by default.** Copying costs disk but means your dataset survives someone
  reorganising the source folder. Offer `copy=False` (symlink) for large archives, and
  warn in the UI that symlinks break the Docker path.
- **`im.verify()`** catches truncated JPEGs at ingest instead of at epoch 40 of training.

## 4. Endpoints

```
POST   /api/projects                {name}                       → Project
GET    /api/projects
POST   /api/projects/{id}/labels    {name, color}                → Label (index auto)
DELETE /api/labels/{id}                                          # index retired
POST   /api/projects/{id}/ingest    {source_dir, copy}           → IngestReport
GET    /api/projects/{id}/images?status=pending&limit=&offset=
GET    /api/images/{id}/file                                     → the bytes
GET    /api/projects/{id}/stats     → {total, pending, annotated, skipped, boxes_per_label}
```

`GET /api/images/{id}/file` must resolve through the DB row and confirm the final path is
inside `settings.images_dir` before opening it — never take a client-supplied path.

```python
resolved = (settings.images_dir / img.rel_path).resolve()
if not resolved.is_relative_to(settings.images_dir.resolve()):
    raise HTTPException(400, "path escape")
```

## 5. Minimal UI

One page, `templates/index.html`: project dropdown + create, label list with colour
swatches + add/delete, a text field for the source folder + Ingest button, and the stats
block. Plain forms are fine. This page is scaffolding for the annotator, not a product.

## 6. EXIF orientation — normalize on ingest, not on read

Phone and camera JPEGs frequently store image data unrotated and record the intended
rotation in an EXIF `Orientation` tag. Pillow's `.size` and OpenCV's `cv2.imread` both
ignore that tag, so a photo that looks upright everywhere else silently loads sideways —
and every box you draw on it will be sideways too, in a way that's invisible until you
compare against the source photo. Every commercial tool (CVAT, Label Studio, Roboflow)
normalizes this on ingest; do the same, once, so nothing downstream has to think about it:

```python
def write_normalized(src: Path, dest: Path, *, copy: bool) -> tuple[int, int]:
    """Bake EXIF rotation into pixels and write the hashed copy. Returns (w, h) post-rotation.
    This is exactly why ingest copies instead of symlinking — a symlink can't hold pixels
    that differ from the source file."""
    with PILImage.open(src) as im:
        im = ImageOps.exif_transpose(im)          # no-op if there's no orientation tag
        w, h = im.size
        im.save(dest, quality=95) if dest.suffix.lower() in {".jpg", ".jpeg"} else im.save(dest)
    return w, h
```

From this point on, every consumer — annotator, exporter, trainer, verify — can assume
pixels are already upright and never touch EXIF again. `copy=False` (symlink) skips this
normalization entirely; the ingest report should say so when that flag is used.

## 7. Optional: video as a source

CVAT and Roboflow both accept video directly and extract frames for you. It's a thin
wrapper around the same ingest path, worth adding once the rest of step 01 works:

```python
def ingest_video(db, project: Project, video_path: Path, *, fps: float = 1.0) -> IngestReport:
    """Extract frames at `fps` and hand each one through the normal image ingest path —
    same hashing, same dedupe. cv2 frames are already upright, so no EXIF step needed."""
    cap = cv2.VideoCapture(str(video_path))
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30
    stride = max(1, round(src_fps / fps))
    ...  # write every `stride`-th frame to a temp dir, then ingest_folder() it
```

Default `fps=1.0` — one frame per second of source video is plenty of visual variety
without drowning the annotation queue in near-duplicate frames from a 30fps camera.

## Acceptance criteria

- Create project "warehouse", add labels `forklift`, `pallet`, `person`.
- Ingest a folder of ~50 mixed images → report shows `added=50`.
- Run the same ingest again → `added=0, skipped=50`.
- Drop a corrupt file in the folder, re-ingest → `failed=1`, app still up.
- `GET /api/images/{id}/file` renders in the browser.
- Delete label `pallet`, add label `crate` → `crate.class_index == 3`, not 1.
- Ingest a phone photo shot in portrait with EXIF orientation 6 → stored image and its
  `width`/`height` are both upright, not rotated.
