# Step 03 — Dataset export

**Goal:** turn the database into a YOLO-format dataset directory that a trainer can
consume, with deterministic splits and a pre-flight report that catches bad data before
you spend an hour training on it.

**Depends on:** step 02.

## 1. Output layout

```
data/datasets/<project_slug>/<timestamp>/
├── data.yaml
├── manifest.json
├── images/train/<sha16>.jpg      labels/train/<sha16>.txt
└── images/val/<sha16>.jpg        labels/val/<sha16>.txt
```

`data.yaml`:

```yaml
path: /abs/path/to/data/datasets/warehouse/20260906-141230
train: images/train
val: images/val
names:
  0: forklift
  1: person
  2: crate
```

Label files are YOLO text — one box per line, `class cx cy w h`, all normalised:

```
0 0.512340 0.447120 0.183000 0.264000
```

Images with zero annotations get an **empty `.txt` file**, not a missing one. Ultralytics
treats those as valid background images, which meaningfully reduces false positives.
A missing file is treated as an error instead.

## 2. Class remapping

`Label.class_index` has gaps after deletions (step 01). The exported dataset must be
contiguous from 0.

```python
labels = db.scalars(select(Label).where(Label.project_id == p.id)
                    .order_by(Label.class_index)).all()
remap = {lbl.id: i for i, lbl in enumerate(labels)}       # db id → contiguous index
names = {i: lbl.name for i, lbl in enumerate(labels)}
```

Write both `names` and the reverse mapping into `manifest.json`, so a model folder can
always be traced back to the exact label set that produced it.

## 3. Deterministic split

```python
def split_for(sha256_hex: str, val_pct: int = 20) -> str:
    return "val" if int(sha256_hex[:8], 16) % 100 < val_pct else "train"
```

Stable across re-exports: an image never changes side when you add new data. This is what
makes mAP comparable between run 1 and run 12. See ARCHITECTURE §4.

Guard the pathological case:

```python
if not val_images:
    raise ExportError("Validation split is empty — annotate more images "
                      f"(have {len(train_images)} train, 0 val).")
```

## 4. Coordinate conversion — the one place it happens

```python
def xyxy_norm_to_yolo(x1, y1, x2, y2) -> tuple[float, float, float, float]:
    """Normalised corners → YOLO centre/size. The ONLY conversion site in the codebase."""
    return ((x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1)
```

Unit-test it with a round trip. It is four lines and it protects the entire pipeline.

## 5. Pre-flight report

Run before writing anything and surface it in the UI. These checks are what turn a
mysteriously bad model into an obvious data problem:

| Check | Why it matters |
|---|---|
| Images per class | <50 instances for a class → that class will not learn. Warn loudly. |
| Class balance ratio | >20:1 between most and least common → expect the rare class to be ignored |
| Annotated image count | <100 total → expect a toy model; say so rather than let them wonder |
| Boxes < 8px on the long side after resize to `imgsz` | Too small for the model's stride; flag them |
| Boxes covering >95% of the frame | Usually a mis-drag, occasionally intentional |
| Duplicate boxes (IoU > 0.9, same class) | Double-click artefacts; offer to dedupe |
| Val split empty or single-class | Metrics will be meaningless |
| Unreadable image files | Fail now, not at epoch 40 |

```python
@dataclass
class PreflightReport:
    errors: list[str]      # block the export
    warnings: list[str]    # show, allow override
    per_class: dict[str, int]
    n_train: int
    n_val: int
```

Block on `errors`, require an explicit "export anyway" for `warnings`.

## 6. Symlink vs copy

Copy the image files into the dataset directory rather than symlinking. Symlinks break
when the directory is mounted into a container at a different path, and a dataset you
cannot reproduce is worse than a few duplicated gigabytes. If disk becomes real, add
hardlinks as an option — same filesystem only, but free.

## 7. Endpoint

```
POST /api/projects/{id}/datasets   {val_pct: 20, force: false}  → {dataset_id, path, report}
GET  /api/projects/{id}/datasets                                → previous exports
```

Keep every export. They are the provenance record for every model you train — being able
to answer "what data made this model" six months later is worth the disk.

## 8. COCO JSON export — optional, for compatibility

YOLO's `.txt` layout is what *this app's* trainer consumes, but it's not universal — COCO
JSON is the format most other tools (CVAT, Labelbox, torchvision, most published eval
scripts) actually expect when you want to move annotations somewhere else, or benchmark
against a script that assumes COCO. Since you already have every box in the DB, this is a
second writer over the same query, not a second pipeline:

```python
def export_coco(db, project: Project, images: list[Image]) -> dict:
    return {
        "images": [{"id": img.id, "file_name": img.rel_path,
                    "width": img.width, "height": img.height} for img in images],
        "annotations": [
            {"id": a.id, "image_id": a.image_id, "category_id": remap[a.label_id],
             "bbox": [a.x1 * img.width, a.y1 * img.height,          # COCO wants
                      (a.x2 - a.x1) * img.width, (a.y2 - a.y1) * img.height],  # pixel x,y,w,h
             "area": (a.x2-a.x1)*img.width * (a.y2-a.y1)*img.height, "iscrowd": 0}
            for a in annotations
        ],
        "categories": [{"id": i, "name": lbl.name} for i, lbl in enumerate(labels)],
    }
```

Note the coordinate convention difference: COCO `bbox` is `[x, y, w, h]` in **absolute
pixels** with top-left origin, not normalised — a second, deliberate exception to the "one
conversion function" rule in §4, because this is a genuinely different target format, not
an accidental second YOLO writer. Keep it in its own function (`export_coco`) so the YOLO
path in §1–§4 stays the single source of truth for training.

```
GET /api/projects/{id}/datasets/{dataset_id}/coco.json
```

Generate it on demand from the same dataset snapshot rather than storing it — one extra
query, and it can never drift out of sync with the YOLO files it's describing.

## Acceptance criteria

- Export a project with 3 classes → directory matches the layout above.
- `python -c "import yaml,sys; print(yaml.safe_load(open('data.yaml')))"` looks right.
- Every `images/**` file has a matching `labels/**` `.txt` (including empty ones).
- Re-export after adding 10 images → the original images are all in their previous split.
- A project with 5 annotated images → export blocked with a clear message.
- Round-trip test: `xyxy → yolo → xyxy` recovers the original within 1e-9.
- (If §8 built) `coco.json` validates against the COCO schema and a box's pixel `bbox`
  matches the corresponding YOLO line once converted by hand.
