# Step 07 — Pre-labelling loop

**Goal:** use the model you just trained to draft boxes on un-annotated images, so you
correct instead of drawing from scratch.

**Depends on:** steps 05 and 06.

This is small — maybe 150 lines — and it is the highest-leverage thing in the project.
Hand-drawing runs 200–600 boxes/hour. Correcting decent pre-labels runs 3–5× that. Since
annotation time is the real bound on a personal ML project (ARCHITECTURE §5.5), this step
is worth more than any modelling improvement you could make instead.

## 1. The loop

```
annotate 200 by hand  →  train v1  →  pre-label the next 500
       ↑                                      │
       └──── correct (fast) ── train v2 ──────┘
```

Each cycle gets more accurate, so each cycle is faster than the last.

## 2. Pre-label job

Reuse step 04's runner with `type="prelabel"`:

```python
def prelabel(db, project_id: int, model_id: int, conf: float = 0.4, limit: int = 500):
    det = OnnxDetector(model_dir(model_id))
    label_by_index = {i: db_label_id for ...}    # model class index → Label.id

    images = db.scalars(
        select(Image).where(Image.project_id == project_id,
                            Image.status == "pending").limit(limit)).all()

    for img in images:
        frame = cv2.imread(str(settings.images_dir / img.rel_path))
        dets = det.predict(frame, conf=conf)
        h, w = frame.shape[:2]
        db.add_all([
            Annotation(image_id=img.id, label_id=label_by_index[d.cls],
                       x1=d.xyxy[0]/w, y1=d.xyxy[1]/h,
                       x2=d.xyxy[2]/w, y2=d.xyxy[3]/h,
                       source="model")               # ← the important field
            for d in dets
        ])
        img.status = "pending"      # STILL pending — a draft is not a label
    db.commit()
```

Two invariants:

**`source="model"` on every drafted box.** Draw model boxes dashed and human boxes solid,
so at a glance you know what you have reviewed. It also lets you delete every draft in one
statement if a bad model pollutes the set:

```sql
DELETE FROM annotations WHERE source='model' AND image_id IN (...);
```

**Status stays `pending` until a human saves the image.** Model output is never training
data until you have looked at it. The moment you break this rule, the model starts training
on its own mistakes and confidently reinforces them — the failure is quiet and it compounds.

The existing full-replace PUT from step 02 handles conversion naturally: when the annotator
saves an image, all boxes come back as `source="human"` and status becomes `annotated`.

## 3. Threshold choice

Use a *lower* confidence for pre-labelling (~0.3–0.4) than for deployment (~0.5+). Deleting
a wrong box is one keystroke; noticing a missing box requires you to actually look. Bias
toward recall — over-labelling is cheaper to fix than under-labelling.

## 4. Review ordering — where to spend attention

Do not review in ingest order. Sort by where the model is least certain, so your time goes
to the images that will actually teach it something:

```python
def uncertainty(dets) -> float:
    if not dets:
        return 0.9                                   # found nothing: suspicious
    scores = [d.score for d in dets]
    return 1.0 - sum(scores) / len(scores)           # low mean confidence → uncertain
```

Offer three orderings in the UI:

| Order | Use when |
|---|---|
| Most uncertain first | Default. Fastest improvement per image reviewed. |
| Zero detections first | Hunting for missed objects / new conditions |
| Most detections first | Hunting for false positives and duplicate boxes |

This is classical active learning, in nine lines. A margin-based score (top class minus
second class) is marginally better if you want to refine it later, but mean-confidence
captures most of the value.

## 5. Guardrails

- Refuse to pre-label images that already have `source="human"` annotations. Never
  overwrite real work.
- Show a count before running: "will draft boxes on 412 pending images with model
  warehouse-v3 @ conf 0.4" — plus a one-click undo that deletes that batch's drafts.
- Record the model id in a batch tag so "undo the drafts from v2" is possible after you
  train v3.

## 6. Bulk accept/reject — for the images the model already nailed

CVAT's and Roboflow's model-assisted review both let you sweep through a batch and accept
or reject whole images at a glance, reserving per-box editing for the ones that actually
need it. Once pre-labelling is decent (after a cycle or two), a meaningful fraction of
drafted images need zero corrections — reopening the full annotator for each is wasted
motion.

Add a lightweight grid review, separate from the box-by-box annotator:

```
GET  /api/projects/{id}/prelabel-batch/{batch_id}   → thumbnails with drafted boxes burned in
POST /api/images/{id}/accept                        → source="model" boxes become "human",
                                                       status → "annotated" (no annotator visit)
POST /api/images/{id}/reject                        → delete this image's drafts, status
                                                       stays "pending" for a from-scratch pass
```

```
┌───────┐ ┌───────┐ ┌───────┐ ┌───────┐
│ [img] │ │ [img] │ │ [img] │ │ [img] │   space = accept focused, X = reject,
│ ✓ ✕   │ │ ✓ ✕   │ │ ✓ ✕   │ │ ✓ ✕   │   arrow keys to move focus, Enter to
└───────┘ └───────┘ └───────┘ └───────┘   open the full annotator for that one
```

Route to the full per-box annotator (step 02) only on reject, or on a deliberate "edit"
action — most of a good batch should be a few seconds of skimming and space-bar, not a
re-annotation. This is the same "review, don't redraw" principle that makes pre-labelling
fast in the first place, applied one level up.

## Acceptance criteria

- Train on 200 images, pre-label 300 → drafts appear, all `source="model"`.
- Annotator shows drafts dashed; saving converts them to solid `human` and marks annotated.
- Pre-labelled-but-unreviewed images are excluded from the next dataset export.
- Undo removes exactly that batch's drafts and nothing else.
- Uncertainty ordering puts empty-detection images near the top.
- Measure it: time 20 hand-drawn images vs 20 corrected. Expect ≥2×.
- (If §6 built) Accepting in the grid view converts drafts to human-sourced and marks
  annotated without opening the per-box annotator; rejecting clears drafts and leaves the
  image pending.
