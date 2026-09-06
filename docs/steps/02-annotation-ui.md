# Step 02 — Annotation UI

**Goal:** open each image in turn, draw bounding boxes, assign a preset label, move on.
Fast enough that annotating 500 images is an afternoon rather than a weekend.

**Depends on:** step 01.

This is the step where a personal project either becomes usable or quietly dies. The
difference is entirely keyboard ergonomics — every mouse trip to a toolbar is a tax you
pay hundreds of times.

## 1. Layout

```
┌────────────────────────────────────────────────────────┬─────────────┐
│                                                        │ LABELS      │
│                                                        │ 1 forklift  │
│           <canvas> — image + boxes                     │ 2 pallet ●  │
│           drag to draw, click to select                │ 3 person    │
│                                                        │             │
│                                                        │ BOXES (3)   │
│                                                        │ pallet …  ✕ │
│                                                        │ person …  ✕ │
├────────────────────────────────────────────────────────┤             │
│  ◀ prev   [ 128 / 540 ]   next ▶     skip     ⌫ delete │ conf: n/a   │
└────────────────────────────────────────────────────────┴─────────────┘
```

## 2. Keybindings — the actual feature

| Key | Action |
|---|---|
| `1`–`9` | Select active label (also retags the currently selected box) |
| drag on canvas | Draw a box with the active label |
| click a box | Select it |
| `Backspace` / `Delete` | Delete selected box |
| `D` / `→` | Save + next image |
| `A` / `←` | Save + previous image |
| `S` | Mark skipped + next |
| `Ctrl/Cmd+Z` | Undo last box operation (in-memory stack, per image) |
| `Esc` | Deselect |
| `+` / `-` | Zoom; `Space`+drag to pan |

Two rules that matter more than they sound:

1. **Autosave on navigate**, not on a Save button. The user should never lose work and
   should never think about saving.
2. **Preserve the active label across images.** You annotate in runs of one class. Resetting
   the selection every image doubles the keystrokes.

## 3. Canvas mechanics

Keep two coordinate spaces and one conversion pair. Everything in the model layer is
normalised 0–1; only the render layer knows about pixels.

```js
// view state
let scale = 1, offsetX = 0, offsetY = 0;   // fit-to-container, then user zoom/pan

const toCanvas = (nx, ny) => [nx * imgW * scale + offsetX,
                              ny * imgH * scale + offsetY];
const toNorm   = (cx, cy) => [clamp01((cx - offsetX) / (imgW * scale)),
                              clamp01((cy - offsetY) / (imgH * scale))];
```

Normalise the box on commit so `x1<x2, y1<y2` regardless of drag direction, and drop
degenerate boxes:

```js
function commitBox(a, b, labelId) {
  const [x1, x2] = [Math.min(a.x, b.x), Math.max(a.x, b.x)];
  const [y1, y2] = [Math.min(a.y, b.y), Math.max(a.y, b.y)];
  if ((x2 - x1) * imgW < 4 || (y2 - y1) * imgH < 4) return;   // stray click, not a box
  boxes.push({ x1, y1, x2, y2, label_id: labelId, source: "human", dirty: true });
}
```

Use `devicePixelRatio` when sizing the canvas backing store or boxes will look soft on a
Retina display and you will misjudge tightness:

```js
const dpr = window.devicePixelRatio || 1;
canvas.width  = rect.width  * dpr;
canvas.height = rect.height * dpr;
ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
```

Draw order: image → boxes (2px stroke in the label colour, 12% alpha fill) → selected box
(dashed white overlay + 8px corner handles) → label text on a filled chip above the box,
flipped below when the box touches the top edge.

## 4. Resize handles

Once you have >50 images you will want to nudge boxes, not redraw them. Hit-test in
canvas pixels with a ~8px tolerance, corners before edges before body:

```js
function hitTest(px, py, box) {
  const [x1,y1] = toCanvas(box.x1, box.y1), [x2,y2] = toCanvas(box.x2, box.y2);
  const T = 8;
  for (const [name, hx, hy] of [["nw",x1,y1],["ne",x2,y1],["sw",x1,y2],["se",x2,y2]])
    if (Math.abs(px-hx) < T && Math.abs(py-hy) < T) return name;
  if (px > x1 && px < x2 && py > y1 && py < y2) return "move";
  return null;
}
```

## 5. Endpoints

```
GET  /api/projects/{id}/next?after={image_id}&status=pending   → next image + its boxes
PUT  /api/images/{id}/annotations   {boxes:[{label_id,x1,y1,x2,y2}]}   # full replace
POST /api/images/{id}/skip
GET  /annotate/{project_id}                                    → the page
```

**Full replace, not incremental patch.** The client holds the truth for one image at a
time; PUT the whole list. It is one transaction, it is idempotent, and it removes an
entire category of sync bug. The list is never more than a few dozen boxes.

Server side:

```python
@router.put("/api/images/{image_id}/annotations")
def put_annotations(image_id: int, payload: AnnotationsIn, db=Depends(get_db)):
    img = db.get(Image, image_id) or abort(404)
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
```

Validate geometry server-side even though the client already did. The client is not a
trusted source, and a `x2 < x1` box poisons a training run in a way that is genuinely
hard to trace back.

## 6. Prefetch

Preload the next image while the user annotates the current one. One line, removes the
entire perceived latency of the loop:

```js
const pre = new Image(); pre.src = `/api/images/${nextId}/file`;
```

## 7. SAM-assisted boxes — optional, worth adding after the basics work

CVAT and Label Studio both ship a Segment Anything (SAM) integration: click roughly on an
object, SAM returns a tight mask/box, you accept and move on. This is the single highest-
leverage add-on to the annotator, because drawing a *precise* box freehand is the slowest
part of manual annotation — clicking a point is not.

SAM 2/3 weights are Apache-2.0 (Meta), so this is free to use, including for anything you
later train on the results. Use **MobileSAM** or **EfficientViT-SAM**, not the full ViT-H
SAM — the full encoder is ~2.4GB and multiple seconds per image on CPU; the mobile variants
run in a few hundred ms and are the difference between this feeling instant and feeling
like a chore.

```
POST /api/images/{id}/sam-box   {x, y}            # a single click, normalised 0-1
  → runs the SAM encoder (cached per image) + a point-prompt decode
  → returns the tightest axis-aligned box around the resulting mask
```

```python
# app/services/sam_assist.py — same OnnxDetector shape as everything else in step 06
class SamAssist:
    def __init__(self, encoder_path: Path, decoder_path: Path):
        self.enc = ort.InferenceSession(str(encoder_path), providers=providers())
        self.dec = ort.InferenceSession(str(decoder_path), providers=providers())
        self._embed_cache: dict[int, np.ndarray] = {}   # image_id → embedding

    def box_from_click(self, image_id: int, bgr: np.ndarray, x: float, y: float) -> BBox:
        embedding = self._embed_cache.get(image_id) or self._encode(bgr)
        self._embed_cache[image_id] = embedding
        mask = self._decode_point(embedding, x, y)
        return mask_to_xyxy(mask)         # tight bounding rect of the largest contour
```

Cache the image embedding per `image_id` — the encoder is the expensive part, and within
one image you may click several objects. Evict the cache on navigate.

UI: hold `Alt` and click instead of dragging → SAM box appears selected and editable like
any other box (same resize handles from §4), just pre-filled instead of hand-drawn. It's an
accelerator for the existing box model, not a new annotation type — no schema change.

Treat this as optional relative to the MVP (step 06). Ship manual dragging first; add SAM
once you're annotating for real and feeling the cost of freehand precision.

## 8. Reviewed status — a deliberate second pass

CVAT distinguishes "labeled" from "reviewed" as separate pipeline stages. Borrow that
distinction cheaply: add a `reviewed_at` timestamp to `Image`, nullable, set by a single
keybinding (`R`) that means "I looked at this again and it's correct" without changing any
boxes. This matters for two situations the current two-state model (`pending`/`annotated`)
doesn't capture:

- **Catching your own mistakes.** A first pass through 300 images is fast and error-prone;
  a second pass, filtered to `status=annotated AND reviewed_at IS NULL`, is where you
  actually catch wrong labels and sloppy boxes — the same value CVAT's review stage
  provides for a team, applied to your own two passes instead of two people.
- **Trusting pre-labelled data** (step 07). A batch of accepted model drafts is
  `status=annotated` but arguably deserves a lower trust level than something you drew
  and reviewed twice. Filter `/api/projects/{id}/stats` by both fields so you can see, at a
  glance, how much of your "annotated" count is actually reviewed.

This is one nullable column and one keybinding — cheap enough to add in step 02 directly
rather than deferring it.

## Acceptance criteria

- Annotate 20 images end to end without touching the mouse except to draw.
- Reload the page mid-run → boxes come back exactly as drawn.
- Draw a box by dragging bottom-right → top-left → stored correctly.
- `POST` a box with `x2 < x1` via curl → 400, nothing written.
- Zoom to 400%, draw a tight box, zoom out → box still aligns with the object.
- `/api/projects/{id}/stats` shows `annotated` climbing.
- (If §7 built) `Alt`+click on an object → box appears within ~1s on CPU, editable like any
  hand-drawn box.
- (If §8 built) Pressing `R` sets `reviewed_at`; filtering by `reviewed_at IS NULL` surfaces
  only unreviewed annotated images.
