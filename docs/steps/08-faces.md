# Step 08 — Faces

**Goal:** detect faces, and optionally answer "who is this" against a gallery you build.

**Depends on:** step 06. Independent of step 09 — do them in either order.

**Read ARCHITECTURE §5.4 first.** Face recognition is biometric processing and is
regulated differently from object detection in several jurisdictions, and the common
pretrained weights are non-commercial-use only. That is your call to make on your own
photos; I want it stated before you build it, not after.

## 1. Detection vs recognition — genuinely different problems

| | Face detection | Face recognition |
|---|---|---|
| Question | "Is there a face, where?" | "Whose face is it?" |
| Method | Same detector as any object | Embedding + nearest neighbour |
| New subject | Retrain the model | Add ~5 photos to a gallery, no training |
| Fits step 01–07? | **Yes, unchanged** | No — separate pipeline |

If you only need "a person's face is here", add a `face` label to a normal project and
you are done at step 07. The rest of this document is about identity.

The crucial property of recognition: **adding a person requires no training.** You embed
a few reference photos, store the vectors, and match by cosine similarity. That is why it
is a separate pipeline rather than more classes on the detector — with a classifier, every
new person means a full retrain.

## 2. Pipeline

```
frame → [detect faces] → [align to 112×112 via 5 landmarks] → [ArcFace embed → 512-d]
                                                                        │
                                                     cosine vs gallery ─┤
                                                                        ▼
                                              best match ≥ threshold ? name : "unknown"
```

### Detector
Two options, both ONNX, both consistent with the rest of the app:

- **SCRFD** (from InsightFace) — purpose-built, returns the 5 landmarks alignment needs.
  Recommended.
- **Your own YOLO face model** — one more class in a normal project. No landmarks, so
  alignment falls back to a crude eye-line estimate and accuracy drops noticeably.

Prefer SCRFD. The landmarks are not optional decoration; alignment is worth several points
of accuracy.

### Alignment
Similarity transform mapping the detected 5 points onto ArcFace's canonical template:

```python
ARCFACE_5PT = np.array([[38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366],
                        [41.5493, 92.3655], [70.7299, 92.2041]], dtype=np.float32)

def align(bgr, landmarks5):
    M, _ = cv2.estimateAffinePartial2D(landmarks5, ARCFACE_5PT, method=cv2.LMEDS)
    return cv2.warpAffine(bgr, M, (112, 112), borderValue=0.0)
```

Skipping this and feeding a raw crop is the most common reason a face pipeline "sort of
works" — recognition accuracy falls off a cliff with pose variation.

### Embedding
ArcFace `w600k_r50.onnx` (InsightFace `buffalo_l` pack) → 512-d vector. **L2-normalise it**,
which makes cosine similarity a plain dot product:

```python
emb = session.run(None, {inp: aligned_nchw})[0][0]
emb = emb / np.linalg.norm(emb)
```

## 3. Gallery

```python
FaceIdentity(id, name, notes, created_at)
FaceEmbedding(id, identity_id, vector_blob, source_image, created_at)
```

Store vectors as `np.float32` bytes in a BLOB. At personal scale, load them all into one
matrix at startup and match with a single matmul (ARCHITECTURE §2):

```python
class Gallery:
    def load(self, db):
        rows = db.scalars(select(FaceEmbedding)).all()
        self.M = np.stack([np.frombuffer(r.vector_blob, np.float32) for r in rows])  # (N,512)
        self.ids = [r.identity_id for r in rows]

    def match(self, emb, threshold=0.35):
        sims = self.M @ emb                       # both L2-normalised → cosine
        i = int(sims.argmax())
        return (self.ids[i], float(sims[i])) if sims[i] >= threshold else (None, float(sims[i]))
```

5–10 reference photos per person, varied in pose, lighting and expression. More matters
less than varied — ten near-identical selfies are worth about as much as two.

## 4. Thresholds

Cosine similarity on ArcFace, rough guidance:

| Threshold | Behaviour |
|---|---|
| 0.28 | Loose — few misses, some wrong names |
| **0.35** | Balanced starting point |
| 0.45 | Strict — mostly "unknown", high precision when it does name someone |

Make it adjustable exactly like the detection threshold in step 06, and **always show the
similarity score next to the name.** "Alice 0.36" and "Alice 0.71" are very different
claims, and hiding the number invites false confidence.

Calibrate on your own data rather than trusting the table: embed a held-out photo of each
known person, record the correct-match and best-wrong-match scores, and set the threshold
between the distributions.

## 5. Integration with the rest of the app

Reuse everything: same `OnnxDetector` shape, same WebSocket path from step 06, same
threshold-slider UI. The face module is `/faces` with:

- **Identities** — CRUD, upload reference photos, per-face embed preview
- **Test** — image/video/webcam with names and scores overlaid
- **Export** — SCRFD + ArcFace ONNX plus `gallery.npz`, so other software can run the same
  pipeline

Note in the UI that the gallery is *your* data and lives in `data/app.db`, and that
deleting an identity deletes its embeddings. Deletability is a feature here, not an
afterthought.

## Acceptance criteria

- Add 3 identities with 5 photos each; embeddings stored.
- A held-out photo of each identifies correctly with similarity > 0.4.
- A stranger returns "unknown", not a wrong name.
- Aligned crops look centred and upright (dump a few to disk and actually look).
- Webcam mode names faces live at ≥5 FPS on CPU.
- Deleting an identity removes its embeddings and it stops matching.
