# Build steps

Each file is self-contained: context, deliverables, code sketches, acceptance criteria.
Hand one to Claude with "implement this step" and it has everything it needs.

| # | Step | Ends with |
|---|---|---|
| [00](00-scaffold.md) | Scaffold | App boots, SQLite created, `/health` 200 |
| [01](01-projects-labels-ingest.md) | Projects, labels, ingest | Folder of images registered in the DB |
| [02](02-annotation-ui.md) | Annotation UI | Keyboard-driven box drawing, autosaved |
| [03](03-dataset-export.md) | Dataset export | Valid YOLO dataset, deterministic splits |
| [04](04-training.md) | Training | Background run, live logs, model row |
| [05](05-export-onnx.md) | ONNX export | Portable model folder + parity check |
| [06](06-inference-playground.md) | Verify playground | Image/video/webcam, threshold slider |
| [07](07-prelabel-loop.md) | Pre-labelling loop | Model drafts boxes, you correct |
| [08](08-faces.md) | Faces | Detection + identity gallery |
| [09](09-docker.md) | Packaging | conda and docker compose, same code |
| [10](../integration.md) | Integration | Model running in other software |

**Spine:** 00 → 06. Do these in order; each depends on the last.
**Then:** 07 (highest leverage), then 08 and 09 in either order.

## Suggested pacing

| Sitting | Steps | Why stop here |
|---|---|---|
| 1 | 00, 01 | Data is in the system — the boring part is done |
| 2 | 02 | The part that decides whether you use this. Get it right. |
| 3 | 03, 04 | First trained model, however bad |
| 4 | 05, 06 | First model you can *see* working, and hand to something else |
| 5 | 07 | Annotation gets 2–5× faster; everything after is cheaper |
| 6 | 09 | Reproducible |
| 7 | 08 | Faces, if you still want them |

Expect the first model at sitting 3 to be poor. 200 images and 100 epochs produces
something that mostly finds your object in conditions like the ones you photographed. That
is the expected result, not a failure — steps 06 and 07 exist to turn it into a real one.
