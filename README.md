# Machine ID — annotate → train → export → verify

A single, self-hosted Python app for building object-detection models from your own
photos, and for putting the resulting model to work in other systems.

The loop it implements:

```
folder of images
      │
      ▼
[1] Ingest ──► [2] Annotate (bounding box + preset label, keyboard driven)
                          │
                          ▼
                  [3] Export dataset (YOLO layout, deterministic splits)
                          │
                          ▼
                  [4] Train (background job, live logs)
                          │
                          ▼
                  [5] Export ONNX (+ OpenVINO / CoreML)  ──► other software
                          │
                          ▼
                  [6] Verify (image / video / webcam, adjustable confidence)
                          │
                          └──► [7] Pre-label the next batch, correct, retrain
```

Two extra modules sit on the same spine:

- **[8] Faces** — face *detection* is just another class; face *recognition* (who is this)
  is a separate detect → align → embed → nearest-neighbour pipeline with its own gallery.
- **[9] Packaging** — one `conda activate` path and one `docker compose` path, same code.

## Where to start

| Doc | What it is |
|---|---|
| [ARCHITECTURE.md](ARCHITECTURE.md) | The stack, why each piece, the data model, the risks I want you to know about up front |
| [docs/steps/](docs/steps/) | 11 self-contained build steps. Each has context, deliverables, code sketches and acceptance criteria |
| [docs/integration.md](docs/integration.md) | How the exported model plugs into Frigate, DeepStream, OpenVINO, VMS platforms, plain Python |
| [docs/comparison.md](docs/comparison.md) | How this stacks up against CVAT, Label Studio, Roboflow, Labelbox and friends — and which of their ideas got folded into the steps |

Read `ARCHITECTURE.md` first — especially **Known constraints**, because two of them
(Ultralytics licensing, webcam-inside-Docker on macOS) will shape decisions you make
in week one.

## Ground rules for the build

1. **Ship step 00→06 before touching faces or Docker.** A working detector on your own
   images is the milestone that proves the whole design; everything else is additive.
2. **Every step ends runnable.** No step leaves the app broken.
3. **The verify module loads the exported ONNX, not the PyTorch checkpoint.** You test
   what you ship, not what you trained. This catches export bugs on day one instead of
   in the VMS three months later.
4. **Data lives under `data/`, code never writes outside it.** That single rule is what
   makes the Docker volume mount trivial.
