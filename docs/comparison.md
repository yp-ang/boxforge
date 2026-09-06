# Commercial annotation software — and what's worth stealing from it

Short answer: yes, this category is well served commercially. Here's an honest look at
where those tools sit relative to this design, and which of their ideas are cheap enough
to fold into a personal build.

## The landscape

| Tool | Model | Self-host free? | Task range | Model-assisted labeling | Notes |
|---|---|---|---|---|---|
| **CVAT** | Open-source (CVAT.ai) | ✅ full self-host | Detection, segmentation, keypoints, 3D, video w/ tracking | ✅ SAM, trackers | Broadest open tool. Online tier free up to 10 tasks/500MB. |
| **Label Studio** | Open-source (HumanSignal) | ✅ full self-host | Detection, NLP, audio, multi-modal | ✅ SAM integration built in | Most flexible for mixed data types; enterprise tier adds RBAC/SSO. |
| **Roboflow** | Proprietary, hosted | ⚠️ free tier only, capped | Detection, segmentation, classification | ✅ + auto-augment, one-click train | Full labeling→training→deploy pipeline; you don't own the infra. |
| **Labelbox / V7 Darwin / Encord / SuperAnnotate** | Proprietary, hosted | ❌ paid, per-seat | Enterprise QA, consensus review, workflow routing | ✅ | Built for teams with a budget and multiple annotators, not solo use. |
| **LabelImg / makesense.ai** | Open-source, minimal | ✅ | Boxes only, no training | ❌ | Closest in *spirit* to a personal tool, but stops at annotation — no train/export/verify loop. |
| **This project** | Yours, self-hosted | ✅ free | Detection (+ face ID) | ✅ SAM-assisted (see below) | Narrower on purpose: one user, one machine, full control of the ONNX contract for VMS/NVR integration. |

## Where the commercial tools are genuinely ahead

Being fair about it, they beat a solo build on:

- **Team workflows** — task assignment, reviewer/annotator separation, consensus scoring
  across multiple people (inter-annotator agreement).
- **Scale** — cloud storage backends, distributed annotation across hundreds of thousands
  of images, org-level billing and access control.
- **Polish** — onboarding flows, hosted infra, support contracts.

None of that is worth building for a one-person project — it's pure overhead with no one
to divide it among. Explicitly **not building**: multi-tenant auth/RBAC, hosted SaaS
billing, consensus/QA workflows across annotators, generic NLP/audio annotation, cloud
storage connectors. If a future need for these appears, self-hosted CVAT or Label Studio
already solve them better than a bespoke rebuild would.

## Where they're not actually ahead of what you're building

- **Deployment contract.** Most of these tools optimize for staying inside their own
  ecosystem (Roboflow's hosted inference, Labelbox's model registry). Your ONNX-first
  export (ARCHITECTURE §2) is arguably *more* portable to Frigate/OpenVINO/DeepStream/VMS
  than what most of them ship, because it's the explicit design goal here rather than a
  side effect.
- **Ownership.** Your data and models never leave your machine. Several of the hosted
  options put training data through a third party by default.
- **Cost.** All free at your scale; CVAT/Label Studio self-hosted are the closest
  free peers, but you additionally get the trained-model and verify pipeline they don't
  bundle for free.

## What's worth stealing — added into the steps

These are the specific commercial-tool ideas that are cheap enough to be worth the extra
build time. Each is now folded into the relevant step doc rather than listed here only:

| Idea | Borrowed from | Where it landed |
|---|---|---|
| SAM-assisted box drawing (click → tight box) | CVAT, Label Studio SAM integration | [step 02 §7](steps/02-annotation-ui.md) |
| EXIF orientation normalization on ingest | Every serious tool does this silently | [step 01 §6](steps/01-projects-labels-ingest.md) |
| Second-pass "reviewed" status, distinct from "annotated" | CVAT's review stage | [step 02 §8](steps/02-annotation-ui.md) |
| COCO JSON export alongside YOLO | Universal in Roboflow/CVAT/Labelbox exports | [step 03 §8](steps/03-dataset-export.md) |
| Augmentation preview before training | Roboflow's dataset version preview | [step 04 §7](steps/04-training.md) |
| Lightweight model registry (compare runs, promote "active") | Labelbox/Roboflow model catalog | [step 04 §8](steps/04-training.md) |
| Bulk accept/reject on pre-labeled batches | CVAT/Roboflow's model-assisted review UI | [step 07 §6](steps/07-prelabel-loop.md) |

None of these change the architecture — they're additive within the existing spine, and
each is scoped to stay small (the annotation UI additions in step 02 are the biggest, at
roughly another 150–200 lines).
