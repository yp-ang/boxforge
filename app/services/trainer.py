"""The only module in the codebase that imports ultralytics (ARCHITECTURE §5.1).

Every ultralytics import is inside a function, not at module scope: the web process
imports this module to read TrainConfig and to build the augmentation preview, and
paying a multi-second torch import on app boot for the sake of a dataclass is a bad
trade. The heavy imports happen in the worker subprocess, where they belong.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from app.config import settings

# Checkpoints we will fetch on demand. Always a .pt — see step 04 §3: fine-tuning COCO
# weights on a few hundred images works, training the same .yaml from scratch does not.
KNOWN_MODELS = ["yolo11n.pt", "yolo11s.pt", "yolo11m.pt", "yolo11l.pt", "yolo11x.pt"]


@dataclass
class TrainConfig:
    model: str = "yolo11n.pt"     # n=nano (fast, weak) … s, m, l, x
    epochs: int = 100
    imgsz: int = 640
    batch: int = -1               # -1 = auto-fit to available memory
    patience: int = 30            # early stop
    device: str = "auto"
    seed: int = 0                 # reproducibility
    fliplr: float = 0.5           # set 0.0 for orientation-sensitive objects (§3)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class TrainResult:
    weights: Path
    run_dir: Path
    metrics: dict[str, float] = field(default_factory=dict)


class Trainer(Protocol):
    def train(self, dataset_yaml: Path, cfg: TrainConfig, run_dir: Path) -> TrainResult: ...


def resolve_device(pref: str) -> str:
    """ARCHITECTURE §5.3: CUDA if present, else MPS on Apple Silicon, else CPU."""
    if pref != "auto":
        return pref
    import torch

    if torch.cuda.is_available():
        return "0"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def resolve_weights(model: str) -> str:
    """Keep pretrained checkpoints under data/ (README ground rule 4). Ultralytics
    downloads a bare name like 'yolo11n.pt' into the process CWD; handing it an absolute
    path under data/pretrained/ puts it where the Docker volume already mounts."""
    candidate = Path(model)
    if candidate.is_absolute() or candidate.parent != Path("."):
        return str(candidate)
    settings.pretrained_dir.mkdir(parents=True, exist_ok=True)
    return str((settings.pretrained_dir / candidate.name).resolve())


def _clean_metric_keys(raw: dict[str, Any]) -> dict[str, float]:
    """'metrics/mAP50-95(B)' -> 'mAP50-95'. Values that will not float (ultralytics puts
    the odd string in there) are dropped rather than blowing up the whole run's report."""
    out: dict[str, float] = {}
    for key, value in raw.items():
        try:
            out[str(key).replace("metrics/", "").replace("(B)", "")] = float(value)
        except (TypeError, ValueError):
            continue
    return out


def extract_metrics(results: Any) -> dict[str, float]:
    """What model.train() hands back is not one type.

    Normally it is a DetMetrics carrying .results_dict. But when the validator object is
    gone by the time train() returns, ultralytics falls back to the plain dict it stashed
    in the checkpoint under 'train_metrics' — same keys, no attributes. Both shapes turn
    up in practice on the same version, so handle the object and the bare dict alike.
    """
    if isinstance(results, dict):
        return _clean_metric_keys(results)

    value = getattr(results, "results_dict", None)
    if isinstance(value, dict):
        return _clean_metric_keys(value)
    return {}


def metrics_from_results_csv(run_dir: Path) -> dict[str, float]:
    """Last resort, and the most durable one: ultralytics writes results.csv every epoch
    regardless of what train() chooses to return. The final row is the final epoch."""
    import csv

    path = run_dir / "results.csv"
    if not path.is_file():
        return {}
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return {}
    return {k: v for k, v in _clean_metric_keys(rows[-1]).items() if k.startswith("mAP")
            or k in ("precision", "recall")}


class UltralyticsTrainer:
    """AGPL-3.0 (ARCHITECTURE §5.1). Swap this class, not the codebase."""

    def train(self, dataset_yaml: Path, cfg: TrainConfig, run_dir: Path) -> TrainResult:
        from ultralytics import YOLO

        device = resolve_device(cfg.device)
        print(f"[trainer] model={cfg.model} device={device} epochs={cfg.epochs} "
              f"imgsz={cfg.imgsz} batch={cfg.batch} seed={cfg.seed} fliplr={cfg.fliplr}",
              flush=True)

        model = YOLO(resolve_weights(cfg.model))
        results = model.train(
            data=str(dataset_yaml), epochs=cfg.epochs, imgsz=cfg.imgsz,
            batch=cfg.batch, patience=cfg.patience, device=device,
            seed=cfg.seed, project=str(run_dir.parent), name=run_dir.name,
            exist_ok=True, plots=True, fliplr=cfg.fliplr,
        )
        best = run_dir / "weights" / "best.pt"
        if not best.is_file():                       # patience=0 runs, or a 1-epoch smoke test
            best = run_dir / "weights" / "last.pt"

        metrics = extract_metrics(results) or metrics_from_results_csv(run_dir)
        if not metrics:
            print("[trainer] no metrics recovered from train() or results.csv", flush=True)
        return TrainResult(weights=best, run_dir=run_dir, metrics=metrics)


# --- augmentation preview (§7) ---------------------------------------------------

def _hyp(cfg: TrainConfig):
    """Ultralytics' own defaults, with the two knobs this app exposes overridden."""
    from ultralytics.cfg import get_cfg

    overrides = {"imgsz": cfg.imgsz, "fliplr": cfg.fliplr}
    return get_cfg(overrides=overrides)


class _PreviewDataset:
    """The minimum surface Mosaic pokes at: a length, an indexable get_image_and_label,
    and the two cache attributes it reads. Real training uses a real YOLODataset; this
    exists so the preview shows the actual transforms rather than an imitation of them."""

    def __init__(self, images: list[np.ndarray], imgsz: int):
        from ultralytics.utils.instance import Instances

        self.images = images
        self.imgsz = imgsz
        # Mosaic reads these two: with cache != "ram" it draws its other three tiles from
        # `buffer`, which for a real dataset is a rolling window of recently loaded images
        # and here would be empty (IndexError on random.choices). Saying "ram" — true, the
        # preview images are already in memory — makes it pick from the whole set instead.
        self.cache = "ram"
        self.buffer: list[int] = list(range(len(images)))
        self._Instances = Instances

    def __len__(self) -> int:
        return len(self.images)

    def get_image_and_label(self, index: int) -> dict[str, Any]:
        im = self.images[index % len(self.images)]
        h, w = im.shape[:2]
        empty = np.zeros((0, 4), dtype=np.float32)
        return {
            "img": im, "cls": np.zeros((0, 1), dtype=np.float32),
            "instances": self._Instances(empty, bbox_format="xywh", normalized=False),
            "im_file": f"preview-{index}", "ori_shape": (h, w),
            "resized_shape": (h, w), "ratio_pad": (1.0, 1.0),
            "rect_shape": None, "mosaic_border": (-self.imgsz // 2, -self.imgsz // 2),
        }


def build_preview_transforms(images: list[np.ndarray], cfg: TrainConfig):
    """Mosaic needs a dataset to draw its other three tiles from; the rest are per-image.
    If a future ultralytics release moves these, fall back to the per-image transforms
    rather than failing the whole preview — a preview without mosaic still catches the
    mirrored '7' this feature exists for."""
    from ultralytics.data.augment import Compose, RandomFlip, RandomHSV, RandomPerspective

    hyp = _hyp(cfg)
    per_image = [
        # size= is what crops the 2x mosaic canvas back to imgsz, exactly as
        # v8_transforms wires it — without it the preview comes back double-size and
        # stops being a picture of what the model will actually see.
        RandomPerspective(degrees=hyp.degrees, translate=hyp.translate, scale=hyp.scale,
                          shear=hyp.shear, perspective=hyp.perspective,
                          size=(cfg.imgsz, cfg.imgsz)),
        RandomHSV(hgain=hyp.hsv_h, sgain=hyp.hsv_s, vgain=hyp.hsv_v),
        RandomFlip(direction="vertical", p=hyp.flipud),
        RandomFlip(direction="horizontal", p=cfg.fliplr),
    ]
    try:
        from ultralytics.data.augment import Mosaic

        mosaic = Mosaic(_PreviewDataset(images, cfg.imgsz), imgsz=cfg.imgsz, p=hyp.mosaic)
        return Compose([mosaic, *per_image])
    except Exception as exc:                       # noqa: BLE001 — preview is best-effort
        print(f"[preview] mosaic unavailable ({exc}); showing per-image transforms only")
        return Compose(per_image)


def augment_preview(images: list[np.ndarray], cfg: TrainConfig) -> list[np.ndarray]:
    """One augmented variant per input image, in the same order."""
    transforms = build_preview_transforms(images, cfg)
    dataset = _PreviewDataset(images, cfg.imgsz)
    out: list[np.ndarray] = []
    for i in range(len(images)):
        labels = dataset.get_image_and_label(i)
        out.append(transforms(labels)["img"])
    return out


def write_model_folder(dest: Path, weights: Path, names: dict[str, str],
                       metadata: dict[str, Any]) -> None:
    """The contract ARCHITECTURE §2 publishes, minus model.onnx — step 05 adds that."""
    import shutil

    dest.mkdir(parents=True, exist_ok=True)
    if weights.is_file():
        shutil.copy2(weights, dest / "best.pt")
    (dest / "classes.json").write_text(json.dumps(names, indent=2))
    (dest / "metadata.json").write_text(json.dumps(metadata, indent=2))
