import json
import shutil
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import yaml
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.config import settings
from app.models import Annotation, Dataset, Image, Label, Project

MIN_INSTANCES_PER_CLASS = 50
MAX_CLASS_IMBALANCE = 20
MIN_TOTAL_ANNOTATED = 100
MIN_BOX_PX_LONG_SIDE = 8
MAX_BOX_FRACTION = 0.95
IOU_DUP_THRESHOLD = 0.9


@dataclass
class PreflightReport:
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    per_class: dict[str, int] = field(default_factory=dict)
    n_train: int = 0
    n_val: int = 0


@dataclass
class ExportResult:
    dataset: Dataset | None
    path: str | None
    report: PreflightReport


def xyxy_norm_to_yolo(x1: float, y1: float, x2: float, y2: float) -> tuple[float, float, float, float]:
    """Normalised corners -> YOLO centre/size. The ONLY conversion site in the codebase."""
    return ((x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1)


def yolo_to_xyxy_norm(cx: float, cy: float, w: float, h: float) -> tuple[float, float, float, float]:
    """Inverse of xyxy_norm_to_yolo — used by the round-trip test and by the on-demand
    COCO writer, which reads the frozen YOLO files back rather than re-querying the DB."""
    return (cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2)


def split_for(sha256_hex: str, val_pct: int = 20) -> str:
    """Stable across re-exports: an image never changes side when new data is added."""
    return "val" if int(sha256_hex[:8], 16) % 100 < val_pct else "train"


def _label_remap(db: Session, project: Project) -> tuple[dict[int, int], dict[int, str]]:
    labels = db.scalars(
        select(Label).where(Label.project_id == project.id).order_by(Label.class_index)
    ).all()
    remap = {lbl.id: i for i, lbl in enumerate(labels)}
    names = {i: lbl.name for i, lbl in enumerate(labels)}
    return remap, names


def _iou(a: Annotation, b: Annotation) -> float:
    ix1, iy1 = max(a.x1, b.x1), max(a.y1, b.y1)
    ix2, iy2 = min(a.x2, b.x2), min(a.y2, b.y2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter <= 0:
        return 0.0
    area_a = (a.x2 - a.x1) * (a.y2 - a.y1)
    area_b = (b.x2 - b.x1) * (b.y2 - b.y1)
    return inter / (area_a + area_b - inter)


def _box_long_side_after_resize(a: Annotation, img: Image, imgsz: int) -> float:
    w_px, h_px = (a.x2 - a.x1) * img.width, (a.y2 - a.y1) * img.height
    scale = imgsz / max(img.width, img.height)
    return max(w_px, h_px) * scale


def run_preflight(
    db: Session, project: Project, val_pct: int, imgsz: int
) -> tuple[PreflightReport, list[Image], dict[int, str]]:
    """Build the split and validate it. Returns (report, exportable images, image_id -> split)."""
    images = db.scalars(
        select(Image).where(Image.project_id == project.id, Image.status == "annotated")
        .order_by(Image.id)
    ).all()

    report = PreflightReport()
    if not images:
        report.errors.append("no annotated images to export")
        return report, [], {}

    _, names = _label_remap(db, project)
    per_class: dict[str, int] = {name: 0 for name in names.values()}
    val_classes: set[str] = set()
    splits: dict[int, str] = {}

    for img in images:
        split = split_for(img.sha256, val_pct)
        splits[img.id] = split

        resolved = settings.images_dir / img.rel_path
        if not resolved.is_file():
            report.errors.append(f"image {img.rel_path} is missing on disk")
            continue

        annotations = db.scalars(select(Annotation).where(Annotation.image_id == img.id)).all()
        for i, a in enumerate(annotations):
            label = db.get(Label, a.label_id)
            name = label.name if label else "?"
            per_class[name] = per_class.get(name, 0) + 1
            if split == "val":
                val_classes.add(name)

            if _box_long_side_after_resize(a, img, imgsz) < MIN_BOX_PX_LONG_SIDE:
                report.warnings.append(
                    f"{img.rel_path}: a '{name}' box is under {MIN_BOX_PX_LONG_SIDE}px on its "
                    f"long side once resized to {imgsz} — too small for the model's stride"
                )
            if (a.x2 - a.x1) * (a.y2 - a.y1) > MAX_BOX_FRACTION:
                report.warnings.append(
                    f"{img.rel_path}: a '{name}' box covers over {MAX_BOX_FRACTION:.0%} of the frame"
                )
            for b in annotations[i + 1:]:
                if a.label_id == b.label_id and _iou(a, b) > IOU_DUP_THRESHOLD:
                    report.warnings.append(
                        f"{img.rel_path}: duplicate '{name}' boxes (IoU > {IOU_DUP_THRESHOLD})"
                    )

    report.per_class = per_class
    report.n_train = sum(1 for s in splits.values() if s == "train")
    report.n_val = sum(1 for s in splits.values() if s == "val")

    for name, count in per_class.items():
        if count < MIN_INSTANCES_PER_CLASS:
            report.warnings.append(
                f"class '{name}' has only {count} instances (<{MIN_INSTANCES_PER_CLASS}) — "
                "it will not learn well"
            )
    counts = [c for c in per_class.values() if c > 0]
    if counts and max(counts) / min(counts) > MAX_CLASS_IMBALANCE:
        report.warnings.append(
            f"class balance ratio is {max(counts) / min(counts):.1f}:1 (> {MAX_CLASS_IMBALANCE}:1) — "
            "expect the rare class to be ignored"
        )
    if len(images) < MIN_TOTAL_ANNOTATED:
        report.warnings.append(
            f"only {len(images)} annotated images (< {MIN_TOTAL_ANNOTATED}) — expect a toy model"
        )

    if report.n_val == 0:
        report.errors.append(
            f"validation split is empty — annotate more images (have {report.n_train} train, 0 val)"
        )
    elif len(names) > 1 and len(val_classes) <= 1:
        report.warnings.append("validation split contains at most one class — metrics will be meaningless")

    return report, images, splits


def export_dataset(db: Session, project: Project, val_pct: int = 20, force: bool = False) -> ExportResult:
    imgsz = settings.default_imgsz
    report, images, splits = run_preflight(db, project, val_pct, imgsz)

    if report.errors or (report.warnings and not force):
        return ExportResult(dataset=None, path=None, report=report)

    remap, names = _label_remap(db, project)
    timestamp = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
    dataset_dir = settings.datasets_dir / project.slug / timestamp
    for split in ("train", "val"):
        (dataset_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (dataset_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

    for img in images:
        split = splits[img.id]
        sha16 = img.sha256[:16]
        ext = Path(img.rel_path).suffix
        shutil.copy2(settings.images_dir / img.rel_path, dataset_dir / "images" / split / f"{sha16}{ext}")

        annotations = db.scalars(select(Annotation).where(Annotation.image_id == img.id)).all()
        lines = [
            "{} {:.6f} {:.6f} {:.6f} {:.6f}".format(remap[a.label_id], *xyxy_norm_to_yolo(a.x1, a.y1, a.x2, a.y2))
            for a in annotations
        ]
        label_path = dataset_dir / "labels" / split / f"{sha16}.txt"
        label_path.write_text("\n".join(lines) + ("\n" if lines else ""))

    data_yaml = {
        "path": str(dataset_dir.resolve()),
        "train": "images/train",
        "val": "images/val",
        "names": names,
    }
    (dataset_dir / "data.yaml").write_text(yaml.safe_dump(data_yaml, sort_keys=False))

    manifest = {
        "project_id": project.id,
        "project_slug": project.slug,
        "created_at": datetime.utcnow().isoformat(),
        "val_pct": val_pct,
        "names": names,
        "label_id_to_class_index": remap,
        "n_train": report.n_train,
        "n_val": report.n_val,
        "per_class": report.per_class,
        "warnings": report.warnings,
    }
    (dataset_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))

    dataset = Dataset(
        project_id=project.id,
        dir_path=settings.rel_data_path(dataset_dir),
        val_pct=val_pct,
        n_train=report.n_train,
        n_val=report.n_val,
        manifest_json=json.dumps(manifest),
    )
    db.add(dataset)
    db.commit()
    db.refresh(dataset)

    return ExportResult(dataset=dataset, path=str(dataset_dir), report=report)


def materialise_data_yaml(dataset_dir: Path) -> Path:
    """Step 09 §3: regenerate data.yaml at training start rather than trusting the one
    written at export time. That file's `path:` key is an absolute filesystem path, and
    the export that wrote it may have run in a different environment (conda on the Mac
    vs. the Docker container's /app/data) than the one about to train — dataset_dir
    itself is already resolved through settings.data_path() by the caller, so writing
    `path: {dataset_dir}` here is always correct for wherever this process is running."""
    manifest = json.loads((dataset_dir / "manifest.json").read_text())
    data_yaml = {
        "path": str(dataset_dir.resolve()),
        "train": "images/train",
        "val": "images/val",
        "names": manifest["names"],
    }
    yaml_path = dataset_dir / "data.yaml"
    yaml_path.write_text(yaml.safe_dump(data_yaml, sort_keys=False))
    return yaml_path


def export_coco(db: Session, project: Project, dataset: Dataset) -> dict:
    """Rebuild COCO JSON from the frozen YOLO files the dataset points to, rather than
    re-querying annotations — so it always matches the exact snapshot on disk, not
    whatever the DB looks like now. Generated on demand rather than stored (§8)."""
    manifest = json.loads(dataset.manifest_json)
    names = {int(i): name for i, name in manifest["names"].items()}
    dataset_dir = settings.data_path(dataset.dir_path)

    coco_images: list[dict] = []
    coco_annotations: list[dict] = []
    ann_id = 1

    for split in ("train", "val"):
        for label_path in sorted((dataset_dir / "labels" / split).glob("*.txt")):
            sha16 = label_path.stem
            img = db.scalar(
                select(Image).where(Image.project_id == project.id, Image.sha256.startswith(sha16))
            )
            if img is None:
                continue
            image_id = int(sha16, 16)
            coco_images.append(
                {"id": image_id, "file_name": img.rel_path, "width": img.width, "height": img.height}
            )
            for line in label_path.read_text().splitlines():
                if not line.strip():
                    continue
                cls_str, cx, cy, w, h = line.split()
                x1, y1, x2, y2 = yolo_to_xyxy_norm(float(cx), float(cy), float(w), float(h))
                bw, bh = (x2 - x1) * img.width, (y2 - y1) * img.height
                coco_annotations.append({
                    "id": ann_id,
                    "image_id": image_id,
                    "category_id": int(cls_str),
                    "bbox": [x1 * img.width, y1 * img.height, bw, bh],
                    "area": bw * bh,
                    "iscrowd": 0,
                })
                ann_id += 1

    return {
        "images": coco_images,
        "annotations": coco_annotations,
        "categories": [{"id": i, "name": name} for i, name in sorted(names.items())],
    }
