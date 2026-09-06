import json

import yaml
from PIL import Image as PILImage

from app.config import settings
from app.models import Annotation, Image, Label, Project
from app.services.dataset_export import (
    export_coco,
    export_dataset,
    split_for,
    xyxy_norm_to_yolo,
    yolo_to_xyxy_norm,
)


def make_project(db, name="warehouse"):
    project = Project(name=name, slug=name)
    db.add(project)
    db.commit()
    db.refresh(project)
    return project


def make_label(db, project, name, class_index):
    label = Label(project_id=project.id, name=name, class_index=class_index)
    db.add(label)
    db.commit()
    db.refresh(label)
    return label


def make_image_file(project, sha256_hex: str, size=(100, 100)) -> str:
    dest_dir = settings.images_dir / project.slug
    dest_dir.mkdir(parents=True, exist_ok=True)
    rel_path = f"{project.slug}/{sha256_hex[:16]}.jpg"
    PILImage.new("RGB", size, color=(120, 80, 40)).save(settings.images_dir / rel_path)
    return rel_path


def make_image(db, project, sha256_hex: str, status="annotated", size=(100, 100)):
    rel_path = make_image_file(project, sha256_hex, size=size)
    img = Image(project_id=project.id, rel_path=rel_path, sha256=sha256_hex,
                width=size[0], height=size[1], status=status)
    db.add(img)
    db.commit()
    db.refresh(img)
    return img


def val_sha(i: int) -> str:
    # int("00000000", 16) % 100 == 0 < 20 -> "val" at the default val_pct=20.
    # `i` goes right after the fixed prefix so the 16-char file-name prefix stays unique.
    return f"00000000{i:08x}" + "0" * 48


def train_sha(i: int) -> str:
    # int("ffffffff", 16) % 100 == 95 >= 20 -> "train" at the default val_pct=20
    return f"ffffffff{i:08x}" + "0" * 48


# --- pure functions -------------------------------------------------------

def test_xyxy_yolo_round_trip():
    for x1, y1, x2, y2 in [(0.1, 0.2, 0.3, 0.4), (0.0, 0.0, 1.0, 1.0), (0.512340, 0.44712, 0.6, 0.7)]:
        cx, cy, w, h = xyxy_norm_to_yolo(x1, y1, x2, y2)
        rx1, ry1, rx2, ry2 = yolo_to_xyxy_norm(cx, cy, w, h)
        assert abs(rx1 - x1) < 1e-9
        assert abs(ry1 - y1) < 1e-9
        assert abs(rx2 - x2) < 1e-9
        assert abs(ry2 - y2) < 1e-9


def test_split_deterministic_and_stable():
    sha = "abc123ef" + "0" * 56
    first = split_for(sha)
    for _ in range(5):
        assert split_for(sha) == first
    assert split_for(val_sha(0)) == "val"
    assert split_for(train_sha(0)) == "train"


# --- service-level export --------------------------------------------------

def test_export_blocked_with_too_few_images(db_session):
    project = make_project(db_session)
    make_label(db_session, project, "forklift", 0)
    label = db_session.query(Label).first()
    for i in range(5):
        img = make_image(db_session, project, train_sha(i))
        db_session.add(Annotation(image_id=img.id, label_id=label.id, x1=0.1, y1=0.1, x2=0.5, y2=0.5))
    db_session.commit()

    result = export_dataset(db_session, project, val_pct=20, force=False)

    assert result.dataset is None
    assert result.path is None
    assert result.report.errors or result.report.warnings


def test_full_export_layout_and_labels(db_session):
    project = make_project(db_session)
    forklift = make_label(db_session, project, "forklift", 0)
    person = make_label(db_session, project, "person", 1)

    # enough of each class, both splits, plus one background (zero-box) image
    for i in range(20):
        img = make_image(db_session, project, train_sha(i))
        db_session.add(Annotation(image_id=img.id, label_id=forklift.id, x1=0.1, y1=0.1, x2=0.5, y2=0.5))
        db_session.add(Annotation(image_id=img.id, label_id=person.id, x1=0.6, y1=0.6, x2=0.9, y2=0.9))
    for i in range(20):
        img = make_image(db_session, project, val_sha(i))
        db_session.add(Annotation(image_id=img.id, label_id=forklift.id, x1=0.1, y1=0.1, x2=0.5, y2=0.5))
        db_session.add(Annotation(image_id=img.id, label_id=person.id, x1=0.6, y1=0.6, x2=0.9, y2=0.9))
    background = make_image(db_session, project, train_sha(999))
    db_session.commit()

    result = export_dataset(db_session, project, val_pct=20, force=True)

    assert result.dataset is not None
    dataset_dir = result.path
    from pathlib import Path
    dataset_path = Path(dataset_dir)

    assert (dataset_path / "data.yaml").is_file()
    assert (dataset_path / "manifest.json").is_file()

    data_yaml = yaml.safe_load((dataset_path / "data.yaml").read_text())
    assert data_yaml["train"] == "images/train"
    assert data_yaml["val"] == "images/val"
    assert data_yaml["names"] == {0: "forklift", 1: "person"}
    assert data_yaml["path"] == str(dataset_path.resolve())

    manifest = json.loads((dataset_path / "manifest.json").read_text())
    assert manifest["names"] == {"0": "forklift", "1": "person"}
    assert manifest["label_id_to_class_index"] == {str(forklift.id): 0, str(person.id): 1}

    # every images/** file has a matching labels/** .txt, including the empty one
    for split in ("train", "val"):
        images_dir = dataset_path / "images" / split
        labels_dir = dataset_path / "labels" / split
        img_files = sorted(images_dir.iterdir())
        assert img_files, f"expected files in {images_dir}"
        for img_file in img_files:
            label_file = labels_dir / f"{img_file.stem}.txt"
            assert label_file.is_file()

    background_label = dataset_path / "labels" / "train" / f"{background.sha256[:16]}.txt"
    assert background_label.read_text() == ""

    non_empty_label = next(
        p for p in (dataset_path / "labels" / "train").iterdir() if p.stem != background.sha256[:16]
    )
    line = non_empty_label.read_text().splitlines()[0]
    cls, cx, cy, w, h = line.split()
    assert cls in ("0", "1")
    assert 0 <= float(cx) <= 1


def test_reexport_keeps_images_in_original_split(db_session):
    project = make_project(db_session)
    label = make_label(db_session, project, "forklift", 0)
    for i in range(20):
        img = make_image(db_session, project, train_sha(i))
        db_session.add(Annotation(image_id=img.id, label_id=label.id, x1=0.1, y1=0.1, x2=0.5, y2=0.5))
    for i in range(20):
        img = make_image(db_session, project, val_sha(i))
        db_session.add(Annotation(image_id=img.id, label_id=label.id, x1=0.1, y1=0.1, x2=0.5, y2=0.5))
    db_session.commit()

    first = export_dataset(db_session, project, val_pct=20, force=True)
    assert first.dataset is not None

    for i in range(20, 30):
        img = make_image(db_session, project, train_sha(i))
        db_session.add(Annotation(image_id=img.id, label_id=label.id, x1=0.1, y1=0.1, x2=0.5, y2=0.5))
    db_session.commit()

    second = export_dataset(db_session, project, val_pct=20, force=True)
    assert second.dataset is not None

    from pathlib import Path
    first_train = {p.stem for p in (Path(first.path) / "images" / "train").iterdir()}
    first_val = {p.stem for p in (Path(first.path) / "images" / "val").iterdir()}
    second_train = {p.stem for p in (Path(second.path) / "images" / "train").iterdir()}
    second_val = {p.stem for p in (Path(second.path) / "images" / "val").iterdir()}

    assert first_train <= second_train
    assert first_val <= second_val
    assert not (first_train & second_val)
    assert not (first_val & second_train)


def test_export_blocked_when_val_split_empty(db_session):
    project = make_project(db_session)
    label = make_label(db_session, project, "forklift", 0)
    for i in range(120):
        img = make_image(db_session, project, train_sha(i))
        db_session.add(Annotation(image_id=img.id, label_id=label.id, x1=0.1, y1=0.1, x2=0.5, y2=0.5))
    db_session.commit()

    result = export_dataset(db_session, project, val_pct=20, force=True)

    assert result.dataset is None
    assert any("validation split is empty" in e for e in result.report.errors)


def test_coco_export_matches_yolo_bbox(db_session):
    project = make_project(db_session)
    label = make_label(db_session, project, "forklift", 0)
    for i in range(20):
        img = make_image(db_session, project, train_sha(i), size=(200, 100))
        db_session.add(Annotation(image_id=img.id, label_id=label.id, x1=0.1, y1=0.2, x2=0.5, y2=0.6))
    for i in range(20):
        img = make_image(db_session, project, val_sha(i), size=(200, 100))
        db_session.add(Annotation(image_id=img.id, label_id=label.id, x1=0.1, y1=0.2, x2=0.5, y2=0.6))
    db_session.commit()

    result = export_dataset(db_session, project, val_pct=20, force=True)
    assert result.dataset is not None

    coco = export_coco(db_session, project, result.dataset)

    assert coco["categories"] == [{"id": 0, "name": "forklift"}]
    assert len(coco["images"]) == 40
    assert len(coco["annotations"]) == 40

    ann = coco["annotations"][0]
    img_entry = next(im for im in coco["images"] if im["id"] == ann["image_id"])
    expected_bbox = [0.1 * img_entry["width"], 0.2 * img_entry["height"],
                      0.4 * img_entry["width"], 0.4 * img_entry["height"]]
    for got, want in zip(ann["bbox"], expected_bbox):
        assert abs(got - want) < 1e-6
