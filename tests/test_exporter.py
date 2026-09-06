import json
from pathlib import Path

import numpy as np
import pytest

from app.config import settings
from app.models import Dataset, Job, Model, Project
from app.services.exporter import (
    Detection,
    ParityImageResult,
    ParityReport,
    build_export_metadata,
    cxcywh_to_xyxy,
    decode,
    letterbox,
    nms_per_class,
    num_anchors,
    parity_check,
)


# --- pure geometry helpers ----------------------------------------------------------

def test_cxcywh_to_xyxy():
    boxes = np.array([[10.0, 10.0, 4.0, 2.0]])
    np.testing.assert_allclose(cxcywh_to_xyxy(boxes), [[8.0, 9.0, 12.0, 11.0]])


def test_nms_per_class_suppresses_only_within_a_class():
    # Two heavily-overlapping boxes of the same class, one of a different class at the
    # same location: the same-class pair collapses to one, the other class survives.
    boxes = np.array([
        [0, 0, 10, 10],
        [1, 1, 11, 11],
        [0, 0, 10, 10],
    ], dtype=np.float32)
    scores = np.array([0.9, 0.8, 0.7])
    classes = np.array([0, 0, 1])

    kept_boxes, kept_scores, kept_classes = nms_per_class(boxes, scores, classes, iou_thresh=0.5)
    assert sorted(kept_classes.tolist()) == [0, 1]
    assert kept_scores.tolist() == [0.9, 0.7]


def test_decode_thresholds_and_drops_low_confidence():
    # [1, 4+nc, N] with nc=2, N=3: one confident detection, two below conf.
    raw = np.zeros((1, 6, 3), dtype=np.float32)
    raw[0, :, 0] = [50, 50, 20, 20, 0.9, 0.1]     # class 0, high score
    raw[0, :, 1] = [10, 10, 5, 5, 0.05, 0.02]     # below conf on both classes
    raw[0, :, 2] = [30, 30, 10, 10, 0.1, 0.6]     # class 1, high score, far from #0

    boxes, scores, classes = decode(raw, conf=0.25, iou=0.5, nc=2)
    assert sorted(classes.tolist()) == [0, 1]
    assert boxes.shape == (2, 4)


def test_decode_returns_empty_when_nothing_clears_conf():
    raw = np.zeros((1, 6, 1), dtype=np.float32)
    raw[0, :, 0] = [10, 10, 5, 5, 0.05, 0.02]
    boxes, scores, classes = decode(raw, conf=0.25, iou=0.5, nc=2)
    assert boxes.shape == (0, 4)
    assert scores.shape == (0,)


def test_letterbox_pads_to_a_square_with_the_documented_value():
    img = np.zeros((100, 200, 3), dtype=np.uint8)          # wide image
    canvas, scale, left, top = letterbox(img, imgsz=64, pad_value=114)
    assert canvas.shape == (64, 64, 3)
    assert scale == pytest.approx(64 / 200)
    assert top > 0 and left == 0                            # padded top/bottom, not sides
    assert canvas[0, 0].tolist() == [114, 114, 114]          # the pad band itself


def test_num_anchors_matches_the_known_640_case():
    assert num_anchors(640) == 8400                          # 80^2 + 40^2 + 20^2


def test_build_export_metadata_documents_the_head_shape():
    meta = build_export_metadata(
        name="warehouse-v3", task="detect", imgsz=640, opset=12, nms=False,
        classes={"0": "forklift", "1": "person"}, metrics={"mAP50-95": 0.57},
        trained_at="2026-09-06T14:12:30Z", dataset="warehouse/20260906-141230",
        framework_version="ultralytics 8.x", parity=None,
    )
    assert meta["input"]["preprocess"] == "letterbox, pad value 114"
    assert meta["output"]["shape"] == [1, 6, 8400]
    assert meta["nms_in_graph"] is False
    assert meta["parity"] is None


# --- parity check, with fake detectors so this stays fast and deterministic --------

class FakeDetector:
    def __init__(self, dets: list[Detection]):
        self._dets = dets

    def predict(self, image, conf, iou):
        return self._dets


def det(cls, score, box):
    return Detection(cls=cls, score=score, xyxy=np.array(box, dtype=np.float32))


def test_parity_check_passes_when_detections_agree():
    torch_dets = [det(0, 0.91, [10, 10, 50, 50])]
    onnx_dets = [det(0, 0.905, [10.5, 9.5, 50.5, 50.0])]      # within tolerance
    report = parity_check(FakeDetector(torch_dets), FakeDetector(onnx_dets), [Path("a.jpg")])
    assert report.passed
    assert report.images[0].passed


def test_parity_check_flags_a_detection_count_mismatch():
    torch_dets = [det(0, 0.9, [0, 0, 10, 10]), det(0, 0.8, [20, 20, 30, 30])]
    onnx_dets = [det(0, 0.9, [0, 0, 10, 10])]
    report = parity_check(FakeDetector(torch_dets), FakeDetector(onnx_dets), [Path("a.jpg")])
    assert not report.passed
    assert "count differs" in report.images[0].detail


def test_parity_check_flags_a_class_mismatch():
    torch_dets = [det(0, 0.9, [0, 0, 10, 10])]
    onnx_dets = [det(1, 0.9, [0, 0, 10, 10])]
    report = parity_check(FakeDetector(torch_dets), FakeDetector(onnx_dets), [Path("a.jpg")])
    assert not report.passed
    assert "class differs" in report.images[0].detail


def test_parity_check_flags_a_score_drift_beyond_tolerance():
    torch_dets = [det(0, 0.90, [0, 0, 10, 10])]
    onnx_dets = [det(0, 0.50, [0, 0, 10, 10])]
    report = parity_check(FakeDetector(torch_dets), FakeDetector(onnx_dets), [Path("a.jpg")],
                          tol_score=0.02)
    assert not report.passed
    assert "score differs" in report.images[0].detail


def test_parity_check_flags_boxes_shifted_the_way_a_wrong_imgsz_export_would():
    """Acceptance criteria: 'deliberately export with the wrong imgsz -> parity fails
    loudly'. A wrong export imgsz shows up exactly this way — boxes correct in shape but
    off by roughly the ratio between the two sizes, well past the pixel tolerance."""
    torch_dets = [det(0, 0.9, [100, 100, 200, 200])]
    onnx_dets = [det(0, 0.9, [50, 50, 100, 100])]     # e.g. computed against half the imgsz
    report = parity_check(FakeDetector(torch_dets), FakeDetector(onnx_dets), [Path("a.jpg")],
                          tol_box=2.0)
    assert not report.passed
    assert "box differs" in report.images[0].detail


def test_parity_check_reports_per_image_and_only_fails_the_bad_one():
    good = ([det(0, 0.9, [0, 0, 10, 10])], [det(0, 0.9, [0, 0, 10, 10])])
    bad = ([det(0, 0.9, [0, 0, 10, 10])], [det(1, 0.9, [0, 0, 10, 10])])

    class MultiDetector:
        def __init__(self, per_image):
            self._per_image = per_image
            self._i = -1

        def predict(self, image, conf, iou):
            self._i += 1
            return self._per_image[self._i]

    report = parity_check(
        MultiDetector([good[0], bad[0]]), MultiDetector([good[1], bad[1]]),
        [Path("good.jpg"), Path("bad.jpg")],
    )
    assert not report.passed
    assert report.images[0].passed and not report.images[1].passed


# --- worker wiring: run_export merges metadata and flags the model row -------------

def make_project(db, name="warehouse"):
    project = Project(name=name, slug=name)
    db.add(project)
    db.commit()
    db.refresh(project)
    return project


def make_dataset_with_val_images(db, project, n=3):
    dataset_dir = settings.datasets_dir / project.slug / "20260101-000000"
    (dataset_dir / "images" / "val").mkdir(parents=True, exist_ok=True)
    for i in range(n):
        (dataset_dir / "images" / "val" / f"{i}.jpg").write_bytes(b"fake")
    dataset = Dataset(project_id=project.id, dir_path=str(dataset_dir), val_pct=20,
                      n_train=8, n_val=n,
                      manifest_json=json.dumps({"names": {"0": "forklift"}}))
    db.add(dataset)
    db.commit()
    db.refresh(dataset)
    return dataset


def make_exportable_model(db, project, dataset):
    model_dir = settings.models_dir / "pending"
    model_dir.mkdir(parents=True, exist_ok=True)
    (model_dir / "best.pt").write_bytes(b"fake-weights")
    (model_dir / "metadata.json").write_text(json.dumps({"imgsz": 640, "trained_at": "then"}))

    model = Model(project_id=project.id, dataset_id=dataset.id, name="run0",
                  dir_path=str(model_dir), classes_json=json.dumps({"0": "forklift"}),
                  metrics_json=json.dumps({"mAP50-95": 0.5}))
    db.add(model)
    db.commit()
    db.refresh(model)
    return model


def fake_export_onnx(onnx_path: Path):
    """A stand-in for exporter.export_onnx: write_bytes() returns a byte count, so
    `write_bytes(...) or onnx_path` would return that count on the first call — this
    avoids the trap by returning the path explicitly."""
    def _export(*a, **k):
        onnx_path.write_bytes(b"x")
        return onnx_path
    return _export


def make_export_job(db, model, **config):
    job = Job(type="export", project_id=model.project_id, status="running",
              params_json=json.dumps({"model_id": model.id, **config}))
    db.add(job)
    db.commit()
    db.refresh(job)
    return job


def test_run_export_writes_metadata_and_flags_a_passing_parity(db_session, monkeypatch):
    from app import worker

    project = make_project(db_session)
    dataset = make_dataset_with_val_images(db_session, project)
    model = make_exportable_model(db_session, project, dataset)
    job = make_export_job(db_session, model, opset=13)

    onnx_path = Path(model.dir_path) / "model.onnx"
    monkeypatch.setattr(worker, "export_onnx", fake_export_onnx(onnx_path))
    monkeypatch.setattr(worker, "check_onnx_model", lambda p: None)
    monkeypatch.setattr(worker, "TorchDetector", lambda *a, **k: object())
    monkeypatch.setattr(worker, "OnnxDetector", lambda *a, **k: object())
    monkeypatch.setattr(
        worker, "parity_check",
        lambda *a, **k: ParityReport(passed=True, images=[ParityImageResult("x", True, "ok")]),
    )

    result = worker.run_export(db_session, job)
    assert result == {"model_id": model.id, "onnx_path": str(onnx_path), "parity_status": "passed"}

    db_session.rollback()
    refreshed = db_session.get(Model, model.id)
    assert refreshed.onnx_path == str(onnx_path)
    assert refreshed.parity_status == "passed"
    assert json.loads(refreshed.parity_json)["passed"] is True

    metadata = json.loads((Path(model.dir_path) / "metadata.json").read_text())
    assert metadata["opset"] == 13
    assert metadata["imgsz"] == 640                      # inherited from training metadata
    assert metadata["dataset"] == f"{project.slug}/{Path(dataset.dir_path).name}"
    assert metadata["parity"]["passed"] is True
    assert metadata["trained_at"] == "then"              # not clobbered by the export step


def test_run_export_flags_a_model_that_fails_parity(db_session, monkeypatch):
    """'A model that fails parity should be visibly flagged in the UI, not silently
    listed' — the flag is parity_status, and this proves a failing check actually sets it."""
    from app import worker

    project = make_project(db_session)
    dataset = make_dataset_with_val_images(db_session, project)
    model = make_exportable_model(db_session, project, dataset)
    job = make_export_job(db_session, model)

    onnx_path = Path(model.dir_path) / "model.onnx"
    monkeypatch.setattr(worker, "export_onnx", fake_export_onnx(onnx_path))
    monkeypatch.setattr(worker, "check_onnx_model", lambda p: None)
    monkeypatch.setattr(worker, "TorchDetector", lambda *a, **k: object())
    monkeypatch.setattr(worker, "OnnxDetector", lambda *a, **k: object())
    monkeypatch.setattr(
        worker, "parity_check",
        lambda *a, **k: ParityReport(passed=False,
                                     images=[ParityImageResult("x", False, "box differs")]),
    )

    result = worker.run_export(db_session, job)
    assert result["parity_status"] == "failed"
    db_session.rollback()
    assert db_session.get(Model, model.id).parity_status == "failed"


def test_run_export_skips_parity_without_dataset_images(db_session, monkeypatch):
    from app import worker

    project = make_project(db_session)
    model_dir = settings.models_dir / "no-dataset"
    model_dir.mkdir(parents=True)
    (model_dir / "best.pt").write_bytes(b"fake-weights")
    model = Model(project_id=project.id, dataset_id=None, name="run0", dir_path=str(model_dir),
                 classes_json=json.dumps({"0": "forklift"}))
    db_session.add(model)
    db_session.commit()
    db_session.refresh(model)
    job = make_export_job(db_session, model, imgsz=320)

    onnx_path = model_dir / "model.onnx"
    monkeypatch.setattr(worker, "export_onnx", fake_export_onnx(onnx_path))
    monkeypatch.setattr(worker, "check_onnx_model", lambda p: None)

    result = worker.run_export(db_session, job)
    assert result["parity_status"] is None
    db_session.rollback()
    refreshed = db_session.get(Model, model.id)
    assert refreshed.parity_status is None
    assert refreshed.parity_json is None


# --- real export, gated on the packages actually being installed -------------------

def test_export_and_parity_check_end_to_end(db_session, tmp_path):
    """Acceptance criteria: a real onnx.checker pass and a real parity check on a real
    image, using a pretrained checkpoint so this doesn't depend on training converging."""
    pytest.importorskip("onnx")
    pytest.importorskip("onnxruntime")
    pytest.importorskip("ultralytics")

    from ultralytics import YOLO
    from ultralytics.utils import ASSETS

    from app.services.exporter import (
        OnnxDetector,
        TorchDetector,
        check_onnx_model,
        export_onnx,
        parity_check,
    )
    from app.services.trainer import resolve_weights

    weights_path = Path(resolve_weights("yolo11n.pt"))
    try:
        YOLO(str(weights_path))       # triggers the download into settings.pretrained_dir
    except Exception as exc:
        pytest.skip(f"could not fetch pretrained weights (offline?): {exc}")

    imgsz = 160
    onnx_path = export_onnx(weights_path, tmp_path / "model", imgsz=imgsz, opset=12)
    assert onnx_path.is_file()
    check_onnx_model(onnx_path)

    sample = ASSETS / "bus.jpg"
    report = parity_check(TorchDetector(weights_path, imgsz), OnnxDetector(onnx_path), [sample])
    assert report.passed, report.to_dict()

    # A real "wrong imgsz" bug: compare against torch inference run at a *different* size
    # than the graph was exported for. The mismatch this produces is what the acceptance
    # criteria calls "parity fails loudly" — proving the check isn't a rubber stamp.
    mismatched = parity_check(TorchDetector(weights_path, 640), OnnxDetector(onnx_path), [sample])
    assert not mismatched.passed
