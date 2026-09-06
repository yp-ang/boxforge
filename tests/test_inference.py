import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from app.config import settings
from app.models import Job, Model, Project
from app.services.runtime import Det, color_for_class, draw_detections, get_detector, invalidate


def make_project(db, name="warehouse"):
    project = Project(name=name, slug=name)
    db.add(project)
    db.commit()
    db.refresh(project)
    return project


def make_model(db, project, name, onnx=False):
    model_dir = settings.models_dir / name
    model = Model(project_id=project.id, name=name, dir_path=str(model_dir),
                 classes_json=json.dumps({"0": "forklift"}))
    if onnx:
        model_dir.mkdir(parents=True, exist_ok=True)
        onnx_path = model_dir / "model.onnx"
        onnx_path.write_bytes(b"fake-onnx")
        model.onnx_path = str(onnx_path)
    db.add(model)
    db.commit()
    db.refresh(model)
    return model


# --- runtime: pure helpers ----------------------------------------------------------

def test_color_for_class_is_stable_and_wraps():
    assert color_for_class(0) == color_for_class(0)
    from app.services.runtime import PALETTE
    assert color_for_class(len(PALETTE)) == color_for_class(0)      # wraps around


def test_draw_detections_paints_into_the_frame():
    frame = np.zeros((50, 50, 3), dtype=np.uint8)
    dets = [Det(cls=0, label="forklift", score=0.87, xyxy=[5.0, 5.0, 30.0, 30.0])]
    draw_detections(frame, dets)
    assert frame.any()      # no longer all-black once a box + label chip is drawn


def test_draw_detections_handles_an_empty_list():
    frame = np.zeros((10, 10, 3), dtype=np.uint8)
    draw_detections(frame, [])
    assert not frame.any()


class _CountingFakeDetector:
    instances = 0

    def __init__(self, model_dir):
        _CountingFakeDetector.instances += 1
        self.model_dir = model_dir


def test_get_detector_caches_by_model_id(monkeypatch):
    monkeypatch.setattr("app.services.runtime.OnnxDetector", _CountingFakeDetector)
    _CountingFakeDetector.instances = 0

    first = get_detector(101, Path("/models/101"))
    second = get_detector(101, Path("/models/101"))
    assert first is second
    assert _CountingFakeDetector.instances == 1

    third = get_detector(102, Path("/models/102"))
    assert third is not first
    assert _CountingFakeDetector.instances == 2

    invalidate(101)
    fourth = get_detector(101, Path("/models/101"))
    assert fourth is not first
    assert _CountingFakeDetector.instances == 3


# --- predict/image: validation --------------------------------------------------

def test_predict_image_404s_on_an_unknown_model(client):
    resp = client.post("/api/models/999/predict/image", files={"file": ("x.jpg", b"x", "image/jpeg")})
    assert resp.status_code == 404


def test_predict_image_requires_an_exported_onnx(client, db_session):
    project = make_project(db_session)
    model = make_model(db_session, project, "run0", onnx=False)
    resp = client.post(f"/api/models/{model.id}/predict/image",
                       files={"file": ("x.jpg", b"x", "image/jpeg")})
    assert resp.status_code == 400
    assert "export" in resp.json()["detail"]


def test_predict_image_rejects_an_undecodable_upload(client, db_session):
    project = make_project(db_session)
    model = make_model(db_session, project, "run0", onnx=True)
    resp = client.post(f"/api/models/{model.id}/predict/image",
                       files={"file": ("x.jpg", b"not-an-image", "image/jpeg")})
    assert resp.status_code == 400


# --- predict/image: real model, end to end (gated on packages + network) ------------

def test_predict_image_endpoint_finds_boxes_on_a_real_model(client, db_session):
    """Acceptance criteria: 'upload an image -> boxes with confidences, sensible
    positions', run against the real pipeline this step exists to prove — the exported
    ONNX, through onnxruntime, exactly as services/exporter.py's own end-to-end test
    proves the export step."""
    pytest.importorskip("onnx")
    pytest.importorskip("onnxruntime")
    pytest.importorskip("ultralytics")

    from ultralytics import YOLO
    from ultralytics.utils import ASSETS

    from app.services.exporter import export_onnx
    from app.services.trainer import resolve_weights

    weights_path = Path(resolve_weights("yolo11n.pt"))
    try:
        yolo = YOLO(str(weights_path))       # triggers the download if not cached
    except Exception as exc:
        pytest.skip(f"could not fetch pretrained weights (offline?): {exc}")

    project = make_project(db_session, "verify-proj")
    imgsz = 320
    model_dir = settings.models_dir / "verify-model"
    onnx_path = export_onnx(weights_path, model_dir, imgsz=imgsz, opset=12)

    coco_names = {str(k): v for k, v in yolo.names.items()}
    (model_dir / "classes.json").write_text(json.dumps(coco_names))
    (model_dir / "metadata.json").write_text(json.dumps({"imgsz": imgsz}))

    model = Model(project_id=project.id, name="verify-model", dir_path=str(model_dir),
                 classes_json=json.dumps(coco_names), onnx_path=str(onnx_path))
    db_session.add(model)
    db_session.commit()
    db_session.refresh(model)

    sample = ASSETS / "bus.jpg"
    with open(sample, "rb") as f:
        resp = client.post(f"/api/models/{model.id}/predict/image",
                           files={"file": ("bus.jpg", f, "image/jpeg")})
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["width"] > 0 and body["height"] > 0
    assert body["image"].startswith("data:image/jpeg;base64,")
    # bus.jpg is ultralytics' own smoke-test image: a bus and several people, easily
    # found by an un-fine-tuned COCO checkpoint.
    assert any(d["label"] == "bus" and d["score"] > 0.3 for d in body["detections"])
    assert all(0.0 <= d["score"] <= 1.0 for d in body["detections"])
    for d in body["detections"]:
        x1, y1, x2, y2 = d["xyxy"]
        assert 0 <= x1 < x2 <= body["width"]
        assert 0 <= y1 < y2 <= body["height"]


# --- predict/video: endpoint validation and job wiring ------------------------------

def test_predict_video_requires_an_exported_onnx(client, db_session):
    project = make_project(db_session)
    model = make_model(db_session, project, "run0", onnx=False)
    resp = client.post(f"/api/models/{model.id}/predict/video",
                       files={"file": ("v.mp4", b"fake", "video/mp4")})
    assert resp.status_code == 400


def test_predict_video_starts_a_job_and_saves_the_upload(client, db_session, monkeypatch):
    from app.services import jobs as jobs_module
    monkeypatch.setattr(jobs_module, "WORKER_MODULE", "tests.fake_worker")

    project = make_project(db_session)
    model = make_model(db_session, project, "run0", onnx=True)

    resp = client.post(
        f"/api/models/{model.id}/predict/video?stride=5&conf=0.3&iou=0.4",
        files={"file": ("v.mp4", b"fake-video-bytes", "video/mp4")},
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["type"] == "predict_video"
    params = json.loads(body["params_json"])
    assert params == {"model_id": model.id, "src_path": params["src_path"],
                      "stride": 5, "conf": 0.3, "iou": 0.4}
    assert Path(params["src_path"]).read_bytes() == b"fake-video-bytes"

    client.post(f"/api/jobs/{body['id']}/cancel")


def test_predict_video_refuses_a_second_run(client, db_session, monkeypatch):
    from app.services import jobs as jobs_module
    monkeypatch.setattr(jobs_module, "WORKER_MODULE", "tests.fake_worker")

    project = make_project(db_session)
    model = make_model(db_session, project, "run0", onnx=True)

    first = client.post(f"/api/models/{model.id}/predict/video",
                        files={"file": ("v.mp4", b"x", "video/mp4")})
    assert first.status_code == 200

    second = client.post(f"/api/models/{model.id}/predict/video",
                         files={"file": ("v.mp4", b"x", "video/mp4")})
    assert second.status_code == 409

    client.post(f"/api/jobs/{first.json()['id']}/cancel")


def test_job_video_and_detections_404_before_ready(client, db_session):
    project = make_project(db_session)
    job = Job(type="predict_video", project_id=project.id, status="running", params_json="{}")
    db_session.add(job)
    db_session.commit()
    db_session.refresh(job)

    assert client.get(f"/api/jobs/{job.id}/video").status_code == 404
    assert client.get(f"/api/jobs/{job.id}/detections").status_code == 404


def test_job_video_and_detections_serve_the_written_files(client, db_session):
    project = make_project(db_session)
    job = Job(type="predict_video", project_id=project.id, status="done", params_json="{}")
    db_session.add(job)
    db_session.commit()
    db_session.refresh(job)

    out_dir = settings.runs_dir / str(job.id) / "predict"
    out_dir.mkdir(parents=True)
    (out_dir / "annotated.mp4").write_bytes(b"fake-mp4-bytes")
    (out_dir / "detections.json").write_text(json.dumps({"frames": []}))

    video_resp = client.get(f"/api/jobs/{job.id}/video")
    assert video_resp.status_code == 200
    assert video_resp.content == b"fake-mp4-bytes"

    det_resp = client.get(f"/api/jobs/{job.id}/detections")
    assert det_resp.status_code == 200
    assert det_resp.json() == {"frames": []}


def test_job_video_404s_on_a_job_of_the_wrong_type(client, db_session):
    project = make_project(db_session)
    job = Job(type="train", project_id=project.id, status="done", params_json="{}")
    db_session.add(job)
    db_session.commit()
    db_session.refresh(job)
    assert client.get(f"/api/jobs/{job.id}/video").status_code == 404


# --- worker: run_predict_video --------------------------------------------------

class _FakeVideoDetector:
    """Deterministic stand-in for services.runtime.OnnxDetector — records every call so
    the stride behaviour (infer every Nth frame, reuse boxes in between) is checkable
    without a real model."""

    def __init__(self, model_dir):
        self.model_dir = model_dir

    def predict(self, frame, conf, iou):
        return [Det(cls=0, label="thing", score=0.9, xyxy=[1.0, 1.0, 5.0, 5.0])]


def _write_fake_video(path: Path, n_frames: int, size=(32, 24), fps=10):
    w, h = size
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    for i in range(n_frames):
        writer.write(np.full((h, w, 3), i % 255, dtype=np.uint8))
    writer.release()


def test_run_predict_video_samples_every_stride_frame(db_session, monkeypatch):
    from app import worker

    monkeypatch.setattr(worker, "RuntimeOnnxDetector", _FakeVideoDetector)

    project = make_project(db_session)
    model = make_model(db_session, project, "run0", onnx=True)

    upload_dir = settings.data_dir / "uploads"
    upload_dir.mkdir(parents=True, exist_ok=True)
    src_path = upload_dir / "in.mp4"
    _write_fake_video(src_path, n_frames=9)

    job = Job(type="predict_video", project_id=project.id, status="running",
              params_json=json.dumps({"model_id": model.id, "src_path": str(src_path),
                                      "stride": 3, "conf": 0.25, "iou": 0.45}))
    db_session.add(job)
    db_session.commit()
    db_session.refresh(job)

    result = worker.run_predict_video(db_session, job)

    out_dir = settings.runs_dir / str(job.id) / "predict"
    assert (out_dir / "annotated.mp4").is_file()
    detections = json.loads((out_dir / "detections.json").read_text())
    assert detections["stride"] == 3
    assert [f["frame"] for f in detections["frames"]] == [0, 3, 6]
    assert result["frames"] == 9
    assert not src_path.exists()          # scratch upload cleaned up after the run


def test_run_predict_video_raises_without_an_export(db_session):
    from app import worker

    project = make_project(db_session)
    model = make_model(db_session, project, "run0", onnx=False)
    job = Job(type="predict_video", project_id=project.id, status="running",
              params_json=json.dumps({"model_id": model.id, "src_path": "/nope.mp4"}))
    db_session.add(job)
    db_session.commit()
    db_session.refresh(job)

    with pytest.raises(RuntimeError, match="no exported ONNX"):
        worker.run_predict_video(db_session, job)


# --- webcam ------------------------------------------------------------------------

def test_webcam_capability_endpoint_reports_a_boolean(client):
    resp = client.get("/api/capabilities/webcam")
    assert resp.status_code == 200
    body = resp.json()
    assert isinstance(body["server_webcam"], bool)
    assert body["server_webcam"] or body["note"]        # a false result always explains itself


def test_ws_predict_closes_for_an_unknown_model(client):
    from starlette.testclient import WebSocketDisconnect

    with pytest.raises(WebSocketDisconnect):
        with client.websocket_connect("/ws/predict/999") as ws:
            ws.receive_bytes()


def test_ws_predict_round_trips_a_frame_and_applies_control_messages(client, db_session, monkeypatch):
    project = make_project(db_session)
    model = make_model(db_session, project, "run0", onnx=True)

    calls = []

    class RecordingDetector:
        def predict(self, frame, conf, iou):
            calls.append((conf, iou))
            return [Det(cls=0, label="x", score=0.5, xyxy=[0.0, 0.0, 1.0, 1.0])]

    monkeypatch.setattr("app.routers.inference.get_detector", lambda *a, **k: RecordingDetector())

    ok, buf = cv2.imencode(".jpg", np.zeros((10, 10, 3), dtype=np.uint8))
    assert ok
    frame_bytes = buf.tobytes()

    with client.websocket_connect(f"/ws/predict/{model.id}") as ws:
        ws.send_text(json.dumps({"conf": 0.6, "iou": 0.3}))
        ws.send_bytes(frame_bytes)
        dets = ws.receive_json()

    assert dets == [{"cls": 0, "label": "x", "score": 0.5, "xyxy": [0.0, 0.0, 1.0, 1.0]}]
    assert calls == [(0.6, 0.3)]
