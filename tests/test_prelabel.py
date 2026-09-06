import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from app.config import settings
from app.models import Annotation, Image, Job, Label, Model, Project
from app.services.prelabel import (
    DEFAULT_PRELABEL_CONF,
    DetLike,
    count_eligible,
    eligible_images,
    prelabel,
    undo_batch,
    uncertainty,
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


def make_model(db, project, name, onnx=True):
    model_dir = settings.models_dir / name
    model = Model(project_id=project.id, name=name, dir_path=str(model_dir),
                 classes_json=json.dumps({"0": "forklift"}))
    if onnx:
        model_dir.mkdir(parents=True, exist_ok=True)
        (model_dir / "model.onnx").write_bytes(b"fake-onnx")
        model.onnx_path = str(model_dir / "model.onnx")
    db.add(model)
    db.commit()
    db.refresh(model)
    return model


def make_image(db, project, rel_path="a.jpg", status="pending", write_file=True):
    if write_file:
        path = settings.images_dir / rel_path
        path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(path), np.zeros((20, 20, 3), dtype=np.uint8))
    img = Image(project_id=project.id, rel_path=rel_path, sha256=rel_path, width=20,
               height=20, status=status)
    db.add(img)
    db.commit()
    db.refresh(img)
    return img


# --- pure helpers ------------------------------------------------------------------

def test_uncertainty_of_no_detections_is_the_sentinel():
    assert uncertainty([]) == 0.9


def test_uncertainty_is_one_minus_mean_confidence():
    dets = [DetLike(cls=0, score=0.8, xyxy=[0, 0, 1, 1]), DetLike(cls=0, score=0.6, xyxy=[0, 0, 1, 1])]
    assert uncertainty(dets) == pytest.approx(1 - 0.7)


# --- eligibility: never touches human work or an existing draft --------------------

def test_eligible_images_excludes_non_pending_and_already_annotated(db_session):
    project = make_project(db_session)
    label = make_label(db_session, project, "forklift", 0)

    pending_clean = make_image(db_session, project, "clean.jpg")
    pending_with_draft = make_image(db_session, project, "draft.jpg")
    db_session.add(Annotation(image_id=pending_with_draft.id, label_id=label.id,
                              x1=0, y1=0, x2=0.5, y2=0.5, source="model"))
    make_image(db_session, project, "annotated.jpg", status="annotated")
    make_image(db_session, project, "skipped.jpg", status="skipped")
    db_session.commit()

    eligible = eligible_images(db_session, project.id, limit=100)
    assert [img.id for img in eligible] == [pending_clean.id]
    assert count_eligible(db_session, project.id, limit=100) == 1


# --- prelabel(): the drafting loop itself -------------------------------------------

class _FakeDetector:
    classes = {0: "forklift", 1: "pallet"}

    def __init__(self, dets_by_call):
        self._dets_by_call = dets_by_call
        self.calls = []

    def predict(self, frame, conf):
        self.calls.append(conf)
        return self._dets_by_call.pop(0)


def _read_image(rel_path):
    return cv2.imread(str(settings.images_dir / rel_path))


def test_prelabel_drafts_boxes_with_source_model_and_leaves_status_pending(db_session):
    project = make_project(db_session)
    make_label(db_session, project, "forklift", 0)
    img = make_image(db_session, project, "a.jpg")

    detector = _FakeDetector([[DetLike(cls=0, score=0.5, xyxy=[2.0, 2.0, 10.0, 10.0])]])
    result = prelabel(db_session, project, detector, conf=0.4, limit=10, read_image=_read_image)

    assert result == {"image_ids": [img.id], "n_images": 1, "n_boxes": 1, "unmapped_classes": []}
    assert detector.calls == [0.4]

    db_session.refresh(img)
    assert img.status == "pending"          # doc §2: a draft is not a label
    assert img.uncertainty == pytest.approx(0.5)

    annotations = db_session.query(Annotation).filter_by(image_id=img.id).all()
    assert len(annotations) == 1
    assert annotations[0].source == "model"
    # normalised against the 20x20 fake image
    assert annotations[0].x1 == pytest.approx(0.1)
    assert annotations[0].x2 == pytest.approx(0.5)


def test_prelabel_skips_detections_for_classes_not_in_the_project(db_session):
    project = make_project(db_session)
    make_label(db_session, project, "forklift", 0)   # "pallet" (class 1) has no label
    img = make_image(db_session, project, "a.jpg")

    detector = _FakeDetector([[DetLike(cls=1, score=0.9, xyxy=[0, 0, 5, 5])]])
    result = prelabel(db_session, project, detector, conf=0.4, limit=10, read_image=_read_image)

    assert result["n_boxes"] == 0
    assert result["unmapped_classes"] == ["pallet"]
    assert db_session.query(Annotation).filter_by(image_id=img.id).count() == 0


def test_prelabel_sets_the_zero_detection_sentinel(db_session):
    project = make_project(db_session)
    make_label(db_session, project, "forklift", 0)
    img = make_image(db_session, project, "a.jpg")

    detector = _FakeDetector([[]])
    prelabel(db_session, project, detector, conf=0.4, limit=10, read_image=_read_image)

    db_session.refresh(img)
    assert img.uncertainty == 0.9


def test_prelabel_never_touches_an_already_drafted_image(db_session):
    project = make_project(db_session)
    label = make_label(db_session, project, "forklift", 0)
    img = make_image(db_session, project, "a.jpg")
    db_session.add(Annotation(image_id=img.id, label_id=label.id, x1=0, y1=0, x2=0.1, y2=0.1,
                              source="model"))
    db_session.commit()

    detector = _FakeDetector([])   # never called: nothing eligible
    result = prelabel(db_session, project, detector, conf=0.4, limit=10, read_image=_read_image)
    assert result["image_ids"] == []
    assert detector.calls == []


# --- undo_batch: scoped exactly to the batch, never touches real work ---------------

def test_undo_batch_deletes_only_model_sourced_rows_on_pending_images(db_session):
    project = make_project(db_session)
    label = make_label(db_session, project, "forklift", 0)

    drafted = make_image(db_session, project, "drafted.jpg")
    drafted.uncertainty = 0.3
    since_annotated = make_image(db_session, project, "since-annotated.jpg", status="annotated")
    db_session.add_all([
        Annotation(image_id=drafted.id, label_id=label.id, x1=0, y1=0, x2=0.1, y2=0.1, source="model"),
        # A human has since saved this image for real — full-replace PUT already turned
        # every box "human" and flipped status, so undo must leave it alone.
        Annotation(image_id=since_annotated.id, label_id=label.id, x1=0, y1=0, x2=0.1, y2=0.1,
                  source="human"),
    ])
    db_session.commit()

    result = undo_batch(db_session, [drafted.id, since_annotated.id])
    assert result == {"deleted_annotations": 1, "images_reset": 1}

    assert db_session.query(Annotation).filter_by(image_id=drafted.id).count() == 0
    assert db_session.query(Annotation).filter_by(image_id=since_annotated.id).count() == 1
    db_session.refresh(drafted)
    assert drafted.uncertainty is None


def test_undo_batch_with_no_ids_is_a_noop(db_session):
    assert undo_batch(db_session, []) == {"deleted_annotations": 0, "images_reset": 0}


# --- worker.run_prelabel ------------------------------------------------------------

class _FakeRuntimeDetector:
    classes = {0: "forklift"}

    def __init__(self, model_dir):
        self.model_dir = model_dir

    def predict(self, frame, conf):
        return [DetLike(cls=0, score=0.5, xyxy=[1.0, 1.0, 5.0, 5.0])]


def test_run_prelabel_end_to_end(db_session, monkeypatch):
    from app import worker

    monkeypatch.setattr(worker, "RuntimeOnnxDetector", _FakeRuntimeDetector)

    project = make_project(db_session)
    make_label(db_session, project, "forklift", 0)
    model = make_model(db_session, project, "run0")
    img = make_image(db_session, project, "a.jpg")

    job = Job(type="prelabel", project_id=project.id, status="running",
             params_json=json.dumps({"model_id": model.id, "conf": 0.35, "limit": 10}))
    db_session.add(job)
    db_session.commit()
    db_session.refresh(job)

    result = worker.run_prelabel(db_session, job)
    assert result["n_images"] == 1
    assert result["n_boxes"] == 1
    assert result["image_ids"] == [img.id]


def test_run_prelabel_raises_without_an_export(db_session):
    from app import worker

    project = make_project(db_session)
    model = make_model(db_session, project, "run0", onnx=False)
    job = Job(type="prelabel", project_id=project.id, status="running",
             params_json=json.dumps({"model_id": model.id}))
    db_session.add(job)
    db_session.commit()
    db_session.refresh(job)

    with pytest.raises(RuntimeError, match="no exported ONNX"):
        worker.run_prelabel(db_session, job)


# --- HTTP: preview + start + conflict -----------------------------------------------

def test_prelabel_preview_counts_eligible_images(client, db_session):
    project = make_project(db_session)
    model = make_model(db_session, project, "run0")
    make_image(db_session, project, "a.jpg", write_file=False)
    make_image(db_session, project, "b.jpg", status="annotated", write_file=False)

    resp = client.get(f"/api/projects/{project.id}/prelabel-preview",
                      params={"model_id": model.id, "conf": 0.4})
    assert resp.status_code == 200
    assert resp.json() == {"eligible": 1, "model_id": model.id, "conf": 0.4}


def test_prelabel_preview_requires_an_exported_model(client, db_session):
    project = make_project(db_session)
    model = make_model(db_session, project, "run0", onnx=False)
    resp = client.get(f"/api/projects/{project.id}/prelabel-preview",
                      params={"model_id": model.id})
    assert resp.status_code == 400


def test_start_prelabel_conflicts_with_a_running_job(client, db_session, monkeypatch):
    from app.services import jobs as jobs_module
    monkeypatch.setattr(jobs_module, "WORKER_MODULE", "tests.fake_worker")

    project = make_project(db_session)
    model = make_model(db_session, project, "run0")

    first = client.post(f"/api/projects/{project.id}/prelabel",
                        json={"model_id": model.id, "conf": 0.4, "limit": 10})
    assert first.status_code == 200, first.text
    assert first.json()["type"] == "prelabel"

    second = client.post(f"/api/projects/{project.id}/prelabel",
                         json={"model_id": model.id, "conf": 0.4, "limit": 10})
    assert second.status_code == 409

    client.post(f"/api/jobs/{first.json()['id']}/cancel")


# --- HTTP: undo ----------------------------------------------------------------------

def test_undo_prelabel_endpoint(client, db_session):
    project = make_project(db_session)
    label = make_label(db_session, project, "forklift", 0)
    img = make_image(db_session, project, "a.jpg", write_file=False)
    db_session.add(Annotation(image_id=img.id, label_id=label.id, x1=0, y1=0, x2=0.1, y2=0.1,
                              source="model"))
    db_session.commit()

    job = Job(type="prelabel", project_id=project.id, status="done",
             params_json="{}", result_json=json.dumps({"image_ids": [img.id], "model_id": 1}))
    db_session.add(job)
    db_session.commit()
    db_session.refresh(job)

    resp = client.post(f"/api/jobs/{job.id}/prelabel/undo")
    assert resp.status_code == 200
    assert resp.json() == {"deleted_annotations": 1, "images_reset": 1}


def test_undo_prelabel_404s_on_a_job_of_the_wrong_type(client, db_session):
    project = make_project(db_session)
    job = Job(type="train", project_id=project.id, status="done", params_json="{}")
    db_session.add(job)
    db_session.commit()
    db_session.refresh(job)
    assert client.post(f"/api/jobs/{job.id}/prelabel/undo").status_code == 404


# --- HTTP: accept / reject / batch grid ---------------------------------------------

def test_accept_promotes_drafts_to_human_and_marks_annotated(client, db_session):
    project = make_project(db_session)
    label = make_label(db_session, project, "forklift", 0)
    img = make_image(db_session, project, "a.jpg", write_file=False)
    db_session.add(Annotation(image_id=img.id, label_id=label.id, x1=0, y1=0, x2=0.1, y2=0.1,
                              source="model"))
    db_session.commit()

    resp = client.post(f"/api/images/{img.id}/accept")
    assert resp.status_code == 200
    assert resp.json()["status"] == "annotated"

    db_session.refresh(img)
    ann = db_session.query(Annotation).filter_by(image_id=img.id).one()
    assert ann.source == "human"


def test_reject_clears_drafts_and_stays_pending(client, db_session):
    project = make_project(db_session)
    label = make_label(db_session, project, "forklift", 0)
    img = make_image(db_session, project, "a.jpg", write_file=False)
    img.uncertainty = 0.5
    db_session.add(Annotation(image_id=img.id, label_id=label.id, x1=0, y1=0, x2=0.1, y2=0.1,
                              source="model"))
    db_session.commit()

    resp = client.post(f"/api/images/{img.id}/reject")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "pending"
    assert body["uncertainty"] is None
    assert db_session.query(Annotation).filter_by(image_id=img.id).count() == 0


# --- HTTP: review ordering on the existing /next endpoint (doc §4) -----------------

def test_next_image_orders_by_uncertainty_descending(client, db_session):
    project = make_project(db_session)
    low = make_image(db_session, project, "low.jpg", write_file=False)
    low.uncertainty = 0.1
    high = make_image(db_session, project, "high.jpg", write_file=False)
    high.uncertainty = 0.8
    never_prelabelled = make_image(db_session, project, "never.jpg", write_file=False)
    db_session.commit()

    resp = client.get(f"/api/projects/{project.id}/next", params={"order": "uncertain"})
    assert resp.status_code == 200
    assert resp.json()["image"]["id"] == high.id

    resp2 = client.get(f"/api/projects/{project.id}/next",
                       params={"order": "uncertain", "after": high.id})
    assert resp2.json()["image"]["id"] == low.id

    resp3 = client.get(f"/api/projects/{project.id}/next",
                       params={"order": "uncertain", "after": low.id})
    assert resp3.json()["image"]["id"] == never_prelabelled.id


def test_next_image_empty_order_puts_zero_detection_images_first(client, db_session):
    project = make_project(db_session)
    some_detections = make_image(db_session, project, "some.jpg", write_file=False)
    some_detections.uncertainty = 0.3
    zero_detections = make_image(db_session, project, "zero.jpg", write_file=False)
    zero_detections.uncertainty = 0.9
    db_session.commit()

    resp = client.get(f"/api/projects/{project.id}/next", params={"order": "empty"})
    assert resp.json()["image"]["id"] == zero_detections.id


def test_next_image_dense_order_puts_most_boxes_first(client, db_session):
    project = make_project(db_session)
    label = make_label(db_session, project, "forklift", 0)
    few = make_image(db_session, project, "few.jpg", status="annotated", write_file=False)
    many = make_image(db_session, project, "many.jpg", status="annotated", write_file=False)
    db_session.add(Annotation(image_id=few.id, label_id=label.id, x1=0, y1=0, x2=0.1, y2=0.1))
    db_session.add_all([
        Annotation(image_id=many.id, label_id=label.id, x1=0, y1=0, x2=0.1, y2=0.1)
        for _ in range(3)
    ])
    db_session.commit()

    resp = client.get(f"/api/projects/{project.id}/next",
                      params={"order": "dense", "status": "annotated"})
    assert resp.json()["image"]["id"] == many.id


def test_next_image_rejects_an_unknown_order(client, db_session):
    project = make_project(db_session)
    make_image(db_session, project, "a.jpg", write_file=False)
    resp = client.get(f"/api/projects/{project.id}/next", params={"order": "bogus"})
    assert resp.status_code == 400


def test_prelabel_batch_grid_lists_only_still_pending_images(client, db_session):
    project = make_project(db_session)
    label = make_label(db_session, project, "forklift", 0)
    still_pending = make_image(db_session, project, "a.jpg", write_file=False)
    already_handled = make_image(db_session, project, "b.jpg", status="annotated", write_file=False)
    db_session.add(Annotation(image_id=still_pending.id, label_id=label.id, x1=0, y1=0, x2=0.1,
                              y2=0.1, source="model"))
    db_session.commit()

    job = Job(type="prelabel", project_id=project.id, status="done", params_json="{}",
             result_json=json.dumps({"image_ids": [still_pending.id, already_handled.id],
                                     "model_id": 1, "conf": 0.4}))
    db_session.add(job)
    db_session.commit()
    db_session.refresh(job)

    resp = client.get(f"/api/projects/{project.id}/prelabel-batch/{job.id}")
    assert resp.status_code == 200
    body = resp.json()
    assert [i["image"]["id"] for i in body["images"]] == [still_pending.id]
    assert len(body["images"][0]["annotations"]) == 1
