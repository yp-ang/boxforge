import json
import os
import signal
import time

import pytest

from app.config import settings
from app.models import Dataset, Job, Model, Project
from app.services import jobs as jobs_module
from app.services.jobs import (
    JobConflict,
    cancel_job,
    reconcile_jobs,
    running_job,
    start_job,
)
from app.services.trainer import (
    TrainConfig,
    extract_metrics,
    metrics_from_results_csv,
    resolve_weights,
)


@pytest.fixture
def fake_worker(monkeypatch):
    monkeypatch.setattr(jobs_module, "WORKER_MODULE", "tests.fake_worker")


def make_project(db, name="warehouse"):
    project = Project(name=name, slug=name)
    db.add(project)
    db.commit()
    db.refresh(project)
    return project


def make_dataset(db, project, with_yaml=True):
    dataset_dir = settings.datasets_dir / project.slug / "20260101-000000"
    dataset_dir.mkdir(parents=True, exist_ok=True)
    if with_yaml:
        (dataset_dir / "data.yaml").write_text("names:\n  0: forklift\n")
    dataset = Dataset(project_id=project.id, dir_path=str(dataset_dir), val_pct=20,
                      n_train=8, n_val=2,
                      manifest_json=json.dumps({"names": {"0": "forklift"}}))
    db.add(dataset)
    db.commit()
    db.refresh(dataset)
    return dataset


def fresh(db, job_id) -> Job:
    """End the session's read transaction before re-reading. The worker writes from
    another process, and a SQLite snapshot taken earlier will not show its commits."""
    db.rollback()
    return db.get(Job, job_id)


def wait_until(predicate, timeout=20.0, interval=0.1):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(interval)
    return False


def pid_alive(pid: int) -> bool:
    """Delegates so the check reaps first. An exited child of this very process is a
    zombie until someone wait()s for it, and a zombie answers kill(pid, 0) quite happily —
    asking raw here would report every finished worker as still running."""
    return jobs_module._pid_alive(pid)


# --- pure helpers -----------------------------------------------------------

def test_resolve_weights_keeps_bare_names_under_data_dir():
    resolved = resolve_weights("yolo11n.pt")
    assert resolved.endswith("yolo11n.pt")
    assert str(settings.pretrained_dir.resolve()) in resolved
    # An explicit path is the caller's business and passes through untouched.
    assert resolve_weights("/tmp/custom.pt") == "/tmp/custom.pt"


RESULTS_DICT = {
    "metrics/mAP50(B)": 0.71, "metrics/mAP50-95(B)": 0.42,
    "metrics/precision(B)": 0.8, "metrics/recall(B)": 0.6,
    "fitness": "not-a-number",
}


def test_extract_metrics_handles_the_detmetrics_object():
    class Results:
        results_dict = RESULTS_DICT

    metrics = extract_metrics(Results())
    assert metrics["mAP50-95"] == pytest.approx(0.42)
    assert metrics["precision"] == pytest.approx(0.8)
    assert "fitness" not in metrics       # unparseable values are dropped, not crashed on


def test_extract_metrics_handles_the_bare_checkpoint_dict():
    """model.train() returns the checkpoint's plain 'train_metrics' dict when the
    validator is already gone — same keys, no .results_dict to reach through."""
    metrics = extract_metrics(RESULTS_DICT)
    assert metrics["mAP50-95"] == pytest.approx(0.42)
    assert metrics["recall"] == pytest.approx(0.6)


def test_extract_metrics_gives_up_quietly_on_something_unrecognised():
    assert extract_metrics(None) == {}
    assert extract_metrics(object()) == {}


def test_metrics_from_results_csv_reads_the_final_epoch(tmp_path):
    (tmp_path / "results.csv").write_text(
        "epoch,train/box_loss,metrics/precision(B),metrics/recall(B),"
        "metrics/mAP50(B),metrics/mAP50-95(B)\n"
        "1,1.19,0.015,1.0,0.028,0.016\n"
        "2,0.76,0.129,1.0,0.367,0.189\n"
    )
    metrics = metrics_from_results_csv(tmp_path)
    assert metrics == {"precision": pytest.approx(0.129), "recall": pytest.approx(1.0),
                       "mAP50": pytest.approx(0.367), "mAP50-95": pytest.approx(0.189)}
    assert "train/box_loss" not in metrics          # losses are not what the table shows
    assert metrics_from_results_csv(tmp_path / "nope") == {}


def test_train_config_round_trips_through_params():
    cfg = TrainConfig(epochs=5, imgsz=416, fliplr=0.0)
    assert TrainConfig(**cfg.to_dict()) == cfg


# --- job runner -------------------------------------------------------------

def test_one_job_at_a_time(db_session, fake_worker):
    project = make_project(db_session)
    first = start_job(db_session, "train", project.id, {"sleep": 30})

    with pytest.raises(JobConflict) as exc:
        start_job(db_session, "train", project.id, {"sleep": 30})
    assert str(first.id) in str(exc.value)

    cancel_job(db_session, first)
    # And once it is out of the way, a new job is allowed again.
    second = start_job(db_session, "train", project.id, {"sleep": 30})
    cancel_job(db_session, second)


def test_cancel_kills_the_whole_process_group(db_session, fake_worker):
    project = make_project(db_session)
    job = start_job(db_session, "train", project.id, {"sleep": 60, "children": 2})

    log_path = settings.runs_dir / str(job.id) / "job.log"
    assert wait_until(lambda: "children" in log_path.read_text())
    child_pids = json.loads(
        log_path.read_text().splitlines()[0].split(": ", 1)[1].replace("'", '"')
    )
    assert all(pid_alive(pid) for pid in child_pids)

    cancel_job(db_session, job)

    assert fresh(db_session, job.id).status == "cancelled"
    assert wait_until(lambda: not pid_alive(job.pid))
    # The grandchildren are the orphans a plain kill(worker_pid) would leave behind.
    for pid in child_pids:
        assert wait_until(lambda pid=pid: not pid_alive(pid)), f"orphan survived: {pid}"


def test_external_kill_reconciles_to_failed(db_session, fake_worker):
    project = make_project(db_session)
    job = start_job(db_session, "train", project.id, {"sleep": 60})
    assert wait_until(lambda: pid_alive(job.pid))

    os.killpg(os.getpgid(job.pid), signal.SIGKILL)     # the app never got a say
    assert wait_until(lambda: not pid_alive(job.pid))

    db_session.rollback()
    assert reconcile_jobs(db_session) == 1
    reconciled = fresh(db_session, job.id)
    assert reconciled.status == "failed"
    assert reconciled.ended_at is not None
    assert "without reporting a status" in json.loads(reconciled.result_json)["error"]

    # Reconciliation is idempotent, and it unblocks the next run.
    assert reconcile_jobs(db_session) == 0
    assert running_job(db_session) is None


def test_reconcile_leaves_a_queued_job_without_a_pid_alone(db_session):
    """The window between committing the row and recording the pid is real; a jobs-list
    request landing inside it must not shoot the job that is still being spawned."""
    project = make_project(db_session)
    job = Job(type="train", status="queued", project_id=project.id, params_json="{}")
    db_session.add(job)
    db_session.commit()

    assert reconcile_jobs(db_session) == 0
    assert fresh(db_session, job.id).status == "queued"


def test_crashed_worker_reconciles_to_failed(db_session, fake_worker):
    project = make_project(db_session)
    job = start_job(db_session, "train", project.id, {"fail": True})
    assert wait_until(lambda: not pid_alive(job.pid))

    db_session.rollback()
    reconcile_jobs(db_session)
    assert fresh(db_session, job.id).status == "failed"
    # The traceback lands in the log the UI is already tailing.
    assert "RuntimeError" in (settings.runs_dir / str(job.id) / "job.log").read_text()


def test_successful_job_reaches_done_and_logs_its_output(db_session, fake_worker):
    project = make_project(db_session)
    job = start_job(db_session, "train", project.id, {"sleep": 0, "lines": 4})

    assert wait_until(lambda: fresh(db_session, job.id).status == "done")
    log = (settings.runs_dir / str(job.id) / "job.log").read_text()
    assert "line 3" in log and "done" in log
    assert reconcile_jobs(db_session) == 0     # a finished job is never re-marked failed


def test_real_worker_rejects_an_unknown_job_type(db_session):
    project = make_project(db_session)
    job = start_job(db_session, "nonsense", project.id, {})

    assert wait_until(lambda: fresh(db_session, job.id).status == "failed")
    assert "unknown job type" in json.loads(fresh(db_session, job.id).result_json)["error"]


def test_clean_log_line_keeps_only_what_a_terminal_would_show():
    from app.routers.training import clean_log_line

    # A tqdm-style redraw: only the final segment was ever visible.
    assert clean_log_line("  0%  0/8\r 50%  4/8\r100%  8/8\n") == "100%  8/8"
    assert clean_log_line("\x1b[34m\x1b[1mengine/trainer:\x1b[0m epochs=5\n") == \
        "engine/trainer: epochs=5"
    assert clean_log_line("plain line\n") == "plain line"


# --- API --------------------------------------------------------------------

def test_train_endpoint_refuses_without_a_dataset(client, db_session):
    project = make_project(db_session)
    resp = client.post(f"/api/projects/{project.id}/train", json={})
    assert resp.status_code == 400
    assert "dataset" in resp.json()["detail"]


def test_train_endpoint_refuses_a_second_run(client, db_session, fake_worker):
    project = make_project(db_session)
    make_dataset(db_session, project)

    first = client.post(f"/api/projects/{project.id}/train",
                        json={"config": {"epochs": 5, "sleep": 30}})
    assert first.status_code == 200, first.text
    job_id = first.json()["id"]

    second = client.post(f"/api/projects/{project.id}/train", json={})
    assert second.status_code == 409
    assert str(job_id) in second.json()["detail"]

    assert client.post(f"/api/jobs/{job_id}/cancel").json()["status"] == "cancelled"


def test_train_endpoint_defaults_to_the_newest_dataset(client, db_session, fake_worker):
    project = make_project(db_session)
    make_dataset(db_session, project)
    newest = make_dataset(db_session, project)

    resp = client.post(f"/api/projects/{project.id}/train", json={"config": {"epochs": 5}})
    assert resp.status_code == 200
    params = json.loads(resp.json()["params_json"])
    assert params["dataset_id"] == newest.id
    assert params["config"]["epochs"] == 5

    client.post(f"/api/jobs/{resp.json()['id']}/cancel")


def test_log_stream_replays_from_the_top_and_ends(client, db_session, fake_worker):
    project = make_project(db_session)
    job = start_job(db_session, "train", project.id, {"sleep": 0, "lines": 3})
    assert wait_until(lambda: fresh(db_session, job.id).status == "done")

    # Reattaching after the run finished must still show the whole log, not an empty pane.
    with client.stream("GET", f"/api/jobs/{job.id}/logs") as stream:
        body = "".join(stream.iter_text())
    assert "data: line 0" in body
    assert "data: line 2" in body
    assert "event: done" in body


# --- augmentation preview (§7) ----------------------------------------------

ultralytics = pytest.importorskip("ultralytics", reason="preview uses the real transforms")


def test_augment_preview_produces_imgsz_variants_that_differ():
    """Guards the two things that silently break this: Mosaic drawing tiles from an empty
    buffer (IndexError), and the mosaic canvas never being cropped back to imgsz."""
    import numpy as np

    from app.services.trainer import TrainConfig, augment_preview

    rng = np.random.default_rng(0)
    images = [rng.integers(0, 255, (192, 192, 3), dtype=np.uint8) for _ in range(6)]

    out = augment_preview(images, TrainConfig(imgsz=320, fliplr=0.5))
    assert len(out) == len(images)
    assert {im.shape for im in out} == {(320, 320, 3)}
    assert all(not np.array_equal(a, b) for a, b in zip(images, out))


def test_augment_preview_works_with_fewer_images_than_a_mosaic_needs():
    import numpy as np

    from app.services.trainer import TrainConfig, augment_preview

    rng = np.random.default_rng(1)
    images = [rng.integers(0, 255, (192, 192, 3), dtype=np.uint8) for _ in range(2)]
    assert len(augment_preview(images, TrainConfig(imgsz=160))) == 2


# --- model registry (§8) ----------------------------------------------------

def make_model(db, project, name, mAP=0.4, active=False):
    model = Model(project_id=project.id, name=name, dir_path=str(settings.models_dir / name),
                  classes_json=json.dumps({"0": "forklift"}),
                  metrics_json=json.dumps({"mAP50-95": mAP}), is_active=active)
    db.add(model)
    db.commit()
    db.refresh(model)
    return model


def test_activating_a_model_clears_the_others(client, db_session):
    project = make_project(db_session)
    other = make_project(db_session, "other")
    models = [make_model(db_session, project, f"run{i}", mAP=0.1 * i) for i in range(3)]
    elsewhere = make_model(db_session, other, "unrelated", active=True)

    listed = client.get(f"/api/projects/{project.id}/models").json()
    assert len(listed) == 3

    assert client.post(f"/api/models/{models[1].id}/activate").json()["is_active"] is True
    assert client.post(f"/api/models/{models[2].id}/activate").json()["is_active"] is True

    db_session.rollback()
    flags = {m["name"]: m["is_active"] for m in
             client.get(f"/api/projects/{project.id}/models").json()}
    assert flags == {"run0": False, "run1": False, "run2": True}

    # Scoped per project — activating here must not disturb another project's choice.
    db_session.refresh(elsewhere)
    assert elsewhere.is_active is True


def test_model_plot_endpoint_rejects_path_escape(client, db_session):
    project = make_project(db_session)
    model = make_model(db_session, project, "run0")
    model.run_dir = str(settings.runs_dir / "1" / "train")
    db_session.commit()

    # Traversal never even reaches the handler — httpx normalises the path away — but the
    # allowlist is what actually stops a hand-crafted request.
    assert client.get(f"/api/models/{model.id}/plots/../../app.db").status_code != 200
    assert client.get(f"/api/models/{model.id}/plots/secrets.txt").status_code == 400
    assert client.get(f"/api/models/{model.id}/plots/results.png").status_code == 404
