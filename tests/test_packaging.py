"""Step 09: paths stored in the DB must resolve correctly regardless of which
environment (conda on the Mac vs. the Docker container's /app/data) reads them back —
see docs/steps/09-docker.md §3 and app.config.Settings.data_path()/rel_data_path()."""
import json
from pathlib import Path

import pytest
import yaml

from app.config import Settings
from app.models import Dataset, Job, Model, Project
from app.services.dataset_export import materialise_data_yaml


def test_data_path_resolves_a_relative_value_against_data_dir(tmp_path):
    settings = Settings(data_dir=tmp_path)
    assert settings.data_path("models/3") == tmp_path / "models/3"


def test_data_path_honours_a_legacy_absolute_value(tmp_path):
    """Rows written before this existed stored a real absolute path — still correct
    within the environment that wrote it, so don't break it."""
    settings = Settings(data_dir=tmp_path)
    legacy = tmp_path / "models" / "3"
    assert settings.data_path(str(legacy)) == legacy


def test_rel_data_path_is_the_inverse_of_data_path(tmp_path):
    settings = Settings(data_dir=tmp_path)
    absolute = tmp_path / "models" / "7"
    absolute.mkdir(parents=True)
    rel = settings.rel_data_path(absolute)
    assert not Path(rel).is_absolute()
    assert settings.data_path(rel) == absolute


def test_a_dataset_and_model_exported_in_one_environment_resolve_in_another(tmp_path):
    """The core step 09 §3 scenario: data/ physically moves from one absolute path
    (simulating conda on the Mac) to another (simulating the Docker bind mount at
    /app/data). Anything stored via rel_data_path() must still resolve; anything the
    contents of data/ actually need (e.g. data.yaml's path: key) must be regenerated,
    not copied verbatim."""
    env_a = tmp_path / "env-a" / "data"
    env_b = tmp_path / "env-b" / "data"

    settings_a = Settings(data_dir=env_a)
    dataset_dir = settings_a.datasets_dir / "proj" / "20260101"
    (dataset_dir / "images" / "train").mkdir(parents=True)
    (dataset_dir / "images" / "val").mkdir(parents=True)
    (dataset_dir / "manifest.json").write_text(json.dumps({"names": {"0": "forklift"}}))
    yaml_path = materialise_data_yaml(dataset_dir)
    assert yaml.safe_load(yaml_path.read_text())["path"] == str(dataset_dir)

    rel_dataset_path = settings_a.rel_data_path(dataset_dir)

    model_dir = settings_a.models_dir / "9"
    model_dir.mkdir(parents=True)
    (model_dir / "model.onnx").write_bytes(b"fake-onnx")
    rel_model_path = settings_a.rel_data_path(model_dir)
    rel_onnx_path = settings_a.rel_data_path(model_dir / "model.onnx")

    # Move the whole data/ directory — this is the bind mount swap.
    env_b.parent.mkdir(parents=True)
    env_a.rename(env_b)

    settings_b = Settings(data_dir=env_b)
    resolved_dataset_dir = settings_b.data_path(rel_dataset_path)
    resolved_model_dir = settings_b.data_path(rel_model_path)
    resolved_onnx_path = settings_b.data_path(rel_onnx_path)

    assert resolved_dataset_dir.is_dir()
    assert resolved_model_dir.is_dir()
    assert resolved_onnx_path.is_file()

    # The stale data.yaml still has env_a's absolute path baked in — exactly why
    # run_train() must call materialise_data_yaml() again before training, not trust
    # the file dataset export left behind.
    stale = yaml.safe_load((resolved_dataset_dir / "data.yaml").read_text())
    assert stale["path"] != str(resolved_dataset_dir)

    fresh_yaml = materialise_data_yaml(resolved_dataset_dir)
    fresh = yaml.safe_load(fresh_yaml.read_text())
    assert fresh["path"] == str(resolved_dataset_dir)
    assert fresh["names"] == {"0": "forklift"}


# --- worker.run_train actually calls materialise_data_yaml ------------------------

def test_run_train_regenerates_data_yaml_for_the_current_environment(db_session, monkeypatch):
    """End-to-end through worker.run_train with a fake trainer: the dataset directory
    is seeded with a data.yaml pointing at some other environment's absolute path (as
    if exported elsewhere), and training must not choke on it or silently train against
    the wrong images."""
    from app import worker
    from app.config import settings
    from app.services.trainer import TrainResult

    project = Project(name="p", slug="p")
    db_session.add(project)
    db_session.commit()
    db_session.refresh(project)

    dataset_dir = settings.datasets_dir / "p" / "20260101"
    (dataset_dir / "images" / "train").mkdir(parents=True)
    (dataset_dir / "images" / "val").mkdir(parents=True)
    (dataset_dir / "manifest.json").write_text(json.dumps({"names": {"0": "forklift"}}))
    stale_yaml = {"path": "/somewhere/that/does/not/exist", "train": "images/train",
                 "val": "images/val", "names": {"0": "forklift"}}
    (dataset_dir / "data.yaml").write_text(yaml.safe_dump(stale_yaml))

    dataset = Dataset(project_id=project.id, dir_path=settings.rel_data_path(dataset_dir),
                      val_pct=20, n_train=1, n_val=1,
                      manifest_json=json.dumps({"names": {"0": "forklift"}}))
    db_session.add(dataset)
    db_session.commit()
    db_session.refresh(dataset)

    job = Job(type="train", project_id=project.id, status="running",
             params_json=json.dumps({"dataset_id": dataset.id, "config": {}}))
    db_session.add(job)
    db_session.commit()
    db_session.refresh(job)

    seen_yaml_path = {}

    class _FakeTrainer:
        def train(self, dataset_yaml, cfg, run_dir):
            seen_yaml_path["path"] = yaml.safe_load(dataset_yaml.read_text())["path"]
            weights = run_dir / "weights" / "best.pt"
            weights.parent.mkdir(parents=True)
            weights.write_bytes(b"fake-weights")
            return TrainResult(weights=weights, run_dir=run_dir, metrics={"mAP50-95": 0.5})

    monkeypatch.setattr(worker, "UltralyticsTrainer", _FakeTrainer)

    result = worker.run_train(db_session, job)

    assert seen_yaml_path["path"] == str(dataset_dir.resolve())
    model = db_session.get(Model, result["model_id"])
    # Stored relative — resolves back to the real folder in *this* process.
    assert settings.data_path(model.dir_path).is_dir()
    assert (settings.data_path(model.dir_path) / "best.pt").is_file()
