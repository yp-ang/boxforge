"""Standalone job entry point: `python -m app.worker <job_id>`.

Opens its own DB session, reads the job row, dispatches on job.type, writes result_json,
sets a terminal status and exits. Everything it prints lands in the job's log file, which
is what the SSE stream tails. It never imports the FastAPI app.
"""
from __future__ import annotations

import json
import os
import sys
import traceback
from datetime import datetime
from pathlib import Path

from app.config import settings
from app.db import SessionLocal, init_db
from app.models import Dataset, Job, Model, Project
from app.services.trainer import TrainConfig, UltralyticsTrainer, write_model_folder


def log(message: str) -> None:
    print(f"[{datetime.now():%H:%M:%S}] {message}", flush=True)


def run_train(db, job: Job) -> dict:
    params = json.loads(job.params_json or "{}")
    project = db.get(Project, job.project_id)
    if project is None:
        raise RuntimeError(f"project {job.project_id} no longer exists")

    dataset = db.get(Dataset, params["dataset_id"])
    if dataset is None or dataset.project_id != project.id:
        raise RuntimeError(f"dataset {params.get('dataset_id')} not found in this project")

    dataset_dir = Path(dataset.dir_path)
    dataset_yaml = dataset_dir / "data.yaml"
    if not dataset_yaml.is_file():
        raise RuntimeError(f"dataset yaml missing: {dataset_yaml}")

    cfg = TrainConfig(**params["config"])
    run_dir = settings.runs_dir / str(job.id) / "train"

    manifest = json.loads(dataset.manifest_json)
    names: dict[str, str] = {str(k): v for k, v in manifest["names"].items()}

    log(f"dataset {dataset.id}: {dataset.n_train} train / {dataset.n_val} val, "
        f"classes {list(names.values())}")
    log(f"run dir {run_dir}")

    result = UltralyticsTrainer().train(dataset_yaml, cfg, run_dir)

    # Training just held this session's read transaction open for the length of the run.
    # Drop it before writing anything: it pins a WAL snapshot, and every row read through
    # it from here on would be the hour-old copy rather than what the web process wrote.
    db.rollback()

    if not result.weights.is_file():
        raise RuntimeError(f"training finished but produced no weights at {result.weights}")

    model = Model(
        project_id=project.id,
        job_id=job.id,
        dataset_id=dataset.id,
        name=f"{Path(cfg.model).stem}-job{job.id}",
        dir_path="",                       # needs the row id; filled in below
        run_dir=str(result.run_dir),
        classes_json=json.dumps(names),
        metrics_json=json.dumps(result.metrics),
    )
    db.add(model)
    db.commit()
    db.refresh(model)

    model_dir = settings.models_dir / str(model.id)
    write_model_folder(
        model_dir,
        result.weights,
        names,
        {
            "model_id": model.id,
            "job_id": job.id,
            "project": project.slug,
            "dataset_id": dataset.id,
            "imgsz": cfg.imgsz,
            "task": project.task,
            "trained_at": datetime.utcnow().isoformat(),
            "config": cfg.to_dict(),
            "metrics": result.metrics,
        },
    )
    model.dir_path = str(model_dir)
    db.commit()

    log(f"model {model.id} written to {model_dir}")
    for key in ("mAP50-95", "mAP50", "precision", "recall"):
        if key in result.metrics:
            log(f"  {key:>9} = {result.metrics[key]:.4f}")

    return {"model_id": model.id, "run_dir": str(result.run_dir),
            "weights": str(model_dir / "best.pt"), "metrics": result.metrics}


HANDLERS = {"train": run_train}


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print("usage: python -m app.worker <job_id>", file=sys.stderr)
        return 2

    job_id = int(argv[1])
    init_db()

    # Ultralytics writes stray files (downloaded checkpoints, its settings file) relative
    # to the process CWD. Everything this worker uses is an absolute path under data/, so
    # parking the CWD there keeps those strays inside the one directory Docker mounts
    # (README ground rule 4).
    settings.pretrained_dir.mkdir(parents=True, exist_ok=True)
    os.chdir(settings.pretrained_dir)

    with SessionLocal() as db:
        job = db.get(Job, job_id)
        if job is None:
            print(f"job {job_id} not found", file=sys.stderr)
            return 2

        handler = HANDLERS.get(job.type)
        if handler is None:
            job.status, job.ended_at = "failed", datetime.utcnow()
            job.result_json = json.dumps({"error": f"unknown job type {job.type!r}"})
            db.commit()
            return 2

        log(f"job {job.id} ({job.type}) starting, pid {os.getpid()}")
        try:
            result = handler(db, job)
        except BaseException as exc:                    # SystemExit/KeyboardInterrupt too
            traceback.print_exc()
            sys.stderr.flush()
            db.rollback()
            db.expire_all()
            job = db.get(Job, job_id)
            # A cancel writes 'cancelled' from the web process and then signals us; don't
            # overwrite that verdict with 'failed' on the way out.
            if job is not None and job.status not in ("cancelled", "done"):
                job.status, job.ended_at = "failed", datetime.utcnow()
                job.result_json = json.dumps({"error": f"{type(exc).__name__}: {exc}"})
                db.commit()
            log("job failed")
            return 1

        # Re-read across a transaction boundary. Session.get() would otherwise hand back
        # the identity-mapped row loaded before training started, and a cancel that landed
        # while we were finishing would be silently overwritten with 'done'.
        db.rollback()
        db.expire_all()
        job = db.get(Job, job_id)
        if job is not None and job.status == "cancelled":
            log("job was cancelled after the work finished; leaving status as cancelled")
            return 0
        job.status, job.ended_at = "done", datetime.utcnow()
        job.result_json = json.dumps(result)
        db.commit()
        log(f"job {job.id} done")
        return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
