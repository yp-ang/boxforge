"""Subprocess job runner. No broker, no thread pool — see ARCHITECTURE §2.

Training runs in a subprocess rather than a thread because it holds the GIL, allocates
gigabytes, and occasionally dies in native code. A subprocess can be killed cleanly and
cannot take the UI down with it.
"""
from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.config import settings
from app.models import Job

# The entry point start_job spawns. A module constant rather than a literal so the job
# runner's process handling — group kills, orphan cleanup, reconciliation — can be tested
# against a cheap stand-in instead of a real GPU run.
WORKER_MODULE = "app.worker"

TERMINAL_STATUSES = ("done", "failed", "cancelled")
ACTIVE_STATUSES = ("queued", "running")
SIGKILL_GRACE_SECONDS = 5.0


class JobConflict(Exception):
    """A second training run was requested while one is already going."""


# Workers this process started, by job id. Nothing here is authoritative — the web server
# restarts and the DB outlives it — but while an entry exists it is the only way to reap
# the child. See _reap_tracked().
_PROCS: dict[int, subprocess.Popen] = {}


def _reap_tracked() -> None:
    """An exited child stays a zombie until someone wait()s for it, and a zombie still
    answers kill(pid, 0). Without this, every finished job would look like it was still
    running, and the one-job-at-a-time check would refuse every run after the first."""
    for job_id, proc in list(_PROCS.items()):
        if proc.poll() is not None:
            _PROCS.pop(job_id, None)


def running_job(db: Session, project_id: int | None = None) -> Job | None:
    stmt = select(Job).where(Job.status.in_(ACTIVE_STATUSES)).order_by(Job.id)
    if project_id is not None:
        stmt = stmt.where(Job.project_id == project_id)
    return db.scalars(stmt).first()


def _pid_alive(pid: int | None) -> bool:
    _reap_tracked()
    if not pid:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:            # alive, owned by someone else
        return True
    return True


def reconcile_jobs(db: Session) -> int:
    """Mark jobs whose process is gone as failed.

    The worker sets its own terminal status before exiting, so anything still 'running'
    with a dead pid died without getting the chance — killed externally, OOM, segfault in
    native code. Called on startup and whenever job state is read, which is often enough
    that a crashed run never sits 'running' in the UI for more than one poll.
    """
    fixed = 0
    # Only rows that got as far as recording a pid. A job is committed as 'queued' a
    # moment before start_job knows the pid, and reconciling on pid IS NULL would kill
    # every new job whose window happened to overlap someone loading the jobs list.
    candidates = select(Job).where(Job.status.in_(ACTIVE_STATUSES), Job.pid.is_not(None))
    for job in db.scalars(candidates).all():
        if _pid_alive(job.pid):
            continue
        job.status = "failed"
        job.ended_at = job.ended_at or datetime.utcnow()
        job.result_json = job.result_json or json.dumps(
            {"error": "process exited without reporting a status (killed, OOM, or crashed)"}
        )
        fixed += 1
    if fixed:
        db.commit()
    return fixed


def start_job(db: Session, job_type: str, project_id: int, params: dict) -> Job:
    """One job at a time, machine-wide. Two runs on one GPU is slower than two runs in
    sequence, and the failure mode is an OOM you then have to diagnose (§1)."""
    reconcile_jobs(db)
    existing = running_job(db)
    if existing is not None:
        raise JobConflict(
            f"job {existing.id} ({existing.type}) is already {existing.status} — "
            "cancel it or wait for it to finish"
        )

    job = Job(type=job_type, project_id=project_id, status="queued",
              params_json=json.dumps(params))
    db.add(job)
    db.commit()
    db.refresh(job)

    log_path = settings.runs_dir / str(job.id) / "job.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    job.log_path = str(log_path)

    env = {
        **os.environ,
        "PYTHONUNBUFFERED": "1",
        "MPLBACKEND": "Agg",                  # ultralytics plots, no display
        # Deliberately NOT setting KMP_DUPLICATE_LIB_OK. If two OpenMP runtimes ever get
        # loaded again, we want the loud "OMP: Error #15" rather than the silent SIGSEGV
        # that suppressing it produces mid-training. See environment.yml.
        # Absolute, so the worker's chdir below cannot change what these resolve to.
        "MID_DATA_DIR": str(settings.data_dir.resolve()),
        "MID_DB_URL": settings.resolved_db_url(),
        "YOLO_CONFIG_DIR": str((settings.data_dir / ".ultralytics").resolve()),
    }

    log_file = open(log_path, "w", buffering=1)   # line buffered; the SSE tail wants lines
    try:
        proc = subprocess.Popen(
            [sys.executable, "-m", WORKER_MODULE, str(job.id)],
            stdout=log_file,
            stderr=subprocess.STDOUT,
            env=env,
            cwd=str(Path(__file__).resolve().parents[2]),
            # Own process group, so cancelling kills the dataloader workers ultralytics
            # forks as well. Without this they survive as orphans holding the GPU.
            start_new_session=True,
        )
    except Exception:
        job.status = "failed"
        job.ended_at = datetime.utcnow()
        db.commit()
        raise
    finally:
        log_file.close()                          # the child holds its own dup of the fd

    _PROCS[job.id] = proc
    job.status = "running"
    job.pid = proc.pid
    job.started_at = datetime.utcnow()
    db.commit()
    db.refresh(job)
    return job


def cancel_job(db: Session, job: Job) -> Job:
    """Status first, then the signal: reconcile_jobs only ever touches 'running' rows, so
    writing 'cancelled' before the process dies is what keeps a cancel from being
    reported as a crash."""
    if job.status in TERMINAL_STATUSES:
        return job

    job.status = "cancelled"
    job.ended_at = datetime.utcnow()
    job.result_json = json.dumps({"error": "cancelled by user"})
    db.commit()

    pid = job.pid
    if pid and _pid_alive(pid):
        try:
            pgid = os.getpgid(pid)
        except ProcessLookupError:
            pgid = None
        if pgid is not None:
            os.killpg(pgid, signal.SIGTERM)
            deadline = time.monotonic() + SIGKILL_GRACE_SECONDS
            while _pid_alive(pid) and time.monotonic() < deadline:
                time.sleep(0.1)
            if _pid_alive(pid):
                os.killpg(pgid, signal.SIGKILL)

    proc = _PROCS.pop(job.id, None)
    if proc is not None:
        try:
            proc.wait(timeout=SIGKILL_GRACE_SECONDS)   # reap it rather than leave a zombie
        except subprocess.TimeoutExpired:
            pass

    db.refresh(job)
    return job


def job_status(job_id: int) -> str | None:
    """Read status on a fresh session. The SSE stream polls this while holding a session
    that has already loaded the row, and that session would happily serve the stale copy."""
    from app.db import SessionLocal

    with SessionLocal() as session:
        job = session.get(Job, job_id)
        return job.status if job else None
