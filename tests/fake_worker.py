"""A stand-in for app.worker, used to test the job runner's process handling.

`python -m tests.fake_worker <job_id>` reads its job's params_json and honours:
  children: how many grandchild processes to spawn (they inherit our process group, so
            they are exactly the orphans a naive kill would leave holding the GPU)
  sleep:    how long to stay alive
  fail:     raise instead of finishing
It writes the same terminal statuses the real worker does.
"""
import json
import subprocess
import sys
import time
from datetime import datetime

from app.db import SessionLocal
from app.models import Job


def main(job_id: int) -> int:
    with SessionLocal() as db:
        job = db.get(Job, job_id)
        params = json.loads(job.params_json or "{}")

        kids = [
            subprocess.Popen([sys.executable, "-c", "import time; time.sleep(300)"])
            for _ in range(params.get("children", 0))
        ]
        print(f"started {len(kids)} children: {[k.pid for k in kids]}", flush=True)
        for i in range(params.get("lines", 3)):
            print(f"line {i}", flush=True)

        if params.get("fail"):
            raise RuntimeError("fake worker asked to fail")

        time.sleep(params.get("sleep", 0))

        job = db.get(Job, job_id)
        job.status, job.ended_at = "done", datetime.utcnow()
        job.result_json = json.dumps({"ok": True})
        db.commit()
        print("done", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(int(sys.argv[1])))
