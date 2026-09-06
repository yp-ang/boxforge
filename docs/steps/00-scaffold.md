# Step 00 — Scaffold

**Goal:** a FastAPI app that boots in conda, creates its SQLite database, and serves a
page. Nothing domain-specific yet. This step exists so that every later step has
somewhere to land.

**Prereqs:** conda installed. Nothing else.

## 1. Environment

`environment.yml`:

```yaml
name: machine-id
channels: [conda-forge]
dependencies:
  - python=3.11
  - pip
  - numpy
  - pillow
  - opencv                 # conda-forge build; avoids the opencv-python wheel mess
  - pip:
      - fastapi
      - "uvicorn[standard]"
      - sqlalchemy>=2.0
      - pydantic-settings
      - jinja2
      - python-multipart      # file uploads
      - onnxruntime           # inference; CPU + CoreML EP on macOS
      - ultralytics           # training only — see ARCHITECTURE §5.1
      - pytest
      - httpx                 # test client
```

```bash
conda env create -f environment.yml
conda activate machine-id
```

Pin `python=3.11` deliberately — see ARCHITECTURE §2.

## 2. Config

`app/config.py` — one settings object, env-overridable, so conda and Docker differ only
by environment variables.

```python
from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_prefix="MID_")

    data_dir: Path = Path("data")
    db_url: str | None = None          # derived if unset
    default_imgsz: int = 640
    default_conf: float = 0.25
    device: str = "auto"               # auto | cpu | mps | cuda:0

    @property
    def images_dir(self) -> Path:  return self.data_dir / "images"
    @property
    def datasets_dir(self) -> Path: return self.data_dir / "datasets"
    @property
    def runs_dir(self) -> Path:    return self.data_dir / "runs"
    @property
    def models_dir(self) -> Path:  return self.data_dir / "models"

    def resolved_db_url(self) -> str:
        return self.db_url or f"sqlite:///{(self.data_dir / 'app.db').resolve()}"

    def ensure_dirs(self) -> None:
        for d in (self.images_dir, self.datasets_dir, self.runs_dir, self.models_dir):
            d.mkdir(parents=True, exist_ok=True)

settings = Settings()
```

`.env.example`:

```
MID_DATA_DIR=data
MID_DEVICE=auto
MID_DEFAULT_CONF=0.25
```

## 3. Database

`app/db.py`:

```python
from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, sessionmaker
from app.config import settings

class Base(DeclarativeBase): pass

engine = create_engine(
    settings.resolved_db_url(),
    connect_args={"check_same_thread": False},   # SQLite + FastAPI threadpool
)
SessionLocal = sessionmaker(bind=engine, expire_on_commit=False)

def init_db() -> None:
    from app import models  # noqa: F401 — register tables
    settings.ensure_dirs()
    Base.metadata.create_all(engine)

def get_db():
    db = SessionLocal()
    try: yield db
    finally: db.close()
```

Enable WAL mode on connect — it is the difference between "works" and "database is
locked" once the training job and the UI touch the DB at the same time:

```python
from sqlalchemy import event
@event.listens_for(engine, "connect")
def _set_sqlite_pragma(conn, _):
    cur = conn.cursor()
    cur.execute("PRAGMA journal_mode=WAL")
    cur.execute("PRAGMA foreign_keys=ON")     # off by default in SQLite. Turn it on.
    cur.close()
```

## 4. App entry

`app/main.py`:

```python
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from app.db import init_db
from app.config import settings

@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    yield

app = FastAPI(title="Machine ID", lifespan=lifespan)
app.mount("/static", StaticFiles(directory="app/static"), name="static")
templates = Jinja2Templates(directory="app/templates")

@app.get("/health")
def health(): return {"status": "ok", "data_dir": str(settings.data_dir)}
```

Serve annotated images later via a dedicated route, not `StaticFiles` on `data/` —
you want to control which project's files are reachable.

## 5. Migrations

Skip Alembic for now. `create_all` handles additive schema during early development; when
you need to change a column on data you care about, add Alembic then. Note the decision in
the repo so future-you doesn't wonder.

## Acceptance criteria

```bash
uvicorn app.main:app --reload --port 8000
curl -s localhost:8000/health          # {"status":"ok",...}
ls data/app.db                         # exists
open http://localhost:8000/docs        # Swagger UI renders
```

`tests/test_health.py` passes under `pytest`.
