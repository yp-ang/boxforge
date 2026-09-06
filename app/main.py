from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from sqlalchemy.orm import Session

from app.config import settings
from app.db import SessionLocal, get_db, init_db
from app.models import Project
from app.routers import annotations, datasets, images, inference, labels, projects, training
from app.services.jobs import reconcile_jobs


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    # A job whose process died while the server was down is still 'running' in the DB.
    # Settle that before anyone can look at the UI, or the one-job-at-a-time check will
    # refuse every new run on behalf of a process that no longer exists.
    with SessionLocal() as db:
        reconcile_jobs(db)
    yield


app = FastAPI(title="Machine ID", lifespan=lifespan)
app.mount("/static", StaticFiles(directory="app/static"), name="static")
templates = Jinja2Templates(directory="app/templates")

app.include_router(projects.router)
app.include_router(labels.router)
app.include_router(images.router)
app.include_router(annotations.router)
app.include_router(datasets.router)
app.include_router(training.router)
app.include_router(inference.router)


@app.get("/health")
def health(): return {"status": "ok", "data_dir": str(settings.data_dir)}


@app.get("/")
def index(request: Request):
    return templates.TemplateResponse(request, "index.html")


@app.get("/annotate/{project_id}")
def annotate_page(request: Request, project_id: int, db: Session = Depends(get_db)):
    project = db.get(Project, project_id)
    if not project:
        raise HTTPException(404, "project not found")
    return templates.TemplateResponse(request, "annotate.html", {"project": project})


@app.get("/train/{project_id}")
def train_page(request: Request, project_id: int, db: Session = Depends(get_db)):
    project = db.get(Project, project_id)
    if not project:
        raise HTTPException(404, "project not found")
    return templates.TemplateResponse(request, "train.html", {"project": project})


@app.get("/verify/{project_id}")
def verify_page(request: Request, project_id: int, db: Session = Depends(get_db)):
    project = db.get(Project, project_id)
    if not project:
        raise HTTPException(404, "project not found")
    return templates.TemplateResponse(request, "verify.html", {"project": project})
