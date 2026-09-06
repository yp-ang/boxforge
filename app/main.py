from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from sqlalchemy.orm import Session

from app.config import settings
from app.db import get_db, init_db
from app.models import Project
from app.routers import annotations, datasets, images, labels, projects


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    yield


app = FastAPI(title="Machine ID", lifespan=lifespan)
app.mount("/static", StaticFiles(directory="app/static"), name="static")
templates = Jinja2Templates(directory="app/templates")

app.include_router(projects.router)
app.include_router(labels.router)
app.include_router(images.router)
app.include_router(annotations.router)
app.include_router(datasets.router)


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
