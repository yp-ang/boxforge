from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from app.config import settings
from app.db import init_db
from app.routers import images, labels, projects


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


@app.get("/health")
def health(): return {"status": "ok", "data_dir": str(settings.data_dir)}


@app.get("/")
def index(request: Request):
    return templates.TemplateResponse(request, "index.html")
