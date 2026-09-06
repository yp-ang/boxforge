from datetime import datetime

from pydantic import BaseModel, ConfigDict


class ProjectCreate(BaseModel):
    name: str


class ProjectOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    name: str
    slug: str
    task: str
    created_at: datetime


class LabelCreate(BaseModel):
    name: str
    color: str | None = None


class LabelOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    project_id: int
    name: str
    color: str
    class_index: int


class ImageOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    project_id: int
    rel_path: str
    sha256: str
    width: int
    height: int
    status: str
    ingested_at: datetime
    reviewed_at: datetime | None = None


class IngestRequest(BaseModel):
    source_dir: str
    copy: bool = True


class IngestReport(BaseModel):
    added: int = 0
    skipped: int = 0
    failed: int = 0
    note: str | None = None


class StatsOut(BaseModel):
    total: int
    pending: int
    annotated: int
    skipped: int
    reviewed: int
    boxes_per_label: dict[str, int]


class BoxIn(BaseModel):
    label_id: int
    x1: float
    y1: float
    x2: float
    y2: float


class AnnotationsIn(BaseModel):
    boxes: list[BoxIn]


class AnnotationOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    label_id: int
    x1: float
    y1: float
    x2: float
    y2: float
    source: str


class ImageDetailOut(BaseModel):
    image: ImageOut
    annotations: list[AnnotationOut]
    index: int
    total: int
