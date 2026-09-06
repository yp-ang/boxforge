from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field


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


class DatasetExportRequest(BaseModel):
    val_pct: int = 20
    force: bool = False


class PreflightReportOut(BaseModel):
    errors: list[str]
    warnings: list[str]
    per_class: dict[str, int]
    n_train: int
    n_val: int


class DatasetOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    project_id: int
    dir_path: str
    val_pct: int
    n_train: int
    n_val: int
    created_at: datetime


class DatasetExportOut(BaseModel):
    dataset_id: int | None
    path: str | None
    report: PreflightReportOut


class TrainConfigIn(BaseModel):
    model_config = ConfigDict(protected_namespaces=())    # we really do want a 'model' field

    model: str = "yolo11n.pt"
    epochs: int = Field(100, ge=1, le=2000)
    imgsz: int = Field(640, ge=64, le=2560)
    batch: int = Field(-1, ge=-1, le=256)
    patience: int = Field(30, ge=0, le=2000)
    device: str = "auto"
    seed: int = 0
    fliplr: float = Field(0.5, ge=0.0, le=1.0)


class TrainRequest(BaseModel):
    dataset_id: int | None = None      # defaults to the project's newest dataset
    config: TrainConfigIn = Field(default_factory=TrainConfigIn)


class JobOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    type: str
    status: str
    project_id: int
    params_json: str | None = None
    result_json: str | None = None
    pid: int | None = None
    started_at: datetime | None = None
    ended_at: datetime | None = None


class ModelOut(BaseModel):
    model_config = ConfigDict(from_attributes=True, protected_namespaces=())

    id: int
    project_id: int
    job_id: int | None = None
    dataset_id: int | None = None
    name: str
    dir_path: str
    run_dir: str | None = None
    classes_json: str
    metrics_json: str | None = None
    is_active: bool
    created_at: datetime


class AugmentPreviewItem(BaseModel):
    image_id: int
    original: str        # data: URI
    augmented: str       # data: URI


class AugmentPreviewOut(BaseModel):
    items: list[AugmentPreviewItem]
    note: str | None = None
