from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.db import get_db
from app.models import Dataset, Project
from app.schemas import DatasetExportOut, DatasetExportRequest, DatasetOut, PreflightReportOut
from app.services.dataset_export import export_coco, export_dataset

router = APIRouter(prefix="/api/projects/{project_id}/datasets", tags=["datasets"])


@router.post("", response_model=DatasetExportOut)
def create_dataset(project_id: int, payload: DatasetExportRequest, db: Session = Depends(get_db)):
    project = db.get(Project, project_id)
    if not project:
        raise HTTPException(404, "project not found")
    result = export_dataset(db, project, val_pct=payload.val_pct, force=payload.force)
    return DatasetExportOut(
        dataset_id=result.dataset.id if result.dataset else None,
        path=result.path,
        report=PreflightReportOut(
            errors=result.report.errors,
            warnings=result.report.warnings,
            per_class=result.report.per_class,
            n_train=result.report.n_train,
            n_val=result.report.n_val,
        ),
    )


@router.get("", response_model=list[DatasetOut])
def list_datasets(project_id: int, db: Session = Depends(get_db)):
    return db.scalars(
        select(Dataset).where(Dataset.project_id == project_id).order_by(Dataset.id.desc())
    ).all()


@router.get("/{dataset_id}/coco.json")
def dataset_coco(project_id: int, dataset_id: int, db: Session = Depends(get_db)):
    project = db.get(Project, project_id)
    if not project:
        raise HTTPException(404, "project not found")
    dataset = db.get(Dataset, dataset_id)
    if not dataset or dataset.project_id != project_id:
        raise HTTPException(404, "dataset not found")
    return export_coco(db, project, dataset)
