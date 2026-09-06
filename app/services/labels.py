from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.models import Label


def next_class_index(db: Session, project_id: int) -> int:
    """Monotonic. Retired indices are never reused — a model's classes.json must keep
    meaning what it meant when it was trained."""
    hi = db.scalar(select(func.max(Label.class_index)).where(Label.project_id == project_id))
    return 0 if hi is None else hi + 1
