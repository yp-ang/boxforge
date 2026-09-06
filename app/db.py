from sqlalchemy import create_engine, event
from sqlalchemy.orm import DeclarativeBase, sessionmaker

from app.config import settings


class Base(DeclarativeBase): pass


engine = create_engine(
    settings.resolved_db_url(),
    connect_args={"check_same_thread": False},   # SQLite + FastAPI threadpool
)
SessionLocal = sessionmaker(bind=engine, expire_on_commit=False)


@event.listens_for(engine, "connect")
def _set_sqlite_pragma(conn, _):
    cur = conn.cursor()
    cur.execute("PRAGMA journal_mode=WAL")
    cur.execute("PRAGMA foreign_keys=ON")     # off by default in SQLite. Turn it on.
    cur.close()


def init_db() -> None:
    from app import models  # noqa: F401 — register tables
    settings.ensure_dirs()
    Base.metadata.create_all(engine)


def get_db():
    db = SessionLocal()
    try: yield db
    finally: db.close()
