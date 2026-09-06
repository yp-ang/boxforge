from sqlalchemy import create_engine, event, inspect, text
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


def _add_missing_columns() -> None:
    """create_all() creates tables but never alters them. Every column this project has
    added since a table first shipped is nullable or defaulted, so a plain ADD COLUMN is
    the whole migration story — no Alembic until something needs a backfill or a drop."""
    inspector = inspect(engine)
    with engine.begin() as conn:
        for table in Base.metadata.sorted_tables:
            if not inspector.has_table(table.name):
                continue
            existing = {c["name"] for c in inspector.get_columns(table.name)}
            for column in table.columns:
                if column.name in existing:
                    continue
                ddl = f"{column.name} {column.type.compile(engine.dialect)}"
                default = column.default.arg if column.default is not None else None
                if isinstance(default, (int, float, str, bool)):
                    ddl += f" DEFAULT {int(default) if isinstance(default, bool) else default!r}"
                conn.execute(text(f"ALTER TABLE {table.name} ADD COLUMN {ddl}"))


def init_db() -> None:
    from app import models  # noqa: F401 — register tables
    settings.ensure_dirs()
    Base.metadata.create_all(engine)
    _add_missing_columns()


def get_db():
    db = SessionLocal()
    try: yield db
    finally: db.close()
