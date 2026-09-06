import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app import config
from app import db as db_module
from app.db import Base


@pytest.fixture
def db_session(tmp_path, monkeypatch):
    """Isolated DB + data dir per test — export writes real files and DB rows, and must
    not touch the real dev data/app.db."""
    monkeypatch.setattr(config.settings, "data_dir", tmp_path)
    # Set explicitly rather than letting resolved_db_url() derive it: start_job hands the
    # worker subprocess MID_DB_URL, and the child must land on this same file.
    monkeypatch.setattr(config.settings, "db_url", f"sqlite:///{tmp_path / 'test.db'}")
    config.settings.ensure_dirs()

    from app import models  # noqa: F401 — register tables on Base

    engine = create_engine(config.settings.resolved_db_url(),
                           connect_args={"check_same_thread": False})
    Base.metadata.create_all(engine)
    TestSession = sessionmaker(bind=engine, expire_on_commit=False)
    monkeypatch.setattr(db_module, "engine", engine)
    monkeypatch.setattr(db_module, "SessionLocal", TestSession)

    session = TestSession()
    try:
        yield session
    finally:
        session.close()


@pytest.fixture
def client(db_session):
    """TestClient sharing db_session's engine. Lifespan is skipped deliberately — it would
    re-init the real app.db — so anything init_db() normally does is already done above."""
    from fastapi.testclient import TestClient

    from app.db import get_db
    from app.main import app

    app.dependency_overrides[get_db] = lambda: db_session
    try:
        yield TestClient(app)
    finally:
        app.dependency_overrides.clear()
