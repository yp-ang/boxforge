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
    config.settings.ensure_dirs()

    from app import models  # noqa: F401 — register tables on Base

    engine = create_engine(f"sqlite:///{tmp_path / 'test.db'}", connect_args={"check_same_thread": False})
    Base.metadata.create_all(engine)
    TestSession = sessionmaker(bind=engine, expire_on_commit=False)
    monkeypatch.setattr(db_module, "engine", engine)
    monkeypatch.setattr(db_module, "SessionLocal", TestSession)

    session = TestSession()
    try:
        yield session
    finally:
        session.close()
