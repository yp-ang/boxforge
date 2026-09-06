from datetime import datetime
from typing import Optional

from sqlalchemy import ForeignKey, LargeBinary, Text, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db import Base


class Project(Base):
    __tablename__ = "projects"

    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(unique=True)
    slug: Mapped[str] = mapped_column(unique=True)
    task: Mapped[str] = mapped_column(default="detect")
    created_at: Mapped[datetime] = mapped_column(default=datetime.utcnow)

    labels: Mapped[list["Label"]] = relationship(back_populates="project",
                                                  cascade="all, delete-orphan")
    images: Mapped[list["Image"]] = relationship(back_populates="project",
                                                  cascade="all, delete-orphan")


class Label(Base):
    __tablename__ = "labels"
    __table_args__ = (
        UniqueConstraint("project_id", "class_index"),
        UniqueConstraint("project_id", "name"),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    project_id: Mapped[int] = mapped_column(ForeignKey("projects.id", ondelete="CASCADE"))
    name: Mapped[str]
    color: Mapped[str] = mapped_column(default="#ff3b30")
    class_index: Mapped[int]

    project: Mapped["Project"] = relationship(back_populates="labels")


class Image(Base):
    __tablename__ = "images"
    __table_args__ = (UniqueConstraint("project_id", "sha256"),)

    id: Mapped[int] = mapped_column(primary_key=True)
    project_id: Mapped[int] = mapped_column(ForeignKey("projects.id", ondelete="CASCADE"))
    rel_path: Mapped[str]           # relative to settings.images_dir
    sha256: Mapped[str] = mapped_column(index=True)
    width: Mapped[int]
    height: Mapped[int]
    status: Mapped[str] = mapped_column(default="pending", index=True)
    ingested_at: Mapped[datetime] = mapped_column(default=datetime.utcnow)

    project: Mapped["Project"] = relationship(back_populates="images")
    annotations: Mapped[list["Annotation"]] = relationship(back_populates="image",
                                                            cascade="all, delete-orphan")


class Annotation(Base):
    __tablename__ = "annotations"

    id: Mapped[int] = mapped_column(primary_key=True)
    image_id: Mapped[int] = mapped_column(ForeignKey("images.id", ondelete="CASCADE"),
                                          index=True)
    label_id: Mapped[int] = mapped_column(ForeignKey("labels.id", ondelete="CASCADE"))
    x1: Mapped[float]
    y1: Mapped[float]
    x2: Mapped[float]
    y2: Mapped[float]
    source: Mapped[str] = mapped_column(default="human")     # human | model
    created_at: Mapped[datetime] = mapped_column(default=datetime.utcnow)

    image: Mapped["Image"] = relationship(back_populates="annotations")


class Job(Base):
    __tablename__ = "jobs"

    id: Mapped[int] = mapped_column(primary_key=True)
    type: Mapped[str]                                         # train | export | prelabel
    status: Mapped[str] = mapped_column(default="queued", index=True)
    project_id: Mapped[int] = mapped_column(ForeignKey("projects.id", ondelete="CASCADE"))
    params_json: Mapped[Optional[str]] = mapped_column(Text, default=None)
    log_path: Mapped[Optional[str]] = mapped_column(default=None)
    result_json: Mapped[Optional[str]] = mapped_column(Text, default=None)
    started_at: Mapped[Optional[datetime]] = mapped_column(default=None)
    ended_at: Mapped[Optional[datetime]] = mapped_column(default=None)


class Model(Base):
    __tablename__ = "models"

    id: Mapped[int] = mapped_column(primary_key=True)
    project_id: Mapped[int] = mapped_column(ForeignKey("projects.id", ondelete="CASCADE"))
    job_id: Mapped[Optional[int]] = mapped_column(ForeignKey("jobs.id", ondelete="SET NULL"),
                                                   default=None)
    name: Mapped[str]
    dir_path: Mapped[str]
    classes_json: Mapped[str] = mapped_column(Text)
    metrics_json: Mapped[Optional[str]] = mapped_column(Text, default=None)
    created_at: Mapped[datetime] = mapped_column(default=datetime.utcnow)


class FaceIdentity(Base):
    __tablename__ = "face_identities"

    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str]
    notes: Mapped[Optional[str]] = mapped_column(Text, default=None)
    created_at: Mapped[datetime] = mapped_column(default=datetime.utcnow)


class FaceEmbedding(Base):
    __tablename__ = "face_embeddings"

    id: Mapped[int] = mapped_column(primary_key=True)
    identity_id: Mapped[int] = mapped_column(ForeignKey("face_identities.id", ondelete="CASCADE"))
    vector_blob: Mapped[bytes] = mapped_column(LargeBinary)
    source_image: Mapped[str]
    created_at: Mapped[datetime] = mapped_column(default=datetime.utcnow)
