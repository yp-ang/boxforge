"""Step 08: face detection + recognition. Read ARCHITECTURE §5.4 and
docs/steps/08-faces.md before pointing this at anything but your own photos."""
from __future__ import annotations

import base64
import io
import json
import uuid
import zipfile
from pathlib import Path

import cv2
import numpy as np
from fastapi import (
    APIRouter,
    Depends,
    File,
    HTTPException,
    UploadFile,
    WebSocket,
    WebSocketDisconnect,
)
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse, StreamingResponse
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.config import settings
from app.db import get_db
from app.models import FaceEmbedding, FaceIdentity
from app.schemas import (
    FaceDetectionOut,
    FaceEmbeddingOut,
    FaceIdentityCreate,
    FaceIdentityDetailOut,
    FaceIdentityOut,
    FaceModelsStatusOut,
    FacePredictImageOut,
    FaceUploadItemOut,
    FaceUploadReportOut,
)
from app.services.face_engine import (
    DEFAULT_DET_CONF,
    DEFAULT_MATCH_THRESHOLD,
    align,
    buffalo_l_ready,
    download_buffalo_l,
    draw_faces,
    get_engine,
    get_gallery,
    identify_faces,
    invalidate_gallery,
)

router = APIRouter(prefix="/api/faces", tags=["faces"])


def _decode_upload(data: bytes) -> np.ndarray:
    bgr = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if bgr is None:
        raise HTTPException(400, "could not decode image")
    return bgr


def _jpeg_data_uri(bgr: np.ndarray) -> str:
    ok, buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, 90])
    if not ok:
        raise HTTPException(500, "could not encode image")
    return "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode()


def _names_by_id(db: Session) -> dict[int, str]:
    return {i.id: i.name for i in db.scalars(select(FaceIdentity))}


# --- identities ---------------------------------------------------------------------

@router.get("/identities", response_model=list[FaceIdentityOut])
def list_identities(db: Session = Depends(get_db)):
    counts = dict(db.execute(
        select(FaceEmbedding.identity_id, func.count(FaceEmbedding.id))
        .group_by(FaceEmbedding.identity_id)
    ).all())
    identities = db.scalars(select(FaceIdentity).order_by(FaceIdentity.name)).all()
    return [
        FaceIdentityOut(id=i.id, name=i.name, notes=i.notes, created_at=i.created_at,
                        embedding_count=counts.get(i.id, 0))
        for i in identities
    ]


@router.post("/identities", response_model=FaceIdentityOut)
def create_identity(payload: FaceIdentityCreate, db: Session = Depends(get_db)):
    if db.scalar(select(FaceIdentity).where(FaceIdentity.name == payload.name)):
        raise HTTPException(400, "an identity with this name already exists")
    identity = FaceIdentity(name=payload.name, notes=payload.notes)
    db.add(identity)
    db.commit()
    db.refresh(identity)
    return FaceIdentityOut(id=identity.id, name=identity.name, notes=identity.notes,
                           created_at=identity.created_at, embedding_count=0)


@router.get("/identities/{identity_id}", response_model=FaceIdentityDetailOut)
def get_identity(identity_id: int, db: Session = Depends(get_db)):
    identity = db.get(FaceIdentity, identity_id)
    if not identity:
        raise HTTPException(404, "identity not found")
    embeddings = db.scalars(
        select(FaceEmbedding).where(FaceEmbedding.identity_id == identity_id)
        .order_by(FaceEmbedding.id)
    ).all()
    return FaceIdentityDetailOut(
        identity=FaceIdentityOut(id=identity.id, name=identity.name, notes=identity.notes,
                                 created_at=identity.created_at, embedding_count=len(embeddings)),
        embeddings=embeddings,
    )


@router.delete("/identities/{identity_id}", status_code=204)
def delete_identity(identity_id: int, db: Session = Depends(get_db)):
    """Deletability is a feature here, not an afterthought (doc §5): removes every
    embedding this identity has, and the reference photos on disk with them."""
    identity = db.get(FaceIdentity, identity_id)
    if not identity:
        raise HTTPException(404, "identity not found")
    db.delete(identity)   # FaceEmbedding rows cascade at the DB level (ON DELETE CASCADE)
    db.commit()
    invalidate_gallery()

    identity_dir = settings.faces_dir / str(identity_id)
    if identity_dir.is_dir():
        import shutil
        shutil.rmtree(identity_dir, ignore_errors=True)


@router.post("/identities/{identity_id}/photos", response_model=FaceUploadReportOut)
async def upload_reference_photos(
    identity_id: int, files: list[UploadFile] = File(...), db: Session = Depends(get_db),
):
    """5-10 reference photos per person, varied in pose/lighting/expression (doc §3).
    Each photo must show exactly one face — ambiguous about whose embedding it would be
    otherwise — and is embedded immediately; there is no training step to wait for."""
    identity = db.get(FaceIdentity, identity_id)
    if not identity:
        raise HTTPException(404, "identity not found")

    try:
        engine = get_engine()
    except RuntimeError as exc:
        raise HTTPException(400, str(exc)) from exc

    identity_dir = settings.faces_dir / str(identity_id)
    identity_dir.mkdir(parents=True, exist_ok=True)

    items: list[FaceUploadItemOut] = []
    for file in files:
        data = await file.read()
        bgr = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
        if bgr is None:
            items.append(FaceUploadItemOut(filename=file.filename or "?", ok=False,
                                           detail="could not decode image"))
            continue

        dets = engine.detector.detect(bgr, conf=DEFAULT_DET_CONF)
        if not dets:
            items.append(FaceUploadItemOut(filename=file.filename or "?", ok=False,
                                           detail="no face found"))
            continue
        if len(dets) > 1:
            items.append(FaceUploadItemOut(filename=file.filename or "?", ok=False,
                                           detail=f"{len(dets)} faces found — crop to one person"))
            continue

        aligned = align(bgr, dets[0].landmarks)
        emb = engine.embedder.embed(aligned)

        ext = Path(file.filename or "photo.jpg").suffix or ".jpg"
        dest_name = f"{uuid.uuid4().hex}{ext}"
        cv2.imwrite(str(identity_dir / dest_name), bgr)

        row = FaceEmbedding(identity_id=identity_id, vector_blob=emb.astype(np.float32).tobytes(),
                            source_image=str(Path(str(identity_id)) / dest_name))
        db.add(row)
        db.commit()
        db.refresh(row)
        items.append(FaceUploadItemOut(filename=file.filename or "?", ok=True,
                                       detail="embedded", embedding_id=row.id))

    if any(i.ok for i in items):
        invalidate_gallery()
    return FaceUploadReportOut(items=items)


@router.get("/embeddings/{embedding_id}/photo")
def embedding_photo(embedding_id: int, db: Session = Depends(get_db)):
    row = db.get(FaceEmbedding, embedding_id)
    if not row:
        raise HTTPException(404, "embedding not found")
    resolved = (settings.faces_dir / row.source_image).resolve()
    if not resolved.is_relative_to(settings.faces_dir.resolve()):
        raise HTTPException(400, "path escape")
    if not resolved.is_file():
        raise HTTPException(404, "photo missing on disk")
    return FileResponse(resolved)


@router.delete("/embeddings/{embedding_id}", status_code=204)
def delete_embedding(embedding_id: int, db: Session = Depends(get_db)):
    row = db.get(FaceEmbedding, embedding_id)
    if not row:
        raise HTTPException(404, "embedding not found")
    photo_path = settings.faces_dir / row.source_image
    db.delete(row)
    db.commit()
    invalidate_gallery()
    photo_path.unlink(missing_ok=True)


# --- model weights (downloaded on demand, like yolo11n.pt) --------------------------

@router.get("/models/status", response_model=FaceModelsStatusOut)
def models_status():
    return FaceModelsStatusOut(ready=buffalo_l_ready())


@router.post("/models/download", response_model=FaceModelsStatusOut)
async def models_download():
    """Blocking on purpose: this is a ~190MB one-time fetch of SCRFD + ArcFace, no
    different in spirit from the pretrained YOLO checkpoint step 04 downloads on first
    use — just too big to hide inside a training job's first log line."""
    try:
        await run_in_threadpool(download_buffalo_l)
    except Exception as exc:
        raise HTTPException(502, f"could not download face models: {exc}") from exc
    return FaceModelsStatusOut(ready=buffalo_l_ready())


# --- test: image ---------------------------------------------------------------------

@router.post("/test/image", response_model=FacePredictImageOut)
async def predict_image(
    file: UploadFile = File(...),
    det_conf: float = DEFAULT_DET_CONF,
    match_threshold: float = DEFAULT_MATCH_THRESHOLD,
    db: Session = Depends(get_db),
):
    try:
        engine = get_engine()
    except RuntimeError as exc:
        raise HTTPException(400, str(exc)) from exc

    bgr = _decode_upload(await file.read())
    gallery = get_gallery(db)
    faces = await run_in_threadpool(identify_faces, engine, gallery, bgr, det_conf, match_threshold)

    names = _names_by_id(db)
    annotated = bgr.copy()
    draw_faces(annotated, faces, names)
    h, w = bgr.shape[:2]

    return FacePredictImageOut(
        width=w, height=h, image=_jpeg_data_uri(annotated),
        faces=[
            FaceDetectionOut(bbox=f.bbox, score=f.score, identity_id=f.identity_id,
                             name=names.get(f.identity_id, "unknown") if f.identity_id else "unknown",
                             similarity=f.similarity)
            for f in faces
        ],
    )


# --- test: webcam (browser-captured, same pattern as step 06) ----------------------

@router.websocket("/ws/predict")
async def ws_predict(ws: WebSocket):
    await ws.accept()

    from app.db import SessionLocal

    try:
        engine = get_engine()
    except RuntimeError as exc:
        await ws.close(code=4004, reason=str(exc))
        return

    with SessionLocal() as session:
        gallery = get_gallery(session)
        names = _names_by_id(session)

    det_conf, match_threshold = DEFAULT_DET_CONF, DEFAULT_MATCH_THRESHOLD

    try:
        while True:
            message = await ws.receive()
            if message["type"] == "websocket.disconnect":
                break
            text = message.get("text")
            if text is not None:
                try:
                    control = json.loads(text)
                    det_conf = float(control.get("det_conf", det_conf))
                    match_threshold = float(control.get("match_threshold", match_threshold))
                except (ValueError, TypeError, json.JSONDecodeError):
                    pass
                continue

            data = message.get("bytes")
            if not data:
                continue
            frame = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
            if frame is None:
                continue
            faces = await run_in_threadpool(identify_faces, engine, gallery, frame,
                                            det_conf, match_threshold)
            await ws.send_json([
                {"bbox": f.bbox, "score": f.score, "identity_id": f.identity_id,
                 "name": names.get(f.identity_id, "unknown") if f.identity_id else "unknown",
                 "similarity": f.similarity}
                for f in faces
            ])
    except WebSocketDisconnect:
        pass


# --- export (doc §5) ------------------------------------------------------------------

@router.get("/export")
def export_faces(db: Session = Depends(get_db)):
    """SCRFD + ArcFace ONNX plus gallery.npz, so other software can run the same
    pipeline without this app or its database."""
    if not buffalo_l_ready():
        raise HTTPException(400, "face models not downloaded yet")

    from app.services.face_engine import buffalo_l_dir, build_gallery_npz

    gallery = build_gallery_npz(db)
    npz_buf = io.BytesIO()
    np.savez(npz_buf, vectors=gallery["vectors"], identity_ids=gallery["identity_ids"])

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        d = buffalo_l_dir()
        zf.write(d / "det_10g.onnx", "det_10g.onnx")
        zf.write(d / "w600k_r50.onnx", "w600k_r50.onnx")
        zf.writestr("gallery.npz", npz_buf.getvalue())
        zf.writestr("names.json", json.dumps(gallery["names"], indent=2))
    buf.seek(0)

    return StreamingResponse(
        buf, media_type="application/zip",
        headers={"Content-Disposition": "attachment; filename=faces-export.zip"},
    )
