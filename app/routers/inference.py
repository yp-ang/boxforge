"""Step 06: verify playground — image/video/webcam inference against the exported
ONNX model. Never best.pt; see README ground rule 3 and services/runtime.py."""
from __future__ import annotations

import base64
import json
import uuid
from pathlib import Path

import cv2
import numpy as np
from fastapi import (
    APIRouter,
    Depends,
    File,
    HTTPException,
    Query,
    UploadFile,
    WebSocket,
    WebSocketDisconnect,
)
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session

from app.config import settings
from app.db import get_db
from app.models import Job, Model
from app.schemas import JobOut, PredictImageOut, WebcamCapabilityOut
from app.services.jobs import JobConflict, start_job
from app.services.runtime import get_detector

router = APIRouter(tags=["inference"])

# Run once at a floor low enough that the confidence slider (0.05-0.95 in the UI) has
# real detections to filter client-side — see step 06 §4.
PREDICT_FLOOR_CONF = 0.01


def _exported_model(db: Session, model_id: int) -> Model:
    model = db.get(Model, model_id)
    if not model:
        raise HTTPException(404, "model not found")
    if not model.onnx_path or not Path(model.onnx_path).is_file():
        raise HTTPException(400, f"model {model_id} has no exported ONNX — run the "
                                  "step 05 export first")
    return model


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


# --- image ---------------------------------------------------------------------

@router.post("/api/models/{model_id}/predict/image", response_model=PredictImageOut)
async def predict_image(model_id: int, file: UploadFile = File(...),
                        db: Session = Depends(get_db)):
    model = _exported_model(db, model_id)
    bgr = _decode_upload(await file.read())
    detector = get_detector(model.id, Path(model.dir_path))
    dets = await run_in_threadpool(detector.predict, bgr, PREDICT_FLOOR_CONF)
    h, w = bgr.shape[:2]
    return PredictImageOut(width=w, height=h, image=_jpeg_data_uri(bgr),
                           detections=[d.as_dict() for d in dets])


# --- video (background job, reusing step 04's runner) ---------------------------

@router.post("/api/models/{model_id}/predict/video", response_model=JobOut)
async def predict_video(
    model_id: int,
    file: UploadFile = File(...),
    stride: int = Query(3, ge=1, le=30, description="infer every Nth frame"),
    conf: float = Query(0.25, ge=0.0, le=1.0),
    iou: float = Query(0.45, ge=0.0, le=1.0),
    db: Session = Depends(get_db),
):
    model = _exported_model(db, model_id)

    upload_dir = settings.data_dir / "uploads"
    upload_dir.mkdir(parents=True, exist_ok=True)
    suffix = Path(file.filename or "video.mp4").suffix or ".mp4"
    src_path = upload_dir / f"predict-{model.id}-{uuid.uuid4().hex}{suffix}"
    src_path.write_bytes(await file.read())

    params = {"model_id": model.id, "src_path": str(src_path), "stride": stride,
              "conf": conf, "iou": iou}
    try:
        return start_job(db, "predict_video", model.project_id, params)
    except JobConflict as exc:
        src_path.unlink(missing_ok=True)
        raise HTTPException(409, str(exc)) from exc


def _predict_job_or_404(db: Session, job_id: int) -> Job:
    job = db.get(Job, job_id)
    if not job or job.type != "predict_video":
        raise HTTPException(404, "video prediction job not found")
    return job


@router.get("/api/jobs/{job_id}/video")
def job_video(job_id: int, db: Session = Depends(get_db)):
    _predict_job_or_404(db, job_id)
    path = settings.runs_dir / str(job_id) / "predict" / "annotated.mp4"
    if not path.is_file():
        raise HTTPException(404, "annotated video not ready yet")
    return FileResponse(path, media_type="video/mp4")


@router.get("/api/jobs/{job_id}/detections")
def job_detections(job_id: int, db: Session = Depends(get_db)):
    _predict_job_or_404(db, job_id)
    path = settings.runs_dir / str(job_id) / "predict" / "detections.json"
    if not path.is_file():
        raise HTTPException(404, "detections not ready yet")
    return FileResponse(path, media_type="application/json")


# --- webcam ----------------------------------------------------------------------

@router.get("/api/capabilities/webcam", response_model=WebcamCapabilityOut)
def webcam_capability():
    """Explicit, user-triggered check (ARCHITECTURE §5.2) — never run automatically,
    since opening the camera can itself trigger an OS permission prompt."""
    cap = cv2.VideoCapture(0)
    try:
        ok = cap.isOpened()
    finally:
        cap.release()
    return WebcamCapabilityOut(
        server_webcam=ok,
        note=None if ok else "no server-side camera in this environment (expected in "
                              "Docker on macOS) — use the browser webcam tab instead",
    )


@router.websocket("/ws/predict/{model_id}")
async def ws_predict(ws: WebSocket, model_id: int):
    await ws.accept()

    # Imported here, not at module scope: tests monkeypatch app.db.SessionLocal onto an
    # isolated engine, and a name bound at import time would keep pointing at the real
    # one (see services/jobs.py's job_status(), same reasoning).
    from app.db import SessionLocal

    with SessionLocal() as session:
        model = session.get(Model, model_id)
        if not model or not model.onnx_path or not Path(model.onnx_path).is_file():
            await ws.close(code=4004, reason="model not found or not exported")
            return
        model_dir = Path(model.dir_path)

    detector = get_detector(model_id, model_dir)
    conf, iou = settings.default_conf, 0.45

    try:
        while True:
            message = await ws.receive()
            if message["type"] == "websocket.disconnect":
                break
            text = message.get("text")
            if text is not None:
                # Control message updating the threshold — see step 06 §4. Malformed
                # control messages are ignored rather than dropping the connection.
                try:
                    control = json.loads(text)
                    conf = float(control.get("conf", conf))
                    iou = float(control.get("iou", iou))
                except (ValueError, TypeError, json.JSONDecodeError):
                    pass
                continue

            data = message.get("bytes")
            if not data:
                continue
            frame = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
            if frame is None:
                continue
            # onnxruntime is blocking; running it inline would stall the event loop
            # (and every other request the server is handling) for the frame's duration.
            dets = await run_in_threadpool(detector.predict, frame, conf, iou)
            await ws.send_json([d.as_dict() for d in dets])
    except WebSocketDisconnect:
        pass
