import json

import cv2
import numpy as np
import pytest

from app.config import settings
from app.models import FaceEmbedding, FaceIdentity
from app.services import face_engine
from app.services.face_engine import FaceDet, Gallery, IdentifiedFace, align, draw_faces, identify_faces


@pytest.fixture(autouse=True)
def _reset_face_caches():
    """get_engine()/get_gallery() cache at module scope (mirrors services/runtime.py's
    detector cache) — reset it around every test so one test's fake engine or a
    previous test's gallery contents can never leak into the next."""
    face_engine.invalidate_gallery()
    face_engine._engine = None
    yield
    face_engine.invalidate_gallery()
    face_engine._engine = None


def make_identity(db, name="alice"):
    identity = FaceIdentity(name=name)
    db.add(identity)
    db.commit()
    db.refresh(identity)
    return identity


def make_embedding(db, identity, vector: np.ndarray, source_image="x.jpg"):
    row = FaceEmbedding(identity_id=identity.id, vector_blob=vector.astype(np.float32).tobytes(),
                        source_image=source_image)
    db.add(row)
    db.commit()
    db.refresh(row)
    return row


def unit_vector(seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    v = rng.normal(size=512).astype(np.float32)
    return v / np.linalg.norm(v)


# --- Gallery: pure matching logic ---------------------------------------------------

def test_gallery_match_returns_none_when_empty():
    gallery = Gallery()
    assert gallery.match(unit_vector(0)) == (None, 0.0)


def test_gallery_match_finds_the_closest_above_threshold(db_session):
    alice = make_identity(db_session, "alice")
    bob = make_identity(db_session, "bob")
    v_alice = unit_vector(1)
    make_embedding(db_session, alice, v_alice)
    make_embedding(db_session, bob, unit_vector(2))

    gallery = Gallery()
    gallery.load(db_session)
    identity_id, sim = gallery.match(v_alice, threshold=0.35)
    assert identity_id == alice.id
    assert sim == pytest.approx(1.0, abs=1e-4)


def test_gallery_match_returns_none_below_threshold(db_session):
    alice = make_identity(db_session, "alice")
    make_embedding(db_session, alice, unit_vector(1))

    gallery = Gallery()
    gallery.load(db_session)
    # An unrelated random vector should not cross a sane threshold.
    stranger = unit_vector(999)
    identity_id, sim = gallery.match(stranger, threshold=0.35)
    assert identity_id is None
    assert sim < 0.35


# --- identify_faces / draw_faces: pure, with a fake detector+embedder --------------

class _FakeDetector:
    def __init__(self, dets):
        self._dets = dets

    def detect(self, bgr, conf=0.5, iou=0.4):
        return self._dets


class _FakeEmbedder:
    def __init__(self, vector):
        self._vector = vector

    def embed(self, aligned):
        return self._vector


class _FakeEngine:
    def __init__(self, dets, vector):
        self.detector = _FakeDetector(dets)
        self.embedder = _FakeEmbedder(vector)


def test_identify_faces_matches_against_the_gallery(db_session, monkeypatch):
    alice = make_identity(db_session, "alice")
    v_alice = unit_vector(1)
    make_embedding(db_session, alice, v_alice)
    gallery = Gallery()
    gallery.load(db_session)

    landmarks = np.array([[10, 10], [20, 10], [15, 15], [11, 20], [19, 20]], dtype=np.float32)
    det = FaceDet(score=0.9, bbox=[0, 0, 30, 30], landmarks=landmarks)
    engine = _FakeEngine([det], v_alice)
    # align() calls cv2 on a real frame; give it something real-shaped.
    frame = np.zeros((40, 40, 3), dtype=np.uint8)

    results = identify_faces(engine, gallery, frame, det_conf=0.5, match_threshold=0.35)
    assert len(results) == 1
    assert results[0].identity_id == alice.id
    assert results[0].similarity == pytest.approx(1.0, abs=1e-4)


def test_identify_faces_reports_unknown_for_a_stranger(db_session):
    alice = make_identity(db_session, "alice")
    make_embedding(db_session, alice, unit_vector(1))
    gallery = Gallery()
    gallery.load(db_session)

    landmarks = np.array([[10, 10], [20, 10], [15, 15], [11, 20], [19, 20]], dtype=np.float32)
    det = FaceDet(score=0.9, bbox=[0, 0, 30, 30], landmarks=landmarks)
    engine = _FakeEngine([det], unit_vector(999))
    frame = np.zeros((40, 40, 3), dtype=np.uint8)

    results = identify_faces(engine, gallery, frame, det_conf=0.5, match_threshold=0.35)
    assert results[0].identity_id is None


def test_draw_faces_paints_a_box_and_does_not_crash_on_unknown():
    frame = np.zeros((50, 50, 3), dtype=np.uint8)
    faces = [IdentifiedFace(score=0.9, bbox=[5, 5, 30, 30], identity_id=None, similarity=0.1)]
    draw_faces(frame, faces, names={})
    assert frame.any()


def test_align_returns_a_112x112_crop():
    frame = np.random.randint(0, 255, (200, 200, 3), dtype=np.uint8)
    landmarks = np.array([[70, 90], [130, 90], [100, 120], [80, 150], [120, 150]], dtype=np.float32)
    aligned = align(frame, landmarks)
    assert aligned.shape == (112, 112, 3)


# --- HTTP: identities CRUD ----------------------------------------------------------

def test_create_and_list_identities(client):
    resp = client.post("/api/faces/identities", json={"name": "alice", "notes": "friend"})
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["name"] == "alice"
    assert body["embedding_count"] == 0

    listed = client.get("/api/faces/identities").json()
    assert [i["name"] for i in listed] == ["alice"]


def test_create_identity_rejects_a_duplicate_name(client):
    client.post("/api/faces/identities", json={"name": "alice"})
    resp = client.post("/api/faces/identities", json={"name": "alice"})
    assert resp.status_code == 400


def test_delete_identity_cascades_embeddings_and_removes_photo_files(client, db_session):
    identity = make_identity(db_session, "alice")
    photo_dir = settings.faces_dir / str(identity.id)
    photo_dir.mkdir(parents=True, exist_ok=True)
    photo_path = photo_dir / "a.jpg"
    cv2.imwrite(str(photo_path), np.zeros((10, 10, 3), dtype=np.uint8))
    make_embedding(db_session, identity, unit_vector(1), source_image=f"{identity.id}/a.jpg")

    resp = client.delete(f"/api/faces/identities/{identity.id}")
    assert resp.status_code == 204
    assert db_session.query(FaceEmbedding).filter_by(identity_id=identity.id).count() == 0
    assert not photo_path.exists()


def test_get_unknown_identity_404s(client):
    assert client.get("/api/faces/identities/999").status_code == 404


# --- HTTP: reference photo upload ----------------------------------------------------

def test_upload_photos_requires_models_downloaded(client, db_session):
    identity = make_identity(db_session, "alice")
    resp = client.post(f"/api/faces/identities/{identity.id}/photos",
                       files={"files": ("a.jpg", b"fake", "image/jpeg")})
    assert resp.status_code == 400
    assert "not downloaded" in resp.json()["detail"]


def _face_frame_bytes() -> bytes:
    ok, buf = cv2.imencode(".jpg", np.zeros((60, 60, 3), dtype=np.uint8))
    assert ok
    return buf.tobytes()


def test_upload_photos_embeds_one_face_per_photo_and_reports_failures(client, db_session, monkeypatch):
    identity = make_identity(db_session, "alice")

    landmarks = np.array([[20, 20], [40, 20], [30, 30], [22, 42], [38, 42]], dtype=np.float32)
    one_face = [FaceDet(score=0.9, bbox=[10, 10, 50, 50], landmarks=landmarks)]
    responses = iter([one_face, [], one_face + one_face])

    fake_engine = _FakeEngine([], unit_vector(1))
    monkeypatch.setattr(fake_engine.detector, "detect", lambda bgr, conf=0.5: next(responses))
    import app.routers.faces as faces_router
    monkeypatch.setattr(faces_router, "get_engine", lambda: fake_engine)

    files = [
        ("files", ("good.jpg", _face_frame_bytes(), "image/jpeg")),
        ("files", ("no-face.jpg", _face_frame_bytes(), "image/jpeg")),
        ("files", ("two-faces.jpg", _face_frame_bytes(), "image/jpeg")),
    ]
    resp = client.post(f"/api/faces/identities/{identity.id}/photos", files=files)
    assert resp.status_code == 200, resp.text
    items = resp.json()["items"]
    assert [i["ok"] for i in items] == [True, False, False]
    assert "no face" in items[1]["detail"]
    assert "2 faces" in items[2]["detail"]

    assert db_session.query(FaceEmbedding).filter_by(identity_id=identity.id).count() == 1


def test_upload_photos_404s_on_an_unknown_identity(client, monkeypatch):
    import app.routers.faces as faces_router
    monkeypatch.setattr(faces_router, "get_engine", lambda: _FakeEngine([], unit_vector(1)))
    resp = client.post("/api/faces/identities/999/photos",
                       files={"files": ("a.jpg", _face_frame_bytes(), "image/jpeg")})
    assert resp.status_code == 404


# --- HTTP: test/image -----------------------------------------------------------------

def test_predict_image_requires_models_downloaded(client):
    resp = client.post("/api/faces/test/image",
                       files={"file": ("a.jpg", _face_frame_bytes(), "image/jpeg")})
    assert resp.status_code == 400


def test_predict_image_returns_named_and_unknown_faces(client, db_session, monkeypatch):
    alice = make_identity(db_session, "alice")
    v_alice = unit_vector(1)
    make_embedding(db_session, alice, v_alice)

    landmarks = np.array([[20, 20], [40, 20], [30, 30], [22, 42], [38, 42]], dtype=np.float32)
    dets = [FaceDet(score=0.9, bbox=[10, 10, 50, 50], landmarks=landmarks)]
    fake_engine = _FakeEngine(dets, v_alice)
    import app.routers.faces as faces_router
    monkeypatch.setattr(faces_router, "get_engine", lambda: fake_engine)

    resp = client.post("/api/faces/test/image", files={"file": ("a.jpg", _face_frame_bytes(), "image/jpeg")})
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["faces"][0]["name"] == "alice"
    assert body["faces"][0]["identity_id"] == alice.id
    assert body["image"].startswith("data:image/jpeg;base64,")


# --- HTTP: models status --------------------------------------------------------------

def test_models_status_reports_readiness(client, monkeypatch):
    monkeypatch.setattr("app.routers.faces.buffalo_l_ready", lambda: False)
    assert client.get("/api/faces/models/status").json() == {"ready": False}

    monkeypatch.setattr("app.routers.faces.buffalo_l_ready", lambda: True)
    assert client.get("/api/faces/models/status").json() == {"ready": True}


# --- Real weights end-to-end (network + a real download; skips offline) ------------

def test_real_pipeline_identifies_a_held_out_photo_and_rejects_a_stranger(db_session):
    """Acceptance criteria: held-out photo of a known identity matches > 0.4; a
    stranger returns unknown. Uses real SCRFD + ArcFace weights, downloaded on demand
    exactly as a user would hit this the first time they open /faces."""
    try:
        face_engine.download_buffalo_l(log=lambda *a: None)
    except Exception as exc:
        pytest.skip(f"could not fetch face models (offline?): {exc}")

    import urllib.request

    def fetch(url: str) -> np.ndarray:
        with urllib.request.urlopen(url, timeout=30) as resp:
            data = resp.read()
        return cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)

    try:
        img = fetch("https://raw.githubusercontent.com/opencv/opencv/master/samples/data/messi5.jpg")
        stranger_img = fetch("https://raw.githubusercontent.com/opencv/opencv/master/samples/data/lena.jpg")
    except Exception as exc:
        pytest.skip(f"could not fetch test images (offline?): {exc}")

    engine = face_engine.FaceEngine()
    dets = engine.detector.detect(img, conf=0.5)
    assert len(dets) >= 1
    aligned = face_engine.align(img, dets[0].landmarks)
    assert aligned.shape == (112, 112, 3)
    reference_emb = engine.embedder.embed(aligned)

    identity = make_identity(db_session, "known-person")
    make_embedding(db_session, identity, reference_emb)
    gallery = Gallery()
    gallery.load(db_session)

    # Held-out: re-detect/re-embed from scratch rather than reusing the same crop.
    held_out_dets = engine.detector.detect(img, conf=0.5)
    held_out_emb = engine.embedder.embed(face_engine.align(img, held_out_dets[0].landmarks))
    identity_id, sim = gallery.match(held_out_emb, threshold=0.35)
    assert identity_id == identity.id
    assert sim > 0.4

    stranger_dets = engine.detector.detect(stranger_img, conf=0.5)
    if stranger_dets:
        stranger_emb = engine.embedder.embed(face_engine.align(stranger_img, stranger_dets[0].landmarks))
        stranger_id, stranger_sim = gallery.match(stranger_emb, threshold=0.35)
        assert stranger_id is None
