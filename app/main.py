"""FastAPI application: API + static web interface."""
from __future__ import annotations

import base64
import csv
import io
import logging
import random
import secrets
import time
import zipfile
from contextlib import asynccontextmanager
from pathlib import Path, PurePosixPath
from typing import Iterable, Literal

from fastapi import APIRouter, Depends, FastAPI, File, Form, HTTPException, Request, Security, UploadFile
from fastapi.responses import FileResponse, Response
from fastapi.security import APIKeyHeader
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from . import config, generator, solver, store, verification

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")


@asynccontextmanager
async def lifespan(_: FastAPI):
    try:
        solver.load_model()  # avoid a slow first request
    except Exception as exc:  # app still starts; /api/solve reports the problem
        logging.getLogger("startup").warning("Solver model not loaded: %s", exc)
    yield


app = FastAPI(title="CAPTCHA Lab", version="1.1.0", lifespan=lifespan,
              description="Synthetic CAPTCHA solving + official token verification (test environment).")
STATIC = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=STATIC), name="static")


# ---------- API key (optional) ----------

_api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False,
                               description="Required only when API_KEY is set on the server.")


def require_api_key(key: str | None = Security(_api_key_header)) -> None:
    """Protects the API when API_KEY is configured; a no-op otherwise."""
    expected = config.API_KEY
    if expected and not (key and secrets.compare_digest(key.encode(), expected.encode())):
        raise HTTPException(401, "Missing or invalid API key. Send it in the X-API-Key header.")


api = APIRouter(prefix="/api", dependencies=[Depends(require_api_key)])


# Answers for generated challenges are kept server-side so results can be scored.
_challenges: dict[str, tuple[str, bytes, float]] = {}
CHALLENGE_TTL = 900


def _prune() -> None:
    now = time.time()
    for cid in [c for c, (_, _, ts) in _challenges.items() if now - ts > CHALLENGE_TTL]:
        del _challenges[cid]


def _score_and_record(result: solver.SolveResult, answer: str | None) -> dict:
    out = result.dict()
    correct = None if answer is None or not result.supported else result.solution == answer
    out["correct"] = correct
    if answer is not None:
        out["expected"] = answer
    if not result.supported:
        success, status = False, "unsupported"
    elif result.solution is None:
        success, status = False, "no_solution"
    elif correct is None:
        success, status = None, "solved_unscored"
    else:
        success, status = correct, "correct" if correct else "wrong"
    out["status"] = status
    store.record("solve", result.challenge_type, result.method, success, status,
                 result.processing_ms, result.confidence, result.est_cost_usd)
    return out


# ---------- open endpoints ----------

@app.get("/", include_in_schema=False)
def index():
    return FileResponse(STATIC / "index.html")


@app.get("/api/health")
def health():
    return {"status": "ok", "model_loaded": solver.MODEL_PATH.exists(),
            "auth_required": bool(config.API_KEY),
            "supported_challenge_types": ["synthetic_text", "synthetic_shape_grid"]}


@app.get("/api/config")
def public_config():
    """Public widget settings only. Secret keys never leave the server."""
    return {pid: {"name": p.name, "site_key": p.site_key, "script_url": p.script_url,
                  "widget_class": p.widget_class, "uses_test_keys": p.uses_test_keys,
                  "test_scenarios": sorted(p.test_scenarios) if p.uses_test_keys else []}
            for pid, p in config.PROVIDERS.items()}


# ---------- solving ----------

@api.post("/challenge/generate")
def generate_challenge(length: int = 5, kind: str = "text"):
    """Create a synthetic challenge: kind is "text" (4 to 6 characters) or "shape_grid"."""
    if kind not in generator.KINDS:
        raise HTTPException(422, f"kind must be one of {', '.join(generator.KINDS)}")
    if length not in (4, 5, 6):
        raise HTTPException(422, "length must be 4, 5 or 6")
    _prune()
    answer, img = generator.generate_kind(kind, length=length)
    png = generator.to_png_bytes(img)
    cid = secrets.token_urlsafe(12)
    _challenges[cid] = (answer, png, time.time())
    return {"challenge_id": cid, "kind": kind, "image_base64": base64.b64encode(png).decode()}


@api.post("/solve")
async def solve(challenge_id: str | None = Form(None), file: UploadFile | None = File(None)):
    """Solve a generated challenge (by id) or an uploaded image."""
    answer = None
    if challenge_id:
        entry = _challenges.get(challenge_id)
        if entry is None:
            raise HTTPException(404, "Unknown or expired challenge_id")
        answer, data, _ = entry
    elif file is not None:
        data = await file.read(config.MAX_UPLOAD_BYTES + 1)
        if len(data) > config.MAX_UPLOAD_BYTES:
            raise HTTPException(413, "File too large (max 2 MB)")
        if not data:
            raise HTTPException(400, "Empty file")
    else:
        raise HTTPException(400, "Send challenge_id or file")
    try:
        result = solver.solve(data)
    except RuntimeError as exc:
        raise HTTPException(503, str(exc))
    return _score_and_record(result, answer)


# ---------- batch processing ----------

def _run_batch(items: Iterable[tuple[str, bytes]]) -> dict:
    """Solve labelled challenges and report accuracy statistics."""
    total = correct = fallback = unsupported = 0
    total_ms, confs = 0.0, []
    by_type: dict[str, list[int]] = {}
    by_len: dict[int, list[int]] = {}
    failures = []
    for answer, data in items:
        try:
            res = solver.solve(data)
        except RuntimeError as exc:
            raise HTTPException(503, str(exc))
        ok = res.supported and res.solution == answer
        total += 1
        correct += ok
        fallback += res.fallback_used
        unsupported += not res.supported
        total_ms += res.processing_ms
        if res.confidence is not None:
            confs.append(res.confidence)
        by_type.setdefault(res.challenge_type, []).append(ok)
        if res.challenge_type == "synthetic_text":
            by_len.setdefault(len(answer), []).append(ok)
        if not ok and len(failures) < 10:
            failures.append({"expected": answer, "got": res.solution, "confidence": res.confidence,
                             "challenge_type": res.challenge_type})
        status = "correct" if ok else "wrong" if res.supported else "unsupported"
        store.record("solve", res.challenge_type, res.method, ok, status,
                     res.processing_ms, res.confidence, res.est_cost_usd)
    if not total:
        raise HTTPException(400, "The dataset holds no labelled images")
    rate = lambda hits: round(sum(hits) / len(hits), 4)
    return {
        "total": total, "correct": correct, "failed": total - correct,
        "unsupported": unsupported,
        "accuracy": round(correct / total, 4),
        "fallback_used": fallback,
        "avg_processing_ms": round(total_ms / total, 2),
        "avg_confidence": round(sum(confs) / len(confs), 4) if confs else None,
        "accuracy_by_type": {k: rate(v) for k, v in sorted(by_type.items())},
        "accuracy_by_length": {k: rate(v) for k, v in sorted(by_len.items())},
        "sample_failures": failures,
        "est_cost_per_challenge_usd": 0.0,
    }


def _generated(count: int, kind: str, rng: random.Random) -> Iterable[tuple[str, bytes]]:
    for _ in range(count):
        which = rng.choice(generator.KINDS) if kind == "mixed" else kind
        answer, img = generator.generate_kind(which, length=rng.choice([4, 5, 6]), rng=rng)
        yield answer, generator.to_png_bytes(img)


class BatchRequest(BaseModel):
    count: int = Field(100, ge=1, le=config.MAX_DATASET_ITEMS)
    kind: Literal["text", "shape_grid", "mixed"] = "text"
    seed: int | None = None


@api.post("/batch")
def batch(req: BatchRequest):
    """Accuracy statistics over a freshly generated synthetic dataset."""
    return {"source": "generated", **_run_batch(_generated(req.count, req.kind, random.Random(req.seed)))}


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".gif", ".webp"}


def _read_dataset(data: bytes) -> list[tuple[str, bytes]]:
    """Read a ZIP of labelled challenge images.

    Labels come from a labels.csv inside the ZIP (rows: filename,label). Without
    it, the label is the file name up to the first underscore ("K7MP2_001.png").
    """
    try:
        archive = zipfile.ZipFile(io.BytesIO(data))
    except zipfile.BadZipFile:
        raise HTTPException(400, "The dataset must be a ZIP file")
    with archive:
        files = [i for i in archive.infolist()
                 if not i.is_dir() and not PurePosixPath(i.filename).name.startswith(".")
                 and "__MACOSX" not in i.filename]
        images = [i for i in files if PurePosixPath(i.filename).suffix.lower() in IMAGE_SUFFIXES]
        if len(images) > config.MAX_DATASET_ITEMS:
            raise HTTPException(413, f"Too many images (max {config.MAX_DATASET_ITEMS})")
        if any(i.file_size > config.MAX_UPLOAD_BYTES for i in images):
            raise HTTPException(413, "An image in the dataset is larger than 2 MB")

        labels: dict[str, str] = {}
        sheet = next((i for i in files if PurePosixPath(i.filename).name.lower() == "labels.csv"), None)
        if sheet is not None and sheet.file_size <= config.MAX_UPLOAD_BYTES:
            text = archive.read(sheet).decode("utf-8-sig", errors="replace")
            for row in csv.reader(io.StringIO(text)):
                if len(row) >= 2 and row[0].strip():
                    labels[PurePosixPath(row[0].strip().replace("\\", "/")).name] = row[1].strip()

        items = []
        for info in sorted(images, key=lambda i: i.filename):
            name = PurePosixPath(info.filename).name
            label = labels.get(name) if labels else PurePosixPath(name).stem.split("_")[0]
            if label:
                items.append((label.upper(), archive.read(info)))
    return items


@api.post("/batch/upload")
async def batch_upload(file: UploadFile = File(...)):
    """Accuracy statistics over a supplied dataset: a ZIP of labelled images."""
    data = await file.read(config.MAX_DATASET_BYTES + 1)
    if len(data) > config.MAX_DATASET_BYTES:
        raise HTTPException(413, "Dataset too large (max 20 MB)")
    return {"source": "uploaded", **_run_batch(_read_dataset(data))}


@api.get("/dataset/sample")
def sample_dataset(count: int = 60, kind: str = "mixed", seed: int | None = None):
    """Download a labelled synthetic dataset (ZIP with images and labels.csv),
    ready to be sent back to /api/batch/upload."""
    if not 1 <= count <= 500:
        raise HTTPException(422, "count must be between 1 and 500")
    if kind not in (*generator.KINDS, "mixed"):
        raise HTTPException(422, "kind must be text, shape_grid or mixed")
    buf, rows = io.BytesIO(), io.StringIO()
    writer = csv.writer(rows, lineterminator="\n")
    writer.writerow(["filename", "label"])
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as archive:
        for n, (answer, png) in enumerate(_generated(count, kind, random.Random(seed)), 1):
            name = f"{n:04d}.png"
            archive.writestr(name, png)
            writer.writerow([name, answer])
        archive.writestr("labels.csv", rows.getvalue())
    return Response(buf.getvalue(), media_type="application/zip",
                    headers={"Content-Disposition": 'attachment; filename="captcha_dataset.zip"'})


# ---------- token verification ----------

class VerifyRequest(BaseModel):
    provider: str
    token: str | None = None
    # Official provider test secret that forces a provider-side result
    # ("always_fail", "already_spent"). Only honoured with the default test keys.
    scenario: str | None = None


@api.post("/verify")
async def verify(req: VerifyRequest, request: Request):
    result = await verification.verify(req.provider, req.token,
                                       request.client.host if request.client else None,
                                       scenario=req.scenario)
    store.record("verify", req.provider, "official_siteverify", result["success"],
                 result["status"], result["processing_ms"])
    return result


class DetectRequest(BaseModel):
    html: str = Field(..., max_length=500_000)


@api.post("/detect-provider")
def detect_provider(req: DetectRequest):
    return verification.detect_provider(req.html)


# ---------- reporting ----------

@api.get("/stats")
def stats():
    return store.stats()


@api.delete("/stats")
def reset_stats():
    store.reset()
    return {"status": "cleared"}


app.include_router(api)
