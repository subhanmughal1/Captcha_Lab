import asyncio
import io
import logging
import os
import random
import zipfile

os.environ["DB_PATH"] = "data/test_attempts.db"

import httpx
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from app import config, generator, solver, verification
from app.main import app

client = TestClient(app)


def png(img):
    return generator.to_png_bytes(img)


# ---------- solver ----------

def test_accuracy_on_synthetic_set():
    rng = random.Random(42)
    n, correct = 300, 0
    for _ in range(n):
        text, img = generator.generate(length=rng.choice([4, 5, 6]), rng=rng)
        correct += solver.solve(png(img)).solution == text
    assert correct / n >= 0.95


def test_accuracy_on_shape_grids():
    rng = random.Random(7)
    n, correct = 150, 0
    for _ in range(n):
        answer, img = generator.generate_shape_grid(rng)
        res = solver.solve(png(img))
        assert res.challenge_type == "synthetic_shape_grid" and res.supported
        correct += res.solution == answer
    assert correct / n >= 0.97


def test_shape_grid_reports_confidence_and_tile():
    answer, img = generator.generate_shape_grid(random.Random(3))
    res = solver.solve(png(img))
    assert res.solution == answer and 0 <= res.confidence <= 1
    assert f"tile {answer}" in res.detail and res.est_cost_usd == 0.0


def test_result_has_confidence_and_zero_cost():
    _, img = generator.generate(rng=random.Random(1))
    res = solver.solve(png(img))
    assert res.supported and res.challenge_type == "synthetic_text"
    assert 0 <= res.confidence <= 1 and res.est_cost_usd == 0.0


def test_non_image_is_rejected():
    res = solver.solve(b"this is not an image")
    assert res.challenge_type == "invalid_input" and not res.supported


def test_square_grid_image_is_unsupported():
    res = solver.solve(png(Image.new("RGB", (300, 300), (40, 120, 200))))
    assert not res.supported and res.solution is None


def test_blank_image_is_unsupported():
    res = solver.solve(png(Image.new("RGB", (220, 80), (255, 255, 255))))
    assert not res.supported


def test_grid_with_an_empty_tile_is_unsupported():
    _, img = generator.generate_shape_grid(random.Random(4))
    img.paste((250, 250, 250), (104, 104, 196, 196))  # wipe the centre tile
    res = solver.solve(png(img))
    assert not res.supported and res.solution is None


# ---------- API ----------

def test_generate_then_solve_is_scored():
    c = client.post("/api/challenge/generate?length=5").json()
    r = client.post("/api/solve", data={"challenge_id": c["challenge_id"]}).json()
    assert r["correct"] in (True, False) and len(r["expected"]) == 5


def test_solve_upload_and_bad_requests():
    _, img = generator.generate()
    r = client.post("/api/solve", files={"file": ("c.png", io.BytesIO(png(img)), "image/png")})
    assert r.status_code == 200 and r.json()["status"] == "solved_unscored"
    assert client.post("/api/solve").status_code == 400
    assert client.post("/api/solve", data={"challenge_id": "nope"}).status_code == 404


def test_generate_shape_grid_then_solve():
    c = client.post("/api/challenge/generate?kind=shape_grid").json()
    r = client.post("/api/solve", data={"challenge_id": c["challenge_id"]}).json()
    assert r["challenge_type"] == "synthetic_shape_grid" and r["expected"] in "123456789"
    assert client.post("/api/challenge/generate?kind=audio").status_code == 422


def test_batch_and_stats():
    r = client.post("/api/batch", json={"count": 30, "seed": 3}).json()
    assert r["total"] == 30 and 0 <= r["accuracy"] <= 1 and r["source"] == "generated"
    mixed = client.post("/api/batch", json={"count": 40, "seed": 5, "kind": "mixed"}).json()
    assert set(mixed["accuracy_by_type"]) == {"synthetic_text", "synthetic_shape_grid"}
    s = client.get("/api/stats").json()
    assert s["total_attempts"] >= 30 and s["est_cost_per_challenge_usd"] == 0.0


def make_zip(files):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as archive:
        for name, data in files.items():
            archive.writestr(name, data)
    return buf.getvalue()


def upload(data):
    return client.post("/api/batch/upload", files={"file": ("set.zip", io.BytesIO(data), "application/zip")})


def test_sample_dataset_round_trip():
    z = client.get("/api/dataset/sample?count=24&kind=mixed&seed=11")
    assert z.status_code == 200 and z.headers["content-type"] == "application/zip"
    names = zipfile.ZipFile(io.BytesIO(z.content)).namelist()
    assert "labels.csv" in names and len(names) == 25
    r = upload(z.content).json()
    assert r["source"] == "uploaded" and r["total"] == 24 and r["accuracy"] >= 0.9


def test_uploaded_dataset_labels_from_file_names():
    rng = random.Random(21)
    files = {}
    for i in range(6):
        text, img = generator.generate(rng=rng)
        files[f"set/{text}_{i}.png"] = png(img)
    files["set/notes.txt"] = b"ignored"
    r = upload(make_zip(files)).json()
    assert r["total"] == 6 and r["correct"] >= 5


def test_uploaded_dataset_counts_unsupported_images():
    _, img = generator.generate(text="ABCD", rng=random.Random(2))
    files = {"ABCD_1.png": png(img), "WXYZ_2.png": png(Image.new("RGB", (300, 300), (40, 120, 200)))}
    r = upload(make_zip(files)).json()
    assert r["total"] == 2 and r["unsupported"] == 1 and r["failed"] >= 1


def test_bad_datasets_are_rejected():
    assert upload(b"not a zip").status_code == 400
    assert upload(make_zip({"readme.txt": b"no images"})).status_code == 400


def test_api_key_is_enforced_when_configured(monkeypatch):
    monkeypatch.setattr(config, "API_KEY", "s3cret-key")
    assert client.get("/api/stats").status_code == 401
    assert client.get("/api/stats", headers={"X-API-Key": "wrong"}).status_code == 401
    assert client.post("/api/batch", json={"count": 1}).status_code == 401
    assert client.get("/api/stats", headers={"X-API-Key": "s3cret-key"}).status_code == 200
    # The page itself and the public settings stay reachable.
    assert client.get("/").status_code == 200
    assert client.get("/api/config").status_code == 200
    assert client.get("/api/health").json()["auth_required"] is True
    assert "s3cret-key" not in client.get("/api/health").text


def test_api_is_open_without_a_key():
    assert config.API_KEY == "" and client.get("/api/stats").status_code == 200


def test_public_config_has_no_secrets():
    text = client.get("/api/config").text
    for p in verification.config.PROVIDERS.values():
        assert p.secret not in text
        for secret in p.test_scenarios.values():
            assert secret not in text


def test_provider_detection():
    r = client.post("/api/detect-provider",
                    json={"html": '<div class="cf-turnstile" data-sitekey="x"></div>'}).json()
    assert [p["id"] for p in r["providers"]] == ["turnstile"]
    assert r["providers"][0]["evidence"] == "signature"
    assert not client.post("/api/detect-provider", json={"html": "<p>hi</p>"}).json()["recognized"]


def test_provider_detection_falls_back_to_keywords():
    # A shop plugin that loads the widget later leaves only the provider's name.
    r = verification.detect_provider('<script>require(["Magento_ReCaptchaFrontendUi"])</script>')
    assert [(p["id"], p["evidence"]) for p in r["providers"]] == [("recaptcha", "keyword")]
    # A real signature wins over a mere mention of another provider.
    r = verification.detect_provider('<div class="h-captcha"></div><!-- recaptcha disabled -->')
    assert [p["id"] for p in r["providers"]] == ["hcaptcha"]


# ---------- token verification (provider API mocked) ----------

def mock_client(body=None, exc=None, seen=None):
    def handler(request):
        if exc:
            raise exc
        if seen is not None:
            seen.append(request.content.decode())
        return httpx.Response(200, json=body)
    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


def run(provider, token, scenario=None, **kw):
    return asyncio.run(verification.verify(provider, token, scenario=scenario, client=mock_client(**kw)))


def test_valid_token():
    r = run("recaptcha", "tok-valid-1", body={"success": True, "hostname": "localhost"})
    assert r["success"] and r["status"] == "verified"


def test_missing_token_never_calls_provider():
    assert run("turnstile", "", exc=AssertionError("should not be called"))["status"] == "missing_token"
    assert run("turnstile", None, exc=AssertionError("should not be called"))["status"] == "missing_token"


def test_invalid_token():
    r = run("hcaptcha", "tok-bad", body={"success": False, "error-codes": ["invalid-input-response"]})
    assert not r["success"] and r["status"] == "invalid_token" and r["decided_by"] == "provider"


def test_malformed_token_is_rejected_without_calling_provider():
    r = run("recaptcha", "not a real token! <b>", exc=AssertionError("should not be called"))
    assert r["status"] == "invalid_token" and r["decided_by"] == "local"
    r = run("recaptcha", "x" * 5000, exc=AssertionError("should not be called"))
    assert r["status"] == "invalid_token"


def test_test_scenario_uses_the_providers_failing_test_secret():
    seen = []
    r = asyncio.run(verification.verify(
        "turnstile", "tok-scenario-1", scenario="always_fail",
        client=mock_client(body={"success": False, "error-codes": ["invalid-input-response"]}, seen=seen)))
    assert r["status"] == "invalid_token" and r["test_scenario"] == "always_fail"
    assert "secret=2x0000000000000000000000000000000AA" in seen[0]
    r = asyncio.run(verification.verify(
        "turnstile", "tok-scenario-2", scenario="already_spent",
        client=mock_client(body={"success": False, "error-codes": ["timeout-or-duplicate"]}, seen=seen)))
    assert r["status"] == "expired_or_reused_token"
    assert "secret=3x0000000000000000000000000000000AA" in seen[1]


def test_test_scenario_is_refused_where_it_does_not_apply(monkeypatch):
    never = AssertionError("should not be called")
    assert run("recaptcha", "tok-s3", scenario="always_fail", exc=never)["status"] == "scenario_unavailable"
    assert run("turnstile", "tok-s4", scenario="made_up", exc=never)["status"] == "scenario_unavailable"
    real = config.Provider(**{**config.PROVIDERS["turnstile"].__dict__, "uses_test_keys": False})
    monkeypatch.setitem(config.PROVIDERS, "turnstile", real)
    assert run("turnstile", "tok-s5", scenario="always_fail", exc=never)["status"] == "scenario_unavailable"


def test_expired_token():
    r = run("hcaptcha", "tok-old", body={"success": False, "error-codes": ["expired-input-response"]})
    assert r["status"] == "expired_token"
    r = run("recaptcha", "tok-old-2", body={"success": False, "error-codes": ["timeout-or-duplicate"]})
    assert r["status"] == "expired_or_reused_token"


def test_reused_token_blocked_locally():
    assert run("turnstile", "tok-once", body={"success": True})["success"]
    r = run("turnstile", "tok-once", exc=AssertionError("replay must not reach provider"))
    assert r["status"] == "reused_token" and not r["success"]


def test_provider_timeout_and_unknown_provider():
    assert run("recaptcha", "tok-slow", exc=httpx.ReadTimeout("slow"))["status"] == "provider_timeout"
    assert run("nope", "tok")["status"] == "unknown_provider"


def test_token_and_secret_never_logged_or_returned(caplog):
    token = "super-sensitive-token-value-123"
    with caplog.at_level(logging.DEBUG):
        r = run("recaptcha", token, body={"success": True})
    secret = verification.config.PROVIDERS["recaptcha"].secret
    assert token not in caplog.text and secret not in caplog.text
    assert token not in str(r) and secret not in str(r)
    with caplog.at_level(logging.DEBUG):
        r = run("turnstile", "another-sensitive-token-456", scenario="always_fail",
                body={"success": False, "error-codes": ["invalid-input-response"]})
    spare = verification.config.PROVIDERS["turnstile"].test_scenarios["always_fail"]
    assert "another-sensitive-token-456" not in caplog.text + str(r)
    assert spare not in caplog.text + str(r)


def test_verify_endpoint_missing_token():
    r = client.post("/api/verify", json={"provider": "recaptcha"}).json()
    assert r["status"] == "missing_token"
