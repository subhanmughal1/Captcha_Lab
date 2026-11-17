# CAPTCHA Lab

A standalone application that recognises and solves two kinds of synthetic CAPTCHA with
a custom computer-vision engine, and demonstrates the official token-verification flow
for Google reCAPTCHA, hCaptcha and Cloudflare Turnstile using their test keys.

**Scope.** As the assignment requires (section 7), this project only solves CAPTCHAs it
generates itself and only verifies tokens on a locally hosted demo page with the
providers' official test environments. It never solves a CAPTCHA on a third-party
website, harvests tokens, or submits forms. The one optional script that reads other
sites (`scripts/detect_from_csv.py`) only downloads public pages to recognise which
provider they use.

## Features

| Requirement | Where |
|---|---|
| Challenge-type recognition (supported vs unsupported) | `app/solver.py` → `classify_challenge` |
| Two supported challenge types: text strip and odd-one-out shape grid | `app/solver.py`, `app/shape_grid.py` |
| Provider recognition (reCAPTCHA / hCaptcha / Turnstile) | `app/verification.py` → `detect_provider` |
| Custom solving engine with confidence score | OpenCV + scikit-learn MLPs (text), contour geometry (shapes) |
| Fallback when the primary method fails | alternative lengths and contour segmentation (text), geometric outlier (shapes) |
| Official token acquisition + server-side validation | UI "Token verification" tab + `app/verification.py` |
| Invalid / expired / reused / missing tokens | structured statuses in `app/verification.py` |
| No secrets or tokens in logs | only a SHA-256 fingerprint is logged |
| Web interface and reporting | `app/static/index.html`, `app/store.py` (SQLite) |
| Batch processing and accuracy statistics | `POST /api/batch` (generated), `POST /api/batch/upload` (supplied ZIP), `scripts/evaluate.py` |
| Reusable API with optional API key | FastAPI, interactive docs at `/docs`, `X-API-Key` header |

## Setup

Requires Python 3.12 or newer (tested on 3.12.10). `requirements.txt` pins the exact
versions the shipped model was trained with.

```bash
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
uvicorn app.main:app --port 8000
```

Open http://localhost:8000. Run the tests with `python -m pytest -q`.

A trained model ships in `models/char_mlp.joblib`. Retraining is only needed if it fails
to load (for example with a different scikit-learn version): `python scripts/train.py`
takes a few minutes (about 7 on a laptop CPU).

### Environment variables

All are optional. Without a `.env` file the app runs on the providers' test keys. To
change a setting, create a file named `.env` in the project folder with `NAME=value` lines
(for example `API_KEY=a-long-random-value`). `.env` is never committed.

| Variable | Purpose | Default |
|---|---|---|
| `RECAPTCHA_SITE_KEY`, `RECAPTCHA_SECRET` | reCAPTCHA v2 keys | Google's published test keys |
| `HCAPTCHA_SITE_KEY`, `HCAPTCHA_SECRET` | hCaptcha keys | hCaptcha's published test keys |
| `TURNSTILE_SITE_KEY`, `TURNSTILE_SECRET` | Turnstile keys | Cloudflare's "always passes" test keys |
| `API_KEY` | when set, the API requires it in the `X-API-Key` header | empty (API open) |
| `VERIFY_TIMEOUT_SECONDS` | provider request timeout | `8` |
| `DB_PATH` | SQLite file for attempt history | `data/attempts.db` |

Secret keys stay on the server. `/api/config` returns only the public site keys.

## Architecture

```
Browser UI (static HTML/JS)
   │
   ▼
FastAPI (app/main.py)
   ├── generator.py     synthetic challenges (Pillow): text strips and 3×3 shape grids, with noise
   ├── solver.py        decode → clean-up → type check → text pipeline → answer + confidence
   ├── shape_grid.py    grid recognition and odd-one-out solving from contour geometry
   ├── verification.py  provider siteverify call, error mapping, replay guard, log redaction
   └── store.py         SQLite attempt log → dashboard statistics
```

**Type recognition.** Before solving, the engine checks decodability, size, grid lines,
aspect ratio, ink ratio and glyph count. An image is solved only if it is a text strip or
a 3×3 shape grid. Anything else (a photo, a screenshot, an image-selection challenge) is
reported as unsupported instead of being guessed at.

**Text pipeline.** The image is thresholded so that only dark glyph ink remains, a 3×3
closing bridges the 1-px gaps cut by noise lines, and small specks are removed by
component area. A small MLP estimates the number of characters (4 to 6) from the column
ink profile. The strip is then cut at the slot boundaries of that layout, each cut moved
to the nearest column with the least ink. Every character is read by two MLPs (256-128
hidden units): one sees the cropped glyph normalised to 28×28, the other a fixed window
centred on the slot, which keeps a glyph whole when its neighbour overlaps it. Their
probabilities are averaged. Confidence is the mean class probability across characters.
If the length estimate or the confidence is below 0.60, the fallback reads the other
lengths and a connected-component segmentation and returns the best-scoring reading.

**Shape-grid pipeline.** Eight tiles hold the same shape and one holds a different one;
the answer is the number of the odd tile (1 to 9, left to right, top to bottom). Each
tile's largest contour is described by solidity, extent and circularity, which do not
depend on size or rotation, and matched to the nearest of four shapes (circle, square,
triangle, star). If the classes do not single out exactly one tile, the fallback picks
the tile whose geometry is farthest from the grid's median.

**Token flow.** The widget issues a token in the browser; the backend posts it with the
secret key to the provider's `siteverify` endpoint and returns a structured result.
`decided_by` says whether the provider or this server made the decision.

| Status | Meaning |
|---|---|
| `verified` | provider confirmed the token |
| `missing_token` | no token sent (provider is not called) |
| `invalid_token` | provider rejected it, or it is malformed (provider is not called) |
| `expired_token` / `expired_or_reused_token` | provider reported timeout or duplicate |
| `reused_token` | same token submitted twice (blocked by the local replay guard) |
| `rejected` | provider refused the token with an unmapped error code |
| `provider_timeout` / `provider_error` | provider unreachable or failed |
| `server_misconfigured` | missing or wrong secret key |
| `unknown_provider` | provider id is not one of the three |
| `scenario_unavailable` | a test scenario was requested where it does not apply |

**Failure cases with test keys.** Test secrets accept almost anything, so real rejections
need a little care:

| Provider | "Send invalid token" in the UI |
|---|---|
| hCaptcha | a made-up token is sent; hCaptcha rejects it |
| Turnstile | sent with Cloudflare's official "always fails" test secret; Cloudflare rejects it |
| reCAPTCHA | Google's test secret accepts every token, so a malformed token is sent and rejected by this server before Google is called |

Cloudflare's "token already spent" test secret is used the same way for the
"Provider says expired or already used" button. These scenarios (`scenario` field of
`POST /api/verify`) are refused as soon as real keys are configured.

## API

Interactive documentation is at `/docs`.

| Endpoint | Purpose |
|---|---|
| `POST /api/challenge/generate` | create a synthetic challenge (`kind`: text or shape_grid, `length`: 4 to 6) |
| `POST /api/solve` | solve a generated challenge (`challenge_id`) or an uploaded image (`file`) |
| `POST /api/batch` | accuracy over a generated dataset (`count`, `kind`: text, shape_grid or mixed) |
| `POST /api/batch/upload` | accuracy over a supplied dataset (ZIP) |
| `GET /api/dataset/sample` | download a labelled sample dataset in the upload format |
| `POST /api/verify` | validate a provider token |
| `POST /api/detect-provider` | identify the provider in a page snippet |
| `GET /api/stats`, `DELETE /api/stats` | attempt statistics, clear history |
| `GET /api/health`, `GET /api/config` | status and public widget settings (never need a key) |

**API key.** Set `API_KEY` to require the `X-API-Key` header on every endpoint except
`/api/health` and `/api/config`. The web page asks for the key once and keeps it for the
browser tab. Without `API_KEY` the API is open, which suits a local demo.

```bash
curl -X POST http://localhost:8000/api/batch \
  -H "X-API-Key: <your key>" -H "Content-Type: application/json" \
  -d '{"count": 100, "kind": "mixed"}'
```

**Dataset format.** A ZIP of images (PNG, JPG, BMP, GIF or WebP) with a `labels.csv`
holding `filename,label` rows. Without the CSV, the label is the file name up to the first
underscore (`K7MP2_001.png`). The label is the text for a text strip and the tile number
for a shape grid. Limits: 20 MB, 2000 images, 2 MB per image.

## Results

Measured on unseen generated challenges with the shipped model:

| Command | Challenge type | Accuracy |
|---|---|---|
| `python scripts/evaluate.py 2000 text` | text, 4 to 6 characters | 99.45% full-string (1989 of 2000) |
| `python scripts/evaluate.py 1000 shape_grid` | shape grid | 100% (1000 of 1000) |

Processing takes about 4 to 6 ms per challenge on a laptop CPU. Numbers vary slightly per
training run and machine.

### Provider recognition on real pages

`python scripts/detect_from_csv.py <pages.csv>` checks provider recognition against a CSV
of public pages with a known provider (columns `Contact Page` and `Captcha`). Each page is
downloaded once with an honest User-Agent and its HTML is passed to `detect_provider`.
Nothing is submitted and no widget is loaded or solved. A per-page report is written to
`data/provider_detection_report.csv`.

On the supplied list of 200 US e-commerce contact pages:

| Result | Pages |
|---|---|
| Page could be read | 82 of 200 |
| Exactly the listed provider recognised | 72 of 82 (87.8%) |
| Listed provider recognised together with others | 6 of 82 |
| Listed provider missed | 4 of 82 |
| Page not read (108 answered HTTP 403, 10 other errors) | 118 of 200 |

The 118 unread pages refused or failed an automated request. The script does not
disguise itself as a browser to get past that, so those pages are reported as not read
rather than guessed. Recognition works on the downloaded HTML only, so a widget that a
page adds later with JavaScript is found only through keywords left in the markup
(`evidence: "keyword"` instead of `"signature"`).

## Cost breakdown

| Component | Cost |
|---|---|
| Custom solving engine (local CPU) | $0 per challenge |
| reCAPTCHA / hCaptcha / Turnstile test keys and siteverify calls | $0 |
| Third-party solving services | not used |

The dashboard shows estimated cost per challenge from the attempt log, so a paid method
could be added later without changing the reporting.

## Notes and limits

- The solver is trained for this project's own generator. It is not a general CAPTCHA
  solver and is not intended to be.
- hCaptcha's widget may refuse to load on `localhost`; if so, map a hostname such as
  `captcha.test` to `127.0.0.1` in your hosts file and open `http://captcha.test:8000`.
- The replay guard and challenge store are in memory, which suits a single-process demo.
- The web interface is a single static page with no build step. Its fonts are bundled in
  `app/static/fonts` (SIL Open Font License), so it looks the same offline. It has a light
  and a dark theme and works on phone-sized screens.
- The API key is a single shared secret, enough to gate integrations in a demo; it is not
  a user-account system.
