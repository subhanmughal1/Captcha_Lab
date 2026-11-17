"""Test provider recognition against a CSV of public pages with known providers.

For every row the page is downloaded once, exactly as a browser would open it,
and the HTML is passed to verification.detect_provider. The result is compared
with the provider named in the CSV.

Read-only: no form is filled or submitted, no CAPTCHA widget is loaded or
solved, and no token is requested. One GET request per page, no retries.

Usage:  python scripts/detect_from_csv.py <pages.csv> [report.csv]

The CSV needs the columns "Contact Page" and "Captcha" (reCAPTCHA,
invisible reCAPTCHA, hCaptcha, Turnstile or none).
"""
import asyncio
import csv
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import httpx

from app import verification

USER_AGENT = "CaptchaLab-ProviderDetection/1.0 (read-only page check)"
CONCURRENCY = 8
TIMEOUT_SECONDS = 20
MAX_PAGE_BYTES = 3 * 1024 * 1024

EXPECTED = {"recaptcha": "recaptcha", "invisible recaptcha": "recaptcha",
            "hcaptcha": "hcaptcha", "turnstile": "turnstile", "none": "none"}


async def fetch(client: httpx.AsyncClient, url: str) -> tuple[str | None, str]:
    """Return (html, note). html is None when the page could not be read."""
    try:
        async with client.stream("GET", url) as resp:
            if resp.status_code != 200:
                return None, f"HTTP {resp.status_code}"
            chunks, size = [], 0
            async for chunk in resp.aiter_bytes():
                chunks.append(chunk)
                size += len(chunk)
                if size > MAX_PAGE_BYTES:
                    break
            return b"".join(chunks).decode(resp.encoding or "utf-8", errors="replace"), ""
    except httpx.TimeoutException:
        return None, "timeout"
    except httpx.HTTPError as exc:
        return None, type(exc).__name__


async def check(rows: list[dict]) -> list[dict]:
    gate = asyncio.Semaphore(CONCURRENCY)
    async with httpx.AsyncClient(timeout=TIMEOUT_SECONDS, follow_redirects=True, max_redirects=5,
                                 headers={"User-Agent": USER_AGENT, "Accept": "text/html"}) as client:
        async def one(row: dict) -> dict:
            expected = EXPECTED.get(row["Captcha"].strip().lower(), "unknown")
            async with gate:
                html, note = await fetch(client, row["Contact Page"].strip())
            if html is None:
                detected, outcome = "", "not_fetched"
            else:
                providers = verification.detect_provider(html)["providers"]
                found = [p["id"] for p in providers]
                detected = "+".join(found) or "none"
                note = providers[0]["evidence"] if providers else ""
                if found == [expected] or (expected == "none" and not found):
                    outcome = "match"
                elif expected in found:
                    outcome = "match_among_several"  # right provider, but others were named too
                else:
                    outcome = "mismatch"
            return {"Company": row["Company"], "Contact Page": row["Contact Page"],
                    "Expected": expected, "Detected": detected, "Outcome": outcome, "Note": note}
        return await asyncio.gather(*(one(r) for r in rows))


def main() -> None:
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    source = Path(sys.argv[1])
    report = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("data/provider_detection_report.csv")
    with source.open(encoding="utf-8-sig", newline="") as fh:
        rows = list(csv.DictReader(fh))
    print(f"checking {len(rows)} pages ...")
    results = asyncio.run(check(rows))

    report.parent.mkdir(parents=True, exist_ok=True)
    with report.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)

    fetched = [r for r in results if r["Outcome"] != "not_fetched"]
    matches = sum(r["Outcome"] == "match" for r in fetched)
    several = sum(r["Outcome"] == "match_among_several" for r in fetched)
    print(f"pages read     : {len(fetched)} of {len(results)}")
    print("not read       :", dict(Counter(r["Note"] for r in results if r["Outcome"] == "not_fetched")))
    if fetched:
        print(f"provider match : {matches} of {len(fetched)} ({100 * matches / len(fetched):.1f}%)")
        print(f"right provider named together with others: {several}")
    print("by expected provider (match / pages read):")
    for name in ("recaptcha", "hcaptcha", "turnstile", "none"):
        group = [r for r in fetched if r["Expected"] == name]
        if group:
            hit = sum(r["Outcome"] == "match" for r in group)
            print(f"  {name:10s} {hit:3d} / {len(group):3d}")
    print("mismatches (expected -> detected):",
          dict(Counter(f'{r["Expected"]} -> {r["Detected"]}' for r in fetched if r["Outcome"] == "mismatch")))
    print("report saved to", report)


if __name__ == "__main__":
    main()
