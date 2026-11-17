"""Server-side verification of CAPTCHA tokens with the official provider APIs.

This is the site-owner side of the flow: the browser widget issues a token,
our backend sends it with the secret key to the provider's siteverify endpoint.
Secrets and tokens are never written to logs or returned to the client.
"""
from __future__ import annotations

import hashlib
import logging
import re
import time

import httpx

from . import config

log = logging.getLogger("verification")

# Provider error code -> our structured status.
ERROR_MAP = {
    "missing-input-response": "missing_token",
    "invalid-input-response": "invalid_token",
    "timeout-or-duplicate": "expired_or_reused_token",
    "expired-input-response": "expired_token",
    "already-seen-response": "reused_token",
    "missing-input-secret": "server_misconfigured",
    "invalid-input-secret": "server_misconfigured",
    "sitekey-secret-mismatch": "server_misconfigured",
    "bad-request": "invalid_token",
    "internal-error": "provider_error",
}

MESSAGES = {
    "verified": "Token is valid.",
    "missing_token": "No token was sent. Complete the CAPTCHA first.",
    "invalid_token": "The provider rejected this token as malformed or not issued by it.",
    "expired_token": "The token has expired. Solve the CAPTCHA again.",
    "reused_token": "This token was already verified once. Tokens are single-use.",
    "expired_or_reused_token": "The token is expired or was already used. Solve the CAPTCHA again.",
    "unknown_provider": "Unsupported provider.",
    "provider_timeout": "The provider did not answer in time. Try again.",
    "provider_error": "The provider could not process the request.",
    "server_misconfigured": "Server keys are misconfigured. Check the environment variables.",
    "rejected": "The provider rejected the token.",
    "scenario_unavailable": "This test scenario needs a provider that publishes a dedicated "
                            "test secret for it (Cloudflare Turnstile) and the default test keys.",
}

# Characters that appear in real provider tokens. Anything else (spaces, quotes,
# markup) cannot be a token, so it is rejected before a provider call is spent.
TOKEN_FORMAT = re.compile(r"[A-Za-z0-9._\-+/=:~]+")

_seen: dict[str, float] = {}  # sha256(token) -> first-seen time (replay guard)
SEEN_TTL = 600


def fingerprint(token: str) -> str:
    """Short, non-reversible id that is safe to log instead of the token."""
    return hashlib.sha256(token.encode()).hexdigest()[:12]


def _result(provider: str, status: str, started: float, decided_by: str = "local", **extra) -> dict:
    """decided_by: "provider" when the answer came from siteverify, "local"
    when this server rejected the request before calling the provider."""
    return {
        "provider": provider,
        "success": status == "verified",
        "status": status,
        "message": MESSAGES.get(status, status),
        "decided_by": decided_by,
        "processing_ms": round((time.perf_counter() - started) * 1000, 2),
        **extra,
    }


def _remember(token: str) -> bool:
    """Return True if this token was already seen (replay)."""
    now = time.time()
    for key in [k for k, t in _seen.items() if now - t > SEEN_TTL]:
        del _seen[key]
    key = hashlib.sha256(token.encode()).hexdigest()
    if key in _seen:
        return True
    _seen[key] = now
    return False


async def verify(provider_id: str, token: str | None, remote_ip: str | None = None,
                 client: httpx.AsyncClient | None = None, scenario: str | None = None) -> dict:
    """Validate a token with the provider.

    scenario: optional name of an official provider test secret that forces a
    provider-side outcome ("always_fail", "already_spent"). It exists so failure
    handling can be demonstrated with test keys, which otherwise accept any
    token. It is refused as soon as real keys are configured.
    """
    started = time.perf_counter()
    provider = config.PROVIDERS.get(provider_id)
    if provider is None:
        return _result(provider_id, "unknown_provider", started)

    secret = provider.secret
    if scenario:
        secret = provider.test_scenarios.get(scenario) if provider.uses_test_keys else None
        if secret is None:
            return _result(provider_id, "scenario_unavailable", started)

    token = (token or "").strip()
    if not token:
        return _result(provider_id, "missing_token", started)
    if len(token) > 4096 or not TOKEN_FORMAT.fullmatch(token):
        log.info("verify provider=%s status=invalid_token (local format check)", provider_id)
        return _result(provider_id, "invalid_token", started,
                       message="The token is malformed, so it was rejected before calling the provider.")

    fp = fingerprint(token)
    if _remember(token):
        log.info("verify provider=%s token=%s status=reused_token (local replay guard)", provider_id, fp)
        return _result(provider_id, "reused_token", started, token_fingerprint=fp)

    payload = {"secret": secret, "response": token}
    if remote_ip:
        payload["remoteip"] = remote_ip

    own_client = client is None
    client = client or httpx.AsyncClient(timeout=config.VERIFY_TIMEOUT_SECONDS)
    try:
        resp = await client.post(provider.verify_url, data=payload)
        resp.raise_for_status()
        body = resp.json()
    except httpx.TimeoutException:
        log.warning("verify provider=%s token=%s status=provider_timeout", provider_id, fp)
        return _result(provider_id, "provider_timeout", started, token_fingerprint=fp)
    except (httpx.HTTPError, ValueError) as exc:
        log.warning("verify provider=%s token=%s status=provider_error (%s)",
                    provider_id, fp, type(exc).__name__)
        return _result(provider_id, "provider_error", started, token_fingerprint=fp)
    finally:
        if own_client:
            await client.aclose()

    codes = body.get("error-codes") or []
    if body.get("success") is True:
        status = "verified"
    else:
        status = next((ERROR_MAP[c] for c in codes if c in ERROR_MAP), "rejected")

    log.info("verify provider=%s token=%s status=%s", provider_id, fp, status)
    return _result(
        provider_id, status, started, decided_by="provider",
        token_fingerprint=fp,
        test_keys=provider.uses_test_keys,
        test_scenario=scenario,
        provider_error_codes=codes,
        hostname=body.get("hostname"),
        challenge_ts=body.get("challenge_ts"),
    )


# Markup that only appears when the provider is integrated: its script URL,
# widget class or JavaScript object.
STRONG_SIGNATURES = {
    "recaptcha": ["google.com/recaptcha", "recaptcha.net/recaptcha", "gstatic.com/recaptcha",
                  "g-recaptcha", "grecaptcha"],
    "hcaptcha": ["hcaptcha.com", "h-captcha"],
    "turnstile": ["challenges.cloudflare.com/turnstile", "cf-turnstile"],
}
# The provider's name alone, as left by shop and form plugins that load the
# widget later with JavaScript. Used only when no strong signature is found.
KEYWORD_SIGNATURES = {"recaptcha": ["recaptcha"], "hcaptcha": ["hcaptcha"], "turnstile": ["turnstile"]}


def detect_provider(html: str) -> dict:
    """Identify which CAPTCHA provider a page's HTML integrates. Read-only: it
    inspects markup that was pasted in or downloaded, and never loads a widget.

    evidence is "signature" for a script URL, widget class or JavaScript
    object, and "keyword" when only the provider's name appears in the page.
    """
    lowered = html.lower()

    def scan(signatures: dict) -> dict:
        found = {pid: [s for s in sigs if s in lowered] for pid, sigs in signatures.items()}
        return {pid: hits for pid, hits in found.items() if hits}

    found, evidence = scan(STRONG_SIGNATURES), "signature"
    if not found:
        found, evidence = scan(KEYWORD_SIGNATURES), "keyword"
    return {
        "recognized": bool(found),
        "providers": [{"id": pid, "name": config.PROVIDERS[pid].name, "matched": hits,
                       "evidence": evidence}
                      for pid, hits in found.items()],
    }
