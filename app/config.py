"""Configuration, loaded from environment variables (.env supported).

Defaults are the providers' published TEST keys, which only work in test mode
and never protect a real site. Put your own keys in .env for an owned site.
"""
import os
from dataclasses import dataclass, field

from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class Provider:
    name: str
    site_key: str
    secret: str
    verify_url: str
    script_url: str
    widget_class: str
    response_field: str
    # True while the provider's published test secret is in use.
    uses_test_keys: bool = True
    # Extra official test secrets that force a provider-side outcome
    # (scenario name -> secret). Only offered while test keys are in use.
    test_scenarios: dict = field(default_factory=dict)


def _provider(prefix: str, test_site_key: str, test_secret: str, **kw) -> Provider:
    secret = os.getenv(f"{prefix}_SECRET") or test_secret
    return Provider(site_key=os.getenv(f"{prefix}_SITE_KEY") or test_site_key,
                    secret=secret, uses_test_keys=secret == test_secret, **kw)


PROVIDERS = {
    "recaptcha": _provider(
        "RECAPTCHA", "6LeIxAcTAAAAAJcZVRqyHh71UMIEGNQ_MXjiZKhI",
        "6LeIxAcTAAAAAGG-vFI1TnRWxMZNFuojJ4WifJWe",
        name="Google reCAPTCHA v2",
        verify_url="https://www.google.com/recaptcha/api/siteverify",
        script_url="https://www.google.com/recaptcha/api.js",
        widget_class="g-recaptcha",
        response_field="g-recaptcha-response",
    ),
    "hcaptcha": _provider(
        "HCAPTCHA", "10000000-ffff-ffff-ffff-000000000001",
        "0x0000000000000000000000000000000000000000",
        name="hCaptcha",
        verify_url="https://api.hcaptcha.com/siteverify",
        script_url="https://js.hcaptcha.com/1/api.js",
        widget_class="h-captcha",
        response_field="h-captcha-response",
    ),
    "turnstile": _provider(
        "TURNSTILE", "1x00000000000000000000AA", "1x0000000000000000000000000000000AA",
        name="Cloudflare Turnstile",
        verify_url="https://challenges.cloudflare.com/turnstile/v0/siteverify",
        script_url="https://challenges.cloudflare.com/turnstile/v0/api.js",
        widget_class="cf-turnstile",
        response_field="cf-turnstile-response",
        # Cloudflare's published dummy secrets: "always fails" and "token already spent".
        test_scenarios={
            "always_fail": "2x0000000000000000000000000000000AA",
            "already_spent": "3x0000000000000000000000000000000AA",
        },
    ),
}

VERIFY_TIMEOUT_SECONDS = float(os.getenv("VERIFY_TIMEOUT_SECONDS") or 8)
DB_PATH = os.getenv("DB_PATH") or "data/attempts.db"
# Optional. When set, every /api/* call except /api/health and /api/config
# must send it in the X-API-Key header.
API_KEY = (os.getenv("API_KEY") or "").strip()

MAX_UPLOAD_BYTES = 2 * 1024 * 1024
MAX_DATASET_BYTES = 20 * 1024 * 1024
MAX_DATASET_ITEMS = 2000
