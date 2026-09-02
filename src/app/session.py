from __future__ import annotations

import base64
import hashlib
import json
import os
import uuid
from dataclasses import fields

from cryptography.fernet import Fernet, InvalidToken
from itsdangerous import BadSignature, URLSafeSerializer

from src.platform.credentials import Credentials

# Phase 2 (BYOK + private uploads) — two cookies, deliberately different
# in kind:
#
# SESSION_COOKIE_NAME: a random session id, SIGNED (itsdangerous) but
# not secret — its only job is tamper-resistance (a visitor can't forge
# another visitor's session id) and it's the isolation key private
# uploads (stage 6) key off, so it must be stable across requests.
#
# CREDENTIALS_COOKIE_NAME: the visitor's real provider API keys,
# ENCRYPTED (Fernet), not just signed — a signed-but-plaintext cookie is
# base64-readable by anyone who can see it (browser extensions, a
# shared machine, XSS); real secrets must not be recoverable from the
# cookie by inspection, only decryptable with the server's own
# COOKIE_ENCRYPTION_KEY.
SESSION_COOKIE_NAME = "cr_session"
CREDENTIALS_COOKIE_NAME = "cr_keys"
_SESSION_SALT = "cr-session-v1"


def is_secure_request(request) -> bool:
    """Whether a `Secure`-flagged cookie is safe to send back on THIS
    request — real bug found live testing BYOK locally: a Secure cookie
    set while browsing http://localhost is genuinely never re-sent by a
    real browser (confirmed directly: `document.cookie` came back empty
    after a save), disproving the earlier assumption that browsers treat
    localhost as an implicit secure context for cookie purposes. Using
    `request.url.scheme` this way only reports the real original scheme
    once the Dockerfile's uvicorn command trusts Render's
    X-Forwarded-Proto header (see Dockerfile's --proxy-headers comment)
    — without that, this would always read "http" behind Render's
    TLS-terminating proxy too, silently disabling Secure cookies in
    production. Local plain-HTTP dev correctly gets a non-Secure cookie
    instead of BYOK silently failing to persist.
    """
    return request.url.scheme == "https"

_CREDENTIAL_FIELDS = {f.name for f in fields(Credentials)}


def _encryption_secret() -> str:
    return os.environ.get("COOKIE_ENCRYPTION_KEY", "")


def _fernet() -> Fernet | None:
    """None when COOKIE_ENCRYPTION_KEY isn't set — BYOK simply can't
    work without it (nowhere safe to hold keys), but every other feature
    of the app must keep working; callers treat None as "BYOK
    unavailable this request," never raise.

    Accepts ANY string as the secret (not just a pre-formatted Fernet
    key) — Render's `generateValue: true` produces a random string, not
    necessarily Fernet's exact required 32-byte-urlsafe-base64 shape, so
    this derives a valid Fernet key from whatever string is configured
    via SHA-256, rather than requiring the operator to hand-generate a
    correctly-shaped key.
    """
    secret = _encryption_secret()
    if not secret:
        return None
    digest = hashlib.sha256(secret.encode()).digest()
    return Fernet(base64.urlsafe_b64encode(digest))


def _signer() -> URLSafeSerializer:
    # Falls back to a fixed dev-only secret so local `docker compose`
    # usage (no COOKIE_ENCRYPTION_KEY set at all — BYOK/uploads are live-
    # deployment-only features) doesn't need every operator to configure
    # this just to get a session id at all.
    secret = _encryption_secret() or "dev-insecure-session-secret"
    return URLSafeSerializer(secret, salt=_SESSION_SALT)


def get_or_create_session_id(cookie_value: str | None) -> tuple[str, bool]:
    """Verifies the signed cookie if present; mints a fresh random
    session id if missing or tampered with. Returns (session_id,
    is_new) — is_new tells the caller whether to actually set the
    cookie on the response.
    """
    if cookie_value:
        try:
            session_id = _signer().loads(cookie_value)
            if isinstance(session_id, str) and session_id:
                return session_id, False
        except BadSignature:
            pass
    return uuid.uuid4().hex, True


def sign_session_id(session_id: str) -> str:
    return _signer().dumps(session_id)


def read_credentials(cookie_value: str | None) -> Credentials:
    """Decrypts the visitor's credentials cookie, or returns an all-None
    Credentials on any failure (missing cookie, no encryption key
    configured, tampered/corrupt value, or a value encrypted under a
    now-rotated key) — never raises. A visitor who can't be decrypted is
    functionally identical to one who never set any keys.
    """
    fernet = _fernet()
    if not cookie_value or fernet is None:
        return Credentials()
    try:
        decrypted = fernet.decrypt(cookie_value.encode())
        data = json.loads(decrypted)
    except (InvalidToken, ValueError, UnicodeDecodeError):
        return Credentials()
    if not isinstance(data, dict):
        return Credentials()
    return Credentials(**{k: v for k, v in data.items() if k in _CREDENTIAL_FIELDS})


def encrypt_credentials(creds: Credentials) -> str | None:
    """None when no COOKIE_ENCRYPTION_KEY is configured — the caller
    (settings route) must not silently pretend the keys were saved."""
    fernet = _fernet()
    if fernet is None:
        return None
    payload = json.dumps({k: v for k, v in creds.__dict__.items() if v})
    return fernet.encrypt(payload.encode()).decode()
