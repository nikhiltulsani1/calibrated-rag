from dataclasses import fields

from fastapi import APIRouter, Form, Request
from fastapi.responses import RedirectResponse

from src.app.deps import templates
from src.app.session import CREDENTIALS_COOKIE_NAME, encrypt_credentials, is_secure_request, read_credentials
from src.platform.credentials import Credentials

router = APIRouter()

# Display order/labels for the settings form — matches the fields on
# Credentials exactly (see src/platform/credentials.py), one input per
# provider this codebase actually calls out to.
_PROVIDER_LABELS = {
    "groq": "Groq",
    "openrouter": "OpenRouter",
    "nvidia": "NVIDIA NIM",
    "mistral": "Mistral",
    "jina": "Jina AI",
    "cohere": "Cohere",
}


def _mask(key: str | None) -> str | None:
    if not key:
        return None
    if len(key) <= 8:
        return "•" * len(key)
    return key[:4] + "•" * 8 + key[-4:]


@router.get("/settings")
def settings_page(request: Request):
    creds = read_credentials(request.cookies.get(CREDENTIALS_COOKIE_NAME))
    providers = [
        {"key": name, "label": label, "masked": _mask(getattr(creds, name))}
        for name, label in _PROVIDER_LABELS.items()
    ]
    return templates.TemplateResponse(
        request, "settings.html", {"active": "settings", "providers": providers, "saved": False}
    )


@router.post("/settings")
def settings_save(
    request: Request,
    groq: str = Form(""),
    openrouter: str = Form(""),
    nvidia: str = Form(""),
    mistral: str = Form(""),
    jina: str = Form(""),
    cohere: str = Form(""),
):
    # A blank field means "keep whatever's already saved for this
    # provider," not "clear it" — re-displaying every key on every save
    # (masked or not) would be real friction for a 6-field form the
    # visitor is meant to fill in once. Explicit clearing is the
    # trash-icon action below, not an accidental empty submit.
    existing = read_credentials(request.cookies.get(CREDENTIALS_COOKIE_NAME))
    submitted = {"groq": groq, "openrouter": openrouter, "nvidia": nvidia, "mistral": mistral, "jina": jina, "cohere": cohere}
    merged = {name: (submitted[name].strip() or getattr(existing, name)) for name in submitted}
    new_creds = Credentials(**merged)

    encrypted = encrypt_credentials(new_creds)
    response = RedirectResponse(url="/settings", status_code=303)
    if encrypted is not None:
        response.set_cookie(
            CREDENTIALS_COOKIE_NAME,
            encrypted,
            httponly=True,
            secure=is_secure_request(request),
            samesite="lax",
            max_age=60 * 60 * 24 * 365,
        )
    return response


@router.post("/settings/clear")
def settings_clear(request: Request, provider: str = Form(...)):
    valid_names = {f.name for f in fields(Credentials)}
    if provider not in valid_names:
        return RedirectResponse(url="/settings", status_code=303)

    existing = read_credentials(request.cookies.get(CREDENTIALS_COOKIE_NAME))
    updated = {f.name: (None if f.name == provider else getattr(existing, f.name)) for f in fields(Credentials)}
    new_creds = Credentials(**updated)

    encrypted = encrypt_credentials(new_creds)
    response = RedirectResponse(url="/settings", status_code=303)
    if encrypted is not None:
        response.set_cookie(
            CREDENTIALS_COOKIE_NAME,
            encrypted,
            httponly=True,
            secure=is_secure_request(request),
            samesite="lax",
            max_age=60 * 60 * 24 * 365,
        )
    return response
