from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import httpx
import yaml

from src.platform.credentials import get_credentials
from src.platform.telemetry import get_tracer

_CONFIG_PATH = Path(__file__).resolve().parents[1] / "config" / "models.yaml"


@dataclass(frozen=True)
class ModelBinding:
    provider: str
    model_id: str


def _parse(spec: str) -> ModelBinding:
    provider, sep, model_id = spec.partition(":")
    if not sep or not provider or not model_id:
        raise ValueError(f"model spec must be 'provider:model_id', got {spec!r}")
    return ModelBinding(provider=provider, model_id=model_id)


def get_model(role: str) -> ModelBinding:
    """Resolve a role to the model that should serve it — its top rung.
    See get_model_ladder() below for every rung, used by roles that now
    actually implement retry-with-fallback (query_planner.py).

    Env var override (RAG_MODEL_<ROLE>) always wins over models.yaml, per
    the "switching models" rule — no file edit needed to override in CI.
    """
    return get_model_ladder(role)[0]


def get_model_ladder(role: str) -> list[ModelBinding]:
    """Every configured rung for a role, in order. A role bound to a
    single spec (not a list) in models.yaml returns a 1-item list. The
    env var override, when set, replaces the whole ladder with that one
    binding — an override means "always use this," not "try this first."
    """
    env_override = os.environ.get(f"RAG_MODEL_{role.upper()}")
    if env_override:
        return [_parse(env_override)]

    config = yaml.safe_load(_CONFIG_PATH.read_text(encoding="utf-8")) or {}
    roles = config.get("roles", {})
    if role not in roles:
        raise KeyError(f"no binding for role {role!r} in {_CONFIG_PATH}")

    spec = roles[role]
    if isinstance(spec, list):
        return [_parse(s) for s in spec]
    return [_parse(spec)]


def provider_has_key(provider: str) -> bool:
    """Whether the API key a provider needs is actually configured —
    used by ladder-descent-on-retry callers to skip a fallback rung
    that would just fail immediately with a *different*, less useful
    error (a missing key) instead of retrying the real problem.

    Phase 2 (BYOK): must check the visitor's own credentials too, not
    just os.environ — real gap flagged during design and confirmed here:
    without this, usable_ladder() would silently filter out a rung the
    visitor DOES have a key for, just because the server's own
    os.environ doesn't happen to have one configured.
    """
    if provider not in _CHAT_ENDPOINTS:
        return False
    if getattr(get_credentials(), provider, None):
        return True
    _, api_key_env = _CHAT_ENDPOINTS[provider]
    return bool(os.environ.get(api_key_env))


def usable_ladder(role: str) -> list[ModelBinding]:
    """The role's model ladder, filtered to rungs whose API key is
    actually configured — a fallback rung with no key would just fail
    immediately with a *different*, less useful error (missing key)
    instead of retrying the real problem. Falls back to the top rung
    alone if literally nothing is configured (matches prior same-model
    retry behavior; that rung's key must already be present or the
    first call attempt wouldn't have gotten this far).

    Promoted out of query_planner.py (2026-08-21) so any caller doing
    retry-with-fallback shares one implementation — see
    is_retryable_http_error below and reason/nodes/assess.py, which hit
    a real bug from NOT having this: its grade-role calls only ever
    tried rung 0, so a single provider's quota exhaustion (real,
    observed: Groq's grade-role rung hit 429 mid-eval-run) silently
    disabled the entire ambiguity/sufficiency guardrail for that
    request, even though a working fallback rung was configured and
    sitting unused in models.yaml the whole time.
    """
    ladder = get_model_ladder(role)
    usable = [binding for binding in ladder if provider_has_key(binding.provider)]
    return usable or ladder[:1]


def is_retryable_http_error(exc: httpx.HTTPStatusError) -> bool:
    """429/5xx are always worth retrying — standard transient conditions.
    A plain 400 usually means OUR request is malformed and retrying is
    pointless, but Groq has one specific, verified-live exception:
    `json_validate_failed` with an empty `failed_generation` — the model
    itself failed to produce anything, not a bad request on our end.
    Reproduced live (2026-08-16) on a real query: 4 of 5 identical calls
    hit this exact code. Checked by code, not by blanket-retrying every
    400, so a genuine malformed-request bug in this codebase still fails
    loud immediately instead of being masked by three silent retries.

    Promoted out of query_planner.py (2026-08-21) — see usable_ladder's
    docstring for why a second caller needed this.
    """
    status = exc.response.status_code
    if status in (429, 500, 502, 503, 504):
        return True
    if status == 400:
        try:
            return exc.response.json().get("error", {}).get("code") == "json_validate_failed"
        except Exception:
            return False
    return False


def is_retryable_error(exc: httpx.HTTPError) -> bool:
    """Real bug found live 2026-08-24: both `plan_query`'s retry loop and
    `assess.py`'s `_complete_with_fallback` only ever caught
    `httpx.HTTPStatusError` — but `httpx.ConnectError`/`ReadTimeout`/
    `ConnectTimeout`/`RemoteProtocolError` are `httpx.TransportError`, a
    SIBLING of `HTTPStatusError` under `httpx.HTTPError`, not a subclass
    of it. `plan_query`'s own comment records real evidence this matters:
    "in one real 80-question run, 10 of 14 failures on this stage were
    plain network connection resets... not bad model output at all" — the
    exact failure class the narrow except silently let through uncaught
    instead of retrying/degrading.

    A bare transport-level failure carries no `.response` to inspect (no
    status code, no body), so `is_retryable_http_error`'s status-code
    logic doesn't apply — and isn't needed: a connection reset/timeout is
    always worth retrying (no code=malformed-request signal is even
    possible without a response), unlike a genuine 400 that might mean
    our own request is broken. `HTTPStatusError` still routes through the
    existing, narrower, tested logic unchanged.
    """
    if isinstance(exc, httpx.HTTPStatusError):
        return is_retryable_http_error(exc)
    return True


# provider -> (chat-completions URL, env var holding its API key). Only
# providers actually bound to a built role are listed — see the embedder's
# equivalent honest-gap pattern in A0. All four are genuinely
# OpenAI-compatible (verified against each provider's own docs, not
# assumed from "most providers are").
#
# mistral added 2026-08-20 for a real 3-way model comparison run (Groq /
# OpenRouter / Mistral direct) — confirmed OpenAI-compatible via Mistral's
# own docs (base URL https://api.mistral.ai/v1, same chat/completions
# shape). La Plateforme's free "Experiment" tier (~1B tokens/month,
# rate-limited, covers all models including their largest) is what makes
# this a real free-tier option, not a paid-only one like every Mistral
# listing on OpenRouter turned out to be.
_CHAT_ENDPOINTS: dict[str, tuple[str, str]] = {
    "groq": ("https://api.groq.com/openai/v1/chat/completions", "GROQ_API_KEY"),
    "openrouter": ("https://openrouter.ai/api/v1/chat/completions", "OPENROUTER_API_KEY"),
    "nvidia": ("https://integrate.api.nvidia.com/v1/chat/completions", "NVIDIA_API_KEY"),
    "mistral": ("https://api.mistral.ai/v1/chat/completions", "MISTRAL_API_KEY"),
}


@dataclass(frozen=True)
class CompletionResult:
    provider: str
    model_served: str
    content: str


def complete(
    role: str,
    messages: list[dict],
    *,
    temperature: float = 0.0,
    json_mode: bool = False,
    model_override: ModelBinding | None = None,
) -> CompletionResult:
    """Resolve `role` to a model and make the actual chat-completion call.

    No retry-on-failure ladder descent happens *inside* this function —
    that's the caller's job (query_planner.py's plan_query is the first
    real one, using get_model_ladder()/provider_has_key() above plus
    `model_override` here to retry a failed call on a genuinely
    different model rather than re-asking the one that just failed).
    This function itself still just makes one call and raises on failure.

    `model_served` on the result is the model the provider actually used
    to serve the request (echoed back in its response) — the trace
    attribute the plan requires so silent substitution stays visible.
    """
    tracer = get_tracer()
    # No manual try/except here: start_as_current_span already records an
    # uncaught exception as a span event and sets ERROR status on exit by
    # default. An earlier version of this duplicated that manually and
    # every span ended up with the exception recorded twice — caught by
    # the telemetry unit test asserting exactly one event, not assumed.
    with tracer.start_as_current_span("llm.complete") as span:
        span.set_attribute("llm.role", role)
        span.set_attribute("llm.json_mode", json_mode)

        binding = model_override or get_model(role)
        span.set_attribute("llm.provider", binding.provider)
        span.set_attribute("llm.model_requested", binding.model_id)

        if binding.provider not in _CHAT_ENDPOINTS:
            raise NotImplementedError(
                f"no chat-completion client wired up for provider {binding.provider!r} "
                f"(role {role!r}) — only {sorted(_CHAT_ENDPOINTS)} are implemented so far"
            )
        url, api_key_env = _CHAT_ENDPOINTS[binding.provider]
        # Phase 2 (BYOK): the visitor's own key (from the request-scoped
        # Credentials contextvar) wins when present, falling back to the
        # server's own os.environ value — a deployment that never sets
        # BYOK credentials sees zero behavior change. See
        # src/platform/credentials.py's docstring for why this is a
        # contextvar and not a second os.environ write.
        api_key = getattr(get_credentials(), binding.provider, None) or os.environ.get(api_key_env)
        if not api_key:
            raise RuntimeError(f"{api_key_env} is not set — required for role {role!r}")

        body: dict = {
            "model": binding.model_id,
            "messages": messages,
            "temperature": temperature,
        }
        if json_mode:
            body["response_format"] = {"type": "json_object"}

        response = httpx.post(
            url,
            json=body,
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=30.0,
        )
        response.raise_for_status()
        payload = response.json()

        result = CompletionResult(
            provider=binding.provider,
            model_served=payload.get("model", binding.model_id),
            content=payload["choices"][0]["message"]["content"],
        )
        # model_served is the trace attribute the plan requires so a
        # silent ladder-descent substitution stays visible after the
        # fact rather than only inferred from a quality drop.
        span.set_attribute("llm.model_served", result.model_served)
        return result
