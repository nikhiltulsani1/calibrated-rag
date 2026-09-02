def friendly_error_message(exc: Exception) -> str:
    """Maps the known, honest-gap exception shapes from src/platform and
    src/index (missing API key, unsupported provider) to a specific
    message — "the reranker timed out" beats "something went wrong" per
    the plan's UI principles. Falls back to a generic-but-still-scoped
    message for anything unrecognized, never a raw traceback.
    """
    text = str(exc)
    if isinstance(exc, RuntimeError) and "is not set" in text:
        return f"This step needs an API key that isn't configured on this server yet ({text})."
    if isinstance(exc, NotImplementedError):
        return f"This step hit an unsupported configuration ({text})."
    return f"Something went wrong while processing this request ({type(exc).__name__}: {text})."
