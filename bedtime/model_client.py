import json
from urllib import error, request

import openai

from bedtime.config import OLLAMA_HOST, OLLAMA_MODEL, OPENAI_API_KEY, OPENAI_MODEL, USE_LOCAL_MODEL


def call_model(prompt: str, max_tokens: int = 3000, temperature: float = 0.1) -> str:
    """Call the assignment's required OpenAI model."""
    if not OPENAI_API_KEY:
        raise RuntimeError(
            "OPENAI_API_KEY is missing. Add it to .env or run with USE_LOCAL_MODEL=true for Ollama development."
        )

    if hasattr(openai, "OpenAI"):
        client = openai.OpenAI(api_key=OPENAI_API_KEY)
        resp = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=temperature,
        )
        return resp.choices[0].message.content or ""

    openai.api_key = OPENAI_API_KEY
    resp = openai.ChatCompletion.create(
        model=OPENAI_MODEL,
        messages=[{"role": "user", "content": prompt}],
        stream=False,
        max_tokens=max_tokens,
        temperature=temperature,
    )
    return resp.choices[0].message["content"]  # type: ignore[index]


def call_local_model(prompt: str, max_tokens: int = 3000, temperature: float = 0.1) -> str:
    """Call local Ollama for development."""
    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": temperature,
            "num_predict": max_tokens,
        },
    }
    data = json.dumps(payload).encode("utf-8")
    req = request.Request(
        f"{OLLAMA_HOST.rstrip('/')}/api/generate",
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with request.urlopen(req, timeout=180) as resp:
            body = json.loads(resp.read().decode("utf-8"))
    except error.URLError as exc:
        raise RuntimeError(
            f"Could not reach Ollama at {OLLAMA_HOST}. Start Ollama and ensure the {OLLAMA_MODEL!r} model is available."
        ) from exc

    response_text = body.get("response", "")
    if not response_text:
        raise RuntimeError(f"Ollama returned an empty response: {body}")
    return response_text


def should_use_local_model() -> bool:
    return USE_LOCAL_MODEL


def call_llm(prompt: str, max_tokens: int = 3000, temperature: float = 0.1) -> str:
    if should_use_local_model():
        return call_local_model(prompt, max_tokens=max_tokens, temperature=temperature)
    return call_model(prompt, max_tokens=max_tokens, temperature=temperature)


def current_provider() -> str:
    return "ollama" if should_use_local_model() else "openai"


def current_model() -> str:
    return OLLAMA_MODEL if should_use_local_model() else OPENAI_MODEL
