"""OpenAI-compatible proxy for the in-workspace AI agent.

The frontend's Next.js route runs the Vercel AI SDK (tool loop, streaming) but
points its provider `baseURL` here. So the *actual* provider traffic flows
through this service, which:

  1. Authorizes the request by the user's own forwarded `token` cookie,
     resolved via backend-auth (no service secret required from the caller).
  2. Checks the user's remaining credits with backend-auth. None left → 402
     (the panel shows the quota banner).
  3. Forwards the chat-completion to the real upstream (OpenAI for gpt-*,
     Google's OpenAI-compatible endpoint for gemini-*), streaming the SSE back
     verbatim. An upstream failure (incl. the server's *own* quota) → 502, which
     the panel shows as a plain Internal error.
  4. Settles credits after the stream by the real token usage: 1 credit = $0.01
     ($1 = 100 credits), priced per model.

Provider keys live in `key/openai.key` and `key/google.key` (same convention as
`key/groq.key`). `SERVICE_SECRET` and `AUTH_URL` come from the environment.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import httpx
from fastapi import APIRouter, Cookie, HTTPException, Request, status
from fastapi.responses import Response, StreamingResponse

router = APIRouter(prefix="/agent", tags=["agent"])

_KEY_DIR = Path(__file__).resolve().parent.parent / "key"


def _read_key(name: str) -> str:
    p = _KEY_DIR / name
    return p.read_text(encoding="utf-8").strip() if p.exists() else ""


OPENAI_KEY = _read_key("openai.key")
GOOGLE_KEY = _read_key("google.key")
# Used ONLY for the backend→backend-auth /credits/minus call (which requires
# X-Service-Secret). The inbound /agent request is authorized by the user's own
# token, not this secret. From the environment or a key/service.key file (same
# convention as groq.key); must match backend-auth's SERVICE_SECRET.
SERVICE_SECRET = os.environ.get("SERVICE_SECRET") or _read_key("service.key")
# 127.0.0.1 (not "localhost") so the server-side HTTP client uses IPv4 directly —
# uvicorn binds 0.0.0.0 (IPv4) and "localhost" can resolve to ::1 (IPv6) → refused.
AUTH_URL = (os.environ.get("AUTH_URL") or "http://127.0.0.1:7001").rstrip("/")
# backend-ai (Ollama proxy) for local gemma* models, authenticated by the shared
# service secret (X-Service-Secret).
AI_URL = (os.environ.get("AI_URL") or "http://127.0.0.1:7002").rstrip("/")

OPENAI_BASE = "https://api.openai.com/v1"
GOOGLE_BASE = "https://generativelanguage.googleapis.com/v1beta/openai"

# Dollars per 1M tokens (input, output). Credits: 1 credit = $0.01 → $1 = 100.
# gemma* run on local Ollama → free (0 cost).
PRICING: dict[str, tuple[float, float]] = {
    "gpt-4o": (2.50, 10.00),
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-4.1": (2.00, 8.00),
    "gpt-4.1-mini": (0.40, 1.60),
    "gemini-2.5-pro": (1.25, 10.00),
    "gemini-2.5-flash": (0.30, 2.50),
    "gemini-2.0-flash": (0.10, 0.40),
    "gemma4:e4b": (0.0, 0.0),
}
_DEFAULT_PRICE = (2.50, 10.00)


def _upstream(model: str) -> tuple[str, dict[str, str] | None]:
    """(base_url, auth_headers); auth_headers is None when the key/secret is missing."""
    if model.startswith("gemma"):  # local Ollama, proxied via backend-ai
        return f"{AI_URL}/ollama/v1", ({"X-Service-Secret": SERVICE_SECRET} if SERVICE_SECRET else None)
    if model.startswith("gemini"):
        return GOOGLE_BASE, ({"Authorization": f"Bearer {GOOGLE_KEY}"} if GOOGLE_KEY else None)
    return OPENAI_BASE, ({"Authorization": f"Bearer {OPENAI_KEY}"} if OPENAI_KEY else None)


def _credits_for(model: str, usage: dict) -> int:
    """Real-cost credits for a call. 1 credit = $0.01, rounded to the nearest."""
    pin, pout = PRICING.get(model, _DEFAULT_PRICE)
    inp = usage.get("prompt_tokens") or 0
    out = usage.get("completion_tokens") or 0
    dollars = (inp * pin + out * pout) / 1_000_000
    return round(dollars * 100)


async def _get_credits(token: str) -> int:
    async with httpx.AsyncClient(timeout=10) as c:
        r = await c.get(f"{AUTH_URL}/credits/me", cookies={"token": token})
    if r.status_code == 401:
        raise HTTPException(status_code=401, detail="Not authenticated")
    if r.status_code != 200:
        raise HTTPException(status_code=502, detail="Auth service error")
    return int(r.json().get("credits", 0))


async def _minus_credits(token: str, amount: int) -> None:
    if amount <= 0:
        return
    # Best-effort settle — never fail the already-streamed response over this.
    try:
        async with httpx.AsyncClient(timeout=10) as c:
            cookies = {"token": token}
            headers = {"X-Service-Secret": SERVICE_SECRET}
            r = await c.post(f"{AUTH_URL}/credits/minus", cookies=cookies, headers=headers, json={"amount": amount})
            if r.status_code == 402:
                # The turn cost more than the balance — drain whatever remains.
                me = await c.get(f"{AUTH_URL}/credits/me", cookies=cookies)
                left = int(me.json().get("credits", 0)) if me.status_code == 200 else 0
                if left > 0:
                    await c.post(f"{AUTH_URL}/credits/minus", cookies=cookies, headers=headers, json={"amount": left})
    except Exception:
        pass


def _scan_usage(text: str, usage: dict) -> None:
    """Pull an OpenAI `usage` object out of complete SSE data lines."""
    for line in text.split("\n"):
        line = line.strip()
        if not line.startswith("data:"):
            continue
        payload = line[5:].strip()
        if not payload or payload == "[DONE]":
            continue
        try:
            obj = json.loads(payload)
        except json.JSONDecodeError:
            continue
        u = obj.get("usage")
        if isinstance(u, dict):
            usage.update(u)


@router.post("/v1/chat/completions")
async def chat_completions(
    request: Request,
    token: str | None = Cookie(default=None),
):
    # Authorize by the user's own credential: their forwarded auth cookie.
    if not token:
        raise HTTPException(status_code=401, detail="Not authenticated")

    payload = await request.json()
    model = str(payload.get("model", ""))
    base, auth_headers = _upstream(model)
    if auth_headers is None:
        raise HTTPException(status_code=500, detail="Provider key not configured")

    # _get_credits validates the token (401 if invalid) and returns the balance.
    # Paid models require a positive balance; local (0-cost) models only need a
    # valid login.
    pin, pout = PRICING.get(model, _DEFAULT_PRICE)
    credits = await _get_credits(token)
    if (pin > 0 or pout > 0) and credits <= 0:
        raise HTTPException(status_code=status.HTTP_402_PAYMENT_REQUIRED, detail="Insufficient credits")

    if payload.get("stream"):
        opts = dict(payload.get("stream_options") or {})
        opts["include_usage"] = True
        payload["stream_options"] = opts

    client = httpx.AsyncClient(timeout=None)
    upstream = await client.send(
        client.build_request(
            "POST",
            f"{base}/chat/completions",
            headers={**auth_headers, "Content-Type": "application/json", "Accept-Encoding": "identity"},
            content=json.dumps(payload),
        ),
        stream=True,
    )

    if upstream.status_code != 200:
        body = (await upstream.aread()).decode("utf-8", "replace")
        await upstream.aclose()
        await client.aclose()
        # The server's own provider failed (incl. its quota) → Internal error.
        raise HTTPException(status_code=502, detail=f"Upstream provider error ({upstream.status_code}): {body[:500]}")

    if not payload.get("stream"):
        data = await upstream.aread()
        await upstream.aclose()
        await client.aclose()
        usage: dict = {}
        try:
            usage = json.loads(data).get("usage") or {}
        except Exception:
            usage = {}
        await _minus_credits(token, _credits_for(model, usage))
        return Response(content=data, media_type="application/json")

    async def gen():
        usage: dict = {}
        buf = ""
        try:
            async for chunk in upstream.aiter_bytes():
                yield chunk
                buf += chunk.decode("utf-8", "replace")
                if "\n" in buf:
                    head, _, buf = buf.rpartition("\n")
                    _scan_usage(head, usage)
        finally:
            await upstream.aclose()
            await client.aclose()
            await _minus_credits(token, _credits_for(model, usage))

    return StreamingResponse(
        gen(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
