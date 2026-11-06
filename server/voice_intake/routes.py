"""HTTP endpoints for the voice demo page."""

from __future__ import annotations

import os
from pathlib import Path

import httpx
from fastapi import APIRouter, HTTPException
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel

from .. import config  # noqa: F401  (loads .env so OPENAI_API_KEY is available)
from . import session as S
from .agent import handle_turn

router = APIRouter()
PAGE = Path(__file__).parent / "static" / "voice.html"


class TurnIn(BaseModel):
    session_id: str
    text: str


class FailureIn(BaseModel):
    session_id: str
    on: bool


def _get(session_id: str) -> S.CallState:
    state = S.get_session(session_id)
    if not state:
        raise HTTPException(404, "unknown session")
    return state


@router.get("/voice", response_class=HTMLResponse)
async def voice_page() -> str:
    return PAGE.read_text(encoding="utf-8")


@router.post("/api/v1/voice/session")
async def start_call() -> dict:
    state = S.new_session()
    return {"session_id": state.session_id, "reply": S.GREETING, "state": state.public()}


@router.post("/api/v1/voice/turn")
async def caller_turn(body: TurnIn) -> dict:
    return await handle_turn(_get(body.session_id), body.text)


@router.post("/api/v1/voice/simulate_failure")
async def simulate_failure(body: FailureIn) -> dict:
    state = _get(body.session_id)
    state.simulate_tool_failure = body.on
    return {"ok": True, "state": state.public()}


# ---------------------------------------------------------------- natural voice (OpenAI text-to-speech)
TTS_VOICE = os.getenv("VOICE_TTS_VOICE", "coral")
TTS_MODEL = os.getenv("VOICE_TTS_MODEL", "gpt-4o-mini-tts")
TTS_STYLE = ("Warm, friendly medical receptionist on a phone call. Brisk, efficient, natural pace, like a real "
             "person who does this all day. No long pauses. Calm and serious when talking about emergencies.")


@router.get("/api/v1/voice/tts")
async def tts(text: str):
    """Streams speech audio so playback starts before the whole file is generated (lower latency)."""
    key = os.getenv("OPENAI_API_KEY", "").strip()
    if not key:
        raise HTTPException(503, "OPENAI_API_KEY not set; the page falls back to the browser voice")
    payload = {"model": TTS_MODEL, "voice": TTS_VOICE, "input": text[:1000], "response_format": "mp3"}
    if "gpt-4o" in TTS_MODEL:
        payload["instructions"] = TTS_STYLE

    client = httpx.AsyncClient(timeout=30)
    req = client.build_request("POST", "https://api.openai.com/v1/audio/speech",
                               headers={"Authorization": f"Bearer {key}"}, json=payload)
    resp = await client.send(req, stream=True)
    if resp.status_code != 200:
        detail = (await resp.aread())[:300]
        await resp.aclose(); await client.aclose()
        raise HTTPException(502, f"TTS failed: {detail!r}")

    async def body():
        try:
            async for chunk in resp.aiter_bytes():
                yield chunk
        finally:
            await resp.aclose(); await client.aclose()

    return StreamingResponse(body(), media_type="audio/mpeg")
