"""HTTP endpoints for the voice demo page."""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

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
