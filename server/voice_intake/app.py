"""Standalone app for the voice demo (no Gmail / Composio needed).

Run from the repo root:
    python -m uvicorn server.voice_intake.app:app --port 8001
Then open http://localhost:8001/voice in Chrome.
"""

from fastapi import FastAPI

from .routes import router

app = FastAPI(title="Maple Family Clinic voice intake")
app.include_router(router)
