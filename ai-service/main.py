"""
Curalink AI Service - Main FastAPI Application
Handles: Research retrieval, ranking, LLM synthesis
"""
import os
import subprocess
from datetime import datetime, timezone
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

from routers import research
from core.embeddings import EmbeddingService

load_dotenv()


def _get_commit():
    """Get the current git commit hash — embedded at image build time."""
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL,
        ).decode().strip()
    except Exception:
        return os.environ.get("GIT_COMMIT", "unknown")


COMMIT_HASH  = _get_commit()
STARTUP_TIME = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

app = FastAPI(
    title="Curalink AI Service",
    description="AI-powered medical research retrieval and reasoning engine",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def startup_event():
    print(f"[INIT] Curalink AI starting — commit: {COMMIT_HASH}, started: {STARTUP_TIME}")
    print("[INIT] Loading embedding model...")
    EmbeddingService.get_instance()
    print("[INIT] ✅ Embedding model loaded")


app.include_router(research.router)


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "service": "curalink-ai-service",
        "commit": COMMIT_HASH,      # ← exact git hash of running code
        "startedAt": STARTUP_TIME,  # ← when this container last restarted
    }
