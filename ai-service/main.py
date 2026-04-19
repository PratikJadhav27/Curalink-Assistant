"""
Curalink AI Service - Main FastAPI Application
Handles: Research retrieval, ranking, LLM synthesis
"""
import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

from routers import research
from core.embeddings import EmbeddingService

load_dotenv()

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

# Startup: pre-load embedding model
@app.on_event("startup")
async def startup_event():
    print("[INIT] Loading embedding model...")
    EmbeddingService.get_instance()
    print("[INIT] Embedding model loaded")

app.include_router(research.router)

@app.get("/health")
async def health():
    return {"status": "ok", "service": "curalink-ai-service"}
