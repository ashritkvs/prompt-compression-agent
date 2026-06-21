"""REST routes: /compress, /health, /metrics."""

from __future__ import annotations

import logging
from dataclasses import asdict

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, field_validator

from agent.agent import run_agent
from agent.metrics import metrics_store

logger = logging.getLogger(__name__)
router = APIRouter()


class CompressRequest(BaseModel):
    prompt: str

    @field_validator("prompt")
    @classmethod
    def validate_prompt(cls, v: str) -> str:
        if len(v.strip()) < 10:
            raise ValueError("Prompt too short — minimum 10 characters")
        if len(v) > 3000:
            raise ValueError("Prompt too long — maximum 3000 characters")
        return v.strip()


@router.post("/compress")
async def compress_endpoint(request: CompressRequest):
    try:
        result = await run_agent(request.prompt)
        metrics_store.record(result)
        return asdict(result)
    except Exception as e:
        logger.error(f"/compress failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/health")
async def health():
    # Import lazily to read the live readiness flag without a circular import.
    from api.main import MODEL_READY
    from agent.tools import OPENAI_MODEL

    if not MODEL_READY:
        raise HTTPException(
            status_code=503,
            detail="Compression model not loaded",
        )
    return {
        "status": "ok",
        "version": "1.0.0",
        "compression_model": "llmlingua-2",
        "verification_model": OPENAI_MODEL,
    }


@router.get("/metrics")
async def metrics():
    return metrics_store.summary()
