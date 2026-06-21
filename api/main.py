"""FastAPI app entrypoint.

Pre-loads LLMLingua-2 at startup (via lifespan) so the model is never loaded
per request. Serves the vanilla HTML frontend at GET / and mounts the REST +
WebSocket routers.
"""

from __future__ import annotations

import logging
import os

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from agent.compressor import get_compressor

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "info").upper(),
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# Absolute path to the frontend so it resolves regardless of CWD.
_FRONTEND = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "frontend",
    "index.html",
)

# Set to True once the model is loaded successfully; gates the /health probe.
MODEL_READY = False


@asynccontextmanager
async def lifespan(app: FastAPI):
    global MODEL_READY
    logger.info("Pre-loading LLMLingua-2...")
    try:
        get_compressor()
        MODEL_READY = True
        logger.info("LLMLingua-2 ready.")
    except Exception as e:
        MODEL_READY = False
        logger.error(f"Failed to load LLMLingua-2: {e}")
    yield


app = FastAPI(
    title="Prompt Compression Agent",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

from api.routes import router  # noqa: E402
from api.websocket import ws_router  # noqa: E402

app.include_router(router)
app.include_router(ws_router)


@app.get("/")
async def serve_frontend():
    return FileResponse(_FRONTEND)
