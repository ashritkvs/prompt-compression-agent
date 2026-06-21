"""WebSocket route that streams each agent StepTrace as it executes."""

from __future__ import annotations

import asyncio
from dataclasses import asdict

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from agent.agent import StepTrace, run_agent
from agent.metrics import metrics_store

ws_router = APIRouter()


@ws_router.websocket("/ws/compress")
async def websocket_compress(websocket: WebSocket):
    await websocket.accept()
    try:
        data = await websocket.receive_json()
        prompt = data.get("prompt", "").strip()

        if len(prompt) < 10 or len(prompt) > 3000:
            await websocket.send_json(
                {
                    "type": "error",
                    "message": "Prompt must be 10–3000 characters",
                }
            )
            return

        async def on_step(trace: StepTrace):
            await websocket.send_json({"type": "step", "data": asdict(trace)})
            await asyncio.sleep(0)

        result = await run_agent(prompt, on_step=on_step)
        metrics_store.record(result)

        await websocket.send_json(
            {"type": "complete", "result": asdict(result)}
        )

    except WebSocketDisconnect:
        pass
    except Exception as e:
        try:
            await websocket.send_json({"type": "error", "message": str(e)})
        except Exception:
            pass
