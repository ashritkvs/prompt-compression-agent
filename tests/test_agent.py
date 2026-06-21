"""Test suite for the Prompt Compression Agent.

Network-dependent tests (real OpenAI calls, model load) require
OPENAI_API_KEY to be set; they are skipped automatically otherwise.
"""

from __future__ import annotations

import asyncio
import os

import pytest
import tiktoken
from httpx import ASGITransport, AsyncClient

from agent.compressor import compress, get_compressor
from agent.tools import execute_analyze, execute_verify

HAS_KEY = bool(os.getenv("OPENAI_API_KEY"))
needs_key = pytest.mark.skipif(not HAS_KEY, reason="OPENAI_API_KEY not set")


# --------------------------------------------------------------------------- #
# Pure / local tests
# --------------------------------------------------------------------------- #

def test_token_counting():
    enc = tiktoken.get_encoding("cl100k_base")
    assert len(enc.encode("hello world")) == 2


def test_analyze_filler_detection():
    out = asyncio.run(
        execute_analyze("Could you please possibly help me if possible")
    )
    assert "please" in out
    assert "possibly" in out


def test_compressor_singleton():
    a = get_compressor()
    b = get_compressor()
    assert a is b


def test_compression_reduces_tokens():
    enc = tiktoken.get_encoding("cl100k_base")
    prompt = (
        "I was wondering if you could please possibly help me understand in a "
        "very detailed and comprehensive way what machine learning actually "
        "is, including all of the different types and subtypes if at all "
        "possible, and please always remember to never forget the math too."
    )
    compressed = compress(prompt, 0.5)
    assert len(enc.encode(compressed)) < len(enc.encode(prompt))


# --------------------------------------------------------------------------- #
# Network-dependent tests (OpenAI)
# --------------------------------------------------------------------------- #

@needs_key
def test_verify_meaning_identical():
    text = "Explain how neural networks learn using backpropagation."
    result = asyncio.run(execute_verify(text, text))
    assert result["score"] >= 85


@needs_key
def test_agent_full_run():
    from agent.agent import run_agent

    prompt = (
        "Could you please possibly help me understand in a very detailed and "
        "comprehensive way what machine learning actually is, including all "
        "the different types and subtypes if possible?"
    )
    result = asyncio.run(run_agent(prompt))
    assert result.success is True
    assert result.reduction_pct > 15
    assert result.meaning_score >= 70
    assert result.total_steps >= 3


# --------------------------------------------------------------------------- #
# API tests
# --------------------------------------------------------------------------- #

def _make_client():
    from api.main import app

    return AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    )


@needs_key
def test_api_health():
    async def _run():
        # Ensure the model is loaded so /health reports ready.
        import api.main as main_mod

        get_compressor()
        main_mod.MODEL_READY = True
        async with _make_client() as client:
            resp = await client.get("/health")
            assert resp.status_code == 200
            assert resp.json()["status"] == "ok"

    asyncio.run(_run())


@needs_key
def test_api_compress():
    async def _run():
        async with _make_client() as client:
            resp = await client.post(
                "/compress",
                json={
                    "prompt": "Could you please help me understand what "
                    "machine learning actually is in detail?"
                },
            )
            assert resp.status_code == 200
            data = resp.json()
            assert data["compressed_prompt"]
            assert data["reduction_pct"] >= 0

    asyncio.run(_run())


def test_api_validation():
    async def _run():
        async with _make_client() as client:
            resp = await client.post("/compress", json={"prompt": "abc"})
            assert resp.status_code == 422

    asyncio.run(_run())
