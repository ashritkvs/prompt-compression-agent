# Prompt Compression Agent

Compresses verbose LLM prompts using LLMLingua-2 (local, zero API cost)
and verifies meaning preservation with an OpenAI model (one API call per run).

## Architecture

```
User → FastAPI → Agent Loop → LLMLingua-2 (local compression)
                            → OpenAI (gpt-4o-mini, verification)
                → WebSocket → Browser UI (streaming trace)
```

The agent (OpenAI `gpt-4o-mini` by default) drives a fixed tool sequence:

1. `analyze_prompt` — token/word counts, filler detection, redundancy estimate
2. `compress_with_llmlingua` — local compression at a target retention ratio
3. `verify_meaning` — the OpenAI model scores 0–100 how well intent is preserved
4. `check_token_reduction` — exact tiktoken token savings

Retry rules adapt the compression ratio based on the verification score and
realized reduction, up to 4 compression attempts.

## Setup

```bash
git clone https://github.com/yourname/prompt-compression-agent
cd prompt-compression-agent
pip install -r requirements.txt
cp .env.example .env
# Add your OPENAI_API_KEY to .env (optionally set OPENAI_MODEL)
uvicorn api.main:app --reload
# Open http://localhost:8000
```

The first start downloads and loads the LLMLingua-2 model (a few hundred MB);
it is loaded once at startup via the FastAPI lifespan and reused for every
request.

## Deploy to Render

1. Push the repo to GitHub.
2. New Web Service on Render → connect the repo.
3. Runtime: Docker.
4. Add env var: `OPENAI_API_KEY` (and optionally `OPENAI_MODEL`).
5. Deploy. The health check at `/health` returns 503 until the model finishes
   loading, then 200.

## Run benchmark

```bash
python benchmark/run_benchmark.py
```

Runs 20 fixed prompts (verbose → concise), prints a table + summary, and saves
`benchmark/benchmark_results.json`.

## Run tests

```bash
pytest tests/
```

Network-dependent tests (real OpenAI calls) are skipped automatically
when `OPENAI_API_KEY` is not set.

## API

Docs: http://localhost:8000/docs

### POST /compress

```bash
curl -X POST http://localhost:8000/compress \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Could you please help me understand what ML is?"}'
```

### GET /health

```json
{
  "status": "ok",
  "version": "1.0.0",
  "compression_model": "llmlingua-2",
  "verification_model": "gpt-4o-mini"
}
```

### GET /metrics

Aggregate stats across all runs (in-memory): average reduction, meaning score,
steps, tokens saved, and success rate.

### WebSocket /ws/compress

```
ws://localhost:8000/ws/compress
Send: {"prompt": "your verbose prompt"}
```

Streams `{"type": "step", ...}` messages as each tool executes, then a final
`{"type": "complete", "result": ...}` payload.

## Models

- **Compression:** `microsoft/llmlingua-2-bert-base-multilingual-cased-meetingbank` (local CPU)
- **Agent + verification:** `gpt-4o-mini` by default (OpenAI API); override with `OPENAI_MODEL`
