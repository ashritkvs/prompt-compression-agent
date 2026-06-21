"""Agent tools: schemas + async executors + dispatcher.

Each tool is defined as:
  (a) an OpenAI function-calling schema dict (collected in TOOLS)
  (b) an async Python executor function

`execute_tool` routes a tool name + inputs to the right executor and always
returns a string (dict returns are JSON-serialized).
"""

from __future__ import annotations

import json
import logging
import os

import tiktoken
from dotenv import load_dotenv
from openai import AsyncOpenAI

from agent.compressor import compress

# Load variables from a local .env file (no-op if the file is absent).
load_dotenv()

logger = logging.getLogger(__name__)

# Shared encoder — cl100k_base matches modern GPT/Claude tokenization closely.
_enc = tiktoken.get_encoding("cl100k_base")

# Model used for both the agent loop and the verification call.
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

# OpenAI client for the verification call.
_client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

FILLERS = [
    "please",
    "could you",
    "maybe",
    "possibly",
    "if possible",
    "very",
    "actually",
    "just",
    "i was wondering",
    "absolutely",
    "always remember",
    "never forget",
    "comprehensive",
    "detailed and",
    "i would like",
    "kindly",
    "feel free",
    "at your earliest convenience",
    "as much as possible",
    "i just wanted to",
    "do not hesitate",
]


# --------------------------------------------------------------------------- #
# Tool schemas (OpenAI function-calling format)
# --------------------------------------------------------------------------- #

ANALYZE_SCHEMA = {
    "type": "function",
    "function": {
        "name": "analyze_prompt",
        "description": (
            "Analyze a prompt for verbosity: token count, word count, filler "
            "phrases, and an estimated redundancy percentage. Call this first."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "text": {
                    "type": "string",
                    "description": "The prompt text to analyze.",
                }
            },
            "required": ["text"],
        },
    },
}

COMPRESS_SCHEMA = {
    "type": "function",
    "function": {
        "name": "compress_with_llmlingua",
        "description": (
            "Compress a prompt locally with LLMLingua-2 at a target retention "
            "ratio (0.0-1.0). Lower ratio = more aggressive compression. Zero "
            "API cost."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "text": {
                    "type": "string",
                    "description": "The original prompt text to compress.",
                },
                "target_ratio": {
                    "type": "number",
                    "description": "Fraction of tokens to retain, e.g. 0.5.",
                },
            },
            "required": ["text", "target_ratio"],
        },
    },
}

VERIFY_SCHEMA = {
    "type": "function",
    "function": {
        "name": "verify_meaning",
        "description": (
            "Judge how well the compressed prompt preserves the full intent "
            "of the original. Returns a 0-100 score, reasoning, a pass flag, "
            "and any lost concepts."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "original": {
                    "type": "string",
                    "description": "The original prompt.",
                },
                "compressed": {
                    "type": "string",
                    "description": "The compressed prompt.",
                },
            },
            "required": ["original", "compressed"],
        },
    },
}

CHECK_SCHEMA = {
    "type": "function",
    "function": {
        "name": "check_token_reduction",
        "description": (
            "Compute exact token reduction between original and compressed "
            "prompts using tiktoken cl100k_base."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "original": {
                    "type": "string",
                    "description": "The original prompt.",
                },
                "compressed": {
                    "type": "string",
                    "description": "The compressed prompt.",
                },
            },
            "required": ["original", "compressed"],
        },
    },
}

TOOLS = [ANALYZE_SCHEMA, COMPRESS_SCHEMA, VERIFY_SCHEMA, CHECK_SCHEMA]


# --------------------------------------------------------------------------- #
# Executors
# --------------------------------------------------------------------------- #

async def execute_analyze(text: str) -> str:
    """Count tokens/words, scan for fillers, estimate redundancy."""
    tokens = len(_enc.encode(text))
    words = len(text.split())
    lower = text.lower()
    found_fillers = [f for f in FILLERS if f in lower]
    redundancy = min(70, len(found_fillers) * 8 + 15)
    return (
        f"tokens={tokens} | words={words} | "
        f"fillers={found_fillers} | redundancy≈{redundancy}%"
    )


async def execute_compress(text: str, target_ratio: float) -> str:
    """Compress locally via LLMLingua-2 (no API cost)."""
    return compress(text, target_ratio)


async def execute_verify(original: str, compressed: str) -> dict:
    """Ask the OpenAI model to score semantic preservation."""
    system = (
        "You are a semantic similarity judge. Compare original and "
        "compressed prompts. Score 0-100 how well compressed preserves "
        "the full intent. Be strict — penalize lost specifics.\n"
        "Respond in raw JSON only, no markdown backticks:\n"
        '{"score": int, "reasoning": str, "pass": bool, '
        '"lost_concepts": [str]}\n'
        "pass is true if score >= 75."
    )
    try:
        response = await _client.chat.completions.create(
            model=OPENAI_MODEL,
            max_tokens=1024,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": system},
                {
                    "role": "user",
                    "content": f"Original: {original}\n\nCompressed: {compressed}",
                },
            ],
        )
        text_out = (response.choices[0].message.content or "").strip()
        parsed = json.loads(text_out)
        return {
            "score": int(parsed.get("score", 0)),
            "reasoning": parsed.get("reasoning", ""),
            "pass": bool(parsed.get("pass", False)),
            "lost_concepts": parsed.get("lost_concepts", []),
        }
    except Exception as e:
        logger.error(f"verify_meaning failed: {e}")
        return {
            "score": 0,
            "pass": False,
            "reasoning": "parse error",
            "lost_concepts": [],
        }


async def execute_check(original: str, compressed: str) -> dict:
    """Exact tiktoken-based token reduction stats."""
    orig_tokens = len(_enc.encode(original))
    comp_tokens = len(_enc.encode(compressed))
    saved = orig_tokens - comp_tokens
    reduction_pct = (
        round((saved / orig_tokens) * 100, 1) if orig_tokens > 0 else 0.0
    )
    return {
        "original_tokens": orig_tokens,
        "compressed_tokens": comp_tokens,
        "saved": saved,
        "reduction_pct": reduction_pct,
    }


# --------------------------------------------------------------------------- #
# Dispatcher
# --------------------------------------------------------------------------- #

async def execute_tool(name: str, inputs: dict) -> str:
    """Route a tool call to its executor and return a string result."""
    if name == "analyze_prompt":
        return await execute_analyze(inputs["text"])
    if name == "compress_with_llmlingua":
        return await execute_compress(inputs["text"], inputs["target_ratio"])
    if name == "verify_meaning":
        result = await execute_verify(inputs["original"], inputs["compressed"])
        return json.dumps(result)
    if name == "check_token_reduction":
        result = await execute_check(inputs["original"], inputs["compressed"])
        return json.dumps(result)
    raise ValueError(f"Unknown tool: {name}")
