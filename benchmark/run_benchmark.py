"""Benchmark the compression agent over 20 fixed prompts.

Runs each prompt synchronously, prints a fixed-width table + summary block,
and saves full results to benchmark/benchmark_results.json.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import time
from datetime import datetime, timezone

# Allow running as `python benchmark/run_benchmark.py` from the repo root.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent.agent import run_agent  # noqa: E402

PROMPTS: list[str] = [
    # ---- 5 extremely verbose (100+ words, heavy filler) ----
    "I was wondering if you could please possibly help me understand, in a "
    "very detailed and comprehensive way, what machine learning actually is, "
    "including all of the different types and subtypes if at all possible, "
    "and please always remember to never forget to explain the underlying "
    "mathematics as well, because I would like a truly thorough explanation "
    "that covers absolutely everything, and feel free to add as much detail "
    "as possible, and kindly do not hesitate to include examples, because I "
    "just wanted to make sure I really actually understand the whole topic "
    "completely and thoroughly from start to finish in great depth.",
    "Could you please, at your earliest convenience, write me a very "
    "comprehensive and detailed and thorough essay about the history of the "
    "Roman Empire, and I would like you to please always remember to cover "
    "absolutely every single important event if possible, and kindly feel "
    "free to include as much detail as possible about the emperors, and "
    "please do not hesitate to actually explain the political and economic "
    "factors as well, because I just wanted to truly understand the complete "
    "and comprehensive picture of how everything actually unfolded over time.",
    "I would like to kindly request, if it is at all possible and not too "
    "much trouble, that you please write a very detailed and comprehensive "
    "and thorough step by step guide explaining how to actually bake a "
    "perfect sourdough bread loaf from scratch, and please always remember "
    "to never forget to include all of the little tips and tricks, and feel "
    "free to add as much detail as possible about the fermentation process, "
    "because I just wanted to make absolutely sure that I understand every "
    "single step completely and thoroughly without missing anything at all.",
    "Please could you possibly maybe help me by actually writing a very "
    "comprehensive, detailed and thorough analysis of the causes of climate "
    "change, and I would like you to kindly always remember to include all "
    "of the scientific evidence if possible, and please do not hesitate to "
    "feel free to explain the greenhouse effect in as much detail as "
    "possible, because I just wanted to truly and actually understand the "
    "complete and comprehensive scientific picture of exactly why the global "
    "climate is changing so rapidly in the modern era of today.",
    "I was just wondering if you could please possibly take the time to "
    "actually write me a very detailed and comprehensive and thorough "
    "explanation of how the human immune system works, and please always "
    "remember to never forget to cover both the innate and adaptive immune "
    "responses if possible, and kindly feel free to add as much detail as "
    "possible about white blood cells, because I just wanted to make sure "
    "that I really and truly understand absolutely everything about this "
    "fascinating and complex biological topic in great and thorough depth.",
    # ---- 5 moderately verbose (50-80 words) ----
    "Could you please help me understand how neural networks learn, and I "
    "would like you to actually explain backpropagation in a fairly detailed "
    "way if possible, because I just wanted to make sure I really understand "
    "the gradient descent part and how the weights are actually updated "
    "during training over many epochs of the learning process.",
    "I would like you to please write a fairly comprehensive summary of the "
    "main events of World War II, and kindly try to cover both the European "
    "and Pacific theaters if possible, because I actually just wanted to get "
    "a reasonably thorough overview of how the whole conflict unfolded from "
    "the beginning to the very end.",
    "Please could you actually explain to me how a relational database index "
    "works, and I would like you to maybe cover B-trees if possible, because "
    "I just wanted to really understand why indexes make queries faster and "
    "what the actual tradeoffs are when you add too many of them to a table.",
    "I was wondering if you could possibly help me understand the difference "
    "between TCP and UDP protocols, and please feel free to explain when you "
    "would actually want to use each one, because I just wanted to get a "
    "fairly clear and practical understanding of how they really differ in "
    "real world networking situations.",
    "Could you kindly explain how photosynthesis actually works at a "
    "reasonably detailed level, and I would like you to cover both the light "
    "dependent and light independent reactions if possible, because I just "
    "wanted to genuinely understand how plants convert sunlight into usable "
    "chemical energy for growth.",
    # ---- 5 slightly verbose (20-40 words) ----
    "Could you please explain what a REST API is and how it actually works, "
    "because I just wanted to understand the basic concepts behind it.",
    "I would like you to summarize the plot of Hamlet for me, and please try "
    "to keep it fairly concise if possible.",
    "Please help me understand the difference between a list and a tuple in "
    "Python, because I just wanted a clear explanation.",
    "Could you actually explain how compound interest works, and maybe give "
    "me a simple example if possible?",
    "I was wondering if you could explain what Docker containers are and why "
    "people actually use them in software development.",
    # ---- 5 control (already concise) ----
    "Explain the difference between HTTP and HTTPS.",
    "Summarize the theory of relativity in two sentences.",
    "List the primary colors and how to mix them.",
    "Write a Python function that reverses a string.",
    "Define recursion and give one example.",
]


async def _run_one(prompt: str) -> dict:
    start = time.perf_counter()
    result = await run_agent(prompt)
    elapsed_ms = round((time.perf_counter() - start) * 1000)
    return {
        "preview": prompt[:40],
        "original_tokens": result.original_tokens,
        "compressed_tokens": result.compressed_tokens,
        "tokens_saved": result.tokens_saved,
        "reduction_pct": result.reduction_pct,
        "meaning_score": result.meaning_score,
        "total_steps": result.total_steps,
        "success": result.success,
        "time_taken_ms": elapsed_ms,
    }


def _pad(value, width: int) -> str:
    s = str(value)
    if len(s) > width:
        s = s[: width - 1] + "…"
    return s.ljust(width)


async def _run_all() -> list[dict]:
    """Run every prompt sequentially within a single event loop.

    The OpenAI async client opens a connection pool bound to the running
    event loop; calling asyncio.run() per prompt would close that loop and
    leave the pool unusable on the next prompt. One loop for all prompts
    avoids that.
    """
    results: list[dict] = []
    for i, prompt in enumerate(PROMPTS, start=1):
        print(f"Running {i}/{len(PROMPTS)}...", flush=True)
        results.append(await _run_one(prompt))
    return results


def main():
    results: list[dict] = asyncio.run(_run_all())

    # ---- table ----
    header = (
        _pad("#", 3)
        + _pad("PREVIEW", 42)
        + _pad("ORIG", 6)
        + _pad("COMP", 6)
        + _pad("SAVED", 7)
        + _pad("RED%", 7)
        + _pad("SCORE", 7)
        + _pad("STEPS", 7)
        + _pad("MS", 8)
    )
    print()
    print(header)
    print("-" * len(header))
    for i, r in enumerate(results, start=1):
        print(
            _pad(i, 3)
            + _pad(r["preview"], 42)
            + _pad(r["original_tokens"], 6)
            + _pad(r["compressed_tokens"], 6)
            + _pad(r["tokens_saved"], 7)
            + _pad(r["reduction_pct"], 7)
            + _pad(r["meaning_score"], 7)
            + _pad(r["total_steps"], 7)
            + _pad(r["time_taken_ms"], 8)
        )

    # ---- summary ----
    n = len(results)
    avg_reduction = round(sum(r["reduction_pct"] for r in results) / n, 1)
    avg_score = round(sum(r["meaning_score"] for r in results) / n, 1)
    avg_steps = round(sum(r["total_steps"] for r in results) / n, 1)
    success_rate = round(
        sum(1 for r in results if r["success"]) / n * 100, 1
    )
    avg_time = round(sum(r["time_taken_ms"] for r in results) / n)

    print()
    print("═══ BENCHMARK SUMMARY ═══")
    print(f"Total prompts:      {n}")
    print(f"Avg reduction:      {avg_reduction}%")
    print(f"Avg meaning score:  {avg_score}/100")
    print(f"Avg steps:          {avg_steps}")
    print(f"Success rate:       {success_rate}%")
    print(f"Avg time/run:       {avg_time}ms")
    print("═════════════════════════")

    # ---- save ----
    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "benchmark_results.json")
    payload = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "summary": {
            "total_prompts": n,
            "avg_reduction_pct": avg_reduction,
            "avg_meaning_score": avg_score,
            "avg_steps": avg_steps,
            "success_rate": success_rate,
            "avg_time_ms": avg_time,
        },
        "results": results,
    }
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
