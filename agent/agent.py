"""The compression agent loop.

Drives an OpenAI model through a fixed tool-use sequence (analyze → compress →
verify → check) with retry rules, streaming each step to an optional callback.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from typing import Awaitable, Callable, Optional

from dotenv import load_dotenv
from openai import AsyncOpenAI

from agent.tools import OPENAI_MODEL, TOOLS, execute_tool

# Load variables from a local .env file (no-op if the file is absent).
load_dotenv()

# Per-request timeout + bounded SDK retries so a stalled call can't hang the
# agent forever.
client = AsyncOpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    timeout=60.0,
    max_retries=2,
)

# Hard safety cap on total model turns, so the agent can never loop forever
# (and never silently burns credits) if the model keeps calling tools.
MAX_TURNS = 12


# --------------------------------------------------------------------------- #
# Data models
# --------------------------------------------------------------------------- #

@dataclass
class StepTrace:
    step_num: int
    tool_name: str
    tool_input: dict
    tool_output: str
    timestamp: float = field(default_factory=time.time)


@dataclass
class AgentResult:
    original_prompt: str
    compressed_prompt: str
    original_tokens: int
    compressed_tokens: int
    tokens_saved: int
    reduction_pct: float
    meaning_score: int
    meaning_reasoning: str
    lost_concepts: list[str]
    steps: list[StepTrace]
    total_steps: int
    iterations: int
    success: bool
    failure_reason: str | None = None


AGENT_SYSTEM_PROMPT = """You are a prompt compression agent with 4 tools.

Always follow this exact sequence:
1. analyze_prompt — identify verbosity and fillers in the input
2. compress_with_llmlingua — compress at target_ratio=0.5 first
3. verify_meaning — check meaning is preserved (need score >= 75 to pass)
4. check_token_reduction — calculate final token savings

Decision rules after verify_meaning:
- If pass=false AND score < 50: retry compress_with_llmlingua at target_ratio=0.7
- If pass=false AND score >= 50: retry compress_with_llmlingua at target_ratio=0.6
- If pass=true AND reduction_pct < 20: retry compress_with_llmlingua at target_ratio=0.35
- If pass=true AND reduction_pct >= 20: proceed to check_token_reduction and finish

Max 4 compression attempts. After completing all tool calls, output ONLY
the final compressed prompt as plain text. No explanation, no preamble."""


# --------------------------------------------------------------------------- #
# Agent loop
# --------------------------------------------------------------------------- #

OnStep = Optional[Callable[[StepTrace], Awaitable[None]]]


async def run_agent(prompt: str, on_step: OnStep = None) -> AgentResult:
    steps: list[StepTrace] = []
    step_num = 0
    iterations = 0
    messages: list[dict] = [
        {"role": "system", "content": AGENT_SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]
    final_compressed: str | None = None
    meaning_score = 0
    meaning_reasoning = ""
    lost_concepts: list[str] = []
    token_data: dict = {}

    turns = 0
    try:
        while True:
            turns += 1
            if turns > MAX_TURNS:
                break

            response = await client.chat.completions.create(
                model=OPENAI_MODEL,
                max_tokens=1024,
                tools=TOOLS,
                messages=messages,
            )

            choice = response.choices[0]
            message = choice.message
            tool_calls = message.tool_calls or []

            if choice.finish_reason == "stop" and not tool_calls:
                if message.content:
                    final_compressed = message.content.strip()
                break

            # Echo the assistant turn (including its tool_calls) back into
            # the conversation so the tool results can reference them.
            messages.append(
                {
                    "role": "assistant",
                    "content": message.content,
                    "tool_calls": [
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {
                                "name": tc.function.name,
                                "arguments": tc.function.arguments,
                            },
                        }
                        for tc in tool_calls
                    ],
                }
            )

            for tc in tool_calls:
                step_num += 1
                name = tc.function.name
                try:
                    args = json.loads(tc.function.arguments or "{}")
                except json.JSONDecodeError:
                    args = {}

                if name == "compress_with_llmlingua":
                    iterations += 1

                result = await execute_tool(name, args)

                trace = StepTrace(
                    step_num=step_num,
                    tool_name=name,
                    tool_input=dict(args),
                    tool_output=result,
                )
                steps.append(trace)

                if on_step:
                    await on_step(trace)

                if name == "verify_meaning":
                    try:
                        parsed = json.loads(result)
                        meaning_score = parsed.get("score", 0)
                        meaning_reasoning = parsed.get("reasoning", "")
                        lost_concepts = parsed.get("lost_concepts", [])
                    except Exception:
                        pass

                if name == "check_token_reduction":
                    try:
                        token_data = json.loads(result)
                    except Exception:
                        pass

                if name == "compress_with_llmlingua":
                    final_compressed = result

                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": result,
                    }
                )

            if iterations >= 4:
                break

    except Exception as e:
        word_count = len(prompt.split())
        return AgentResult(
            original_prompt=prompt,
            compressed_prompt=prompt,
            original_tokens=word_count,
            compressed_tokens=word_count,
            tokens_saved=0,
            reduction_pct=0.0,
            meaning_score=0,
            meaning_reasoning="",
            lost_concepts=[],
            steps=steps,
            total_steps=step_num,
            iterations=iterations,
            success=False,
            failure_reason=str(e),
        )

    orig_tokens = token_data.get("original_tokens", len(prompt.split()))
    comp_tokens = token_data.get(
        "compressed_tokens",
        len(final_compressed.split()) if final_compressed else orig_tokens,
    )
    saved = token_data.get("saved", orig_tokens - comp_tokens)
    reduction = token_data.get(
        "reduction_pct",
        round((saved / orig_tokens) * 100, 1) if orig_tokens > 0 else 0,
    )

    return AgentResult(
        original_prompt=prompt,
        compressed_prompt=final_compressed or prompt,
        original_tokens=orig_tokens,
        compressed_tokens=comp_tokens,
        tokens_saved=saved,
        reduction_pct=reduction,
        meaning_score=meaning_score,
        meaning_reasoning=meaning_reasoning,
        lost_concepts=lost_concepts,
        steps=steps,
        total_steps=step_num,
        iterations=iterations,
        success=bool(final_compressed and meaning_score >= 75),
    )
