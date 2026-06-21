"""LLMLingua-2 local compression.

Loads the LLMLingua-2 model once (lazy singleton) and exposes a `compress`
helper. This runs entirely on the server CPU — zero API cost.
"""

from __future__ import annotations

import logging

from llmlingua import PromptCompressor

_compressor: PromptCompressor | None = None
logger = logging.getLogger(__name__)


def get_compressor() -> PromptCompressor:
    """Return the process-wide LLMLingua-2 compressor, loading it once."""
    global _compressor
    if _compressor is None:
        logger.info("Loading LLMLingua-2 model...")
        _compressor = PromptCompressor(
            model_name="microsoft/llmlingua-2-bert-base-multilingual-cased-meetingbank",
            use_llmlingua2=True,
            device_map="cpu",
        )
        logger.info("LLMLingua-2 loaded.")
    return _compressor


def compress(text: str, target_ratio: float) -> str:
    """Compress `text` to roughly `target_ratio` of its original token budget."""
    try:
        result = get_compressor().compress_prompt(
            text,
            rate=target_ratio,
            force_tokens=[],
            drop_consecutive=True,
        )
        return result["compressed_prompt"]
    except Exception as e:
        logger.error(f"LLMLingua-2 compression failed: {e}")
        raise
