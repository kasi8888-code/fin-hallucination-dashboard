"""
HallucinationBrain - NLP / Data-Science core ("The Brain") of the
Self-Consistency Based Hallucination Risk Dashboard for Financial LLM Responses.

This package is the algorithm layer of the project. It deliberately has NO UI
and NO persistence concerns: those belong to the R/Shiny builder and the
Data/API engineer roles. It only implements the math behind hallucination
detection and exposes it through `HallucinationBrain`.

Public surface (import lazily to keep CLI/imports cheap):
    from src.brain import HallucinationBrain          # scoring API
    from src.brain import risk_assessment            # full pipeline helper
    from src.brain.llm_sampler import LLMSampler      # SelfCheckGPT sampling
    from src.brain.embeddings import EmbeddingModel   # cosine-similarity backends
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - static analysis only
    from .hallucination_brain import HallucinationBrain, RiskAssessment, risk_assessment

__all__ = ["HallucinationBrain", "risk_assessment"]


def __getattr__(name: str):
    # PEP 562 lazy submodule import so `from src.brain import HallucinationBrain`
    # works without eagerly loading numpy/httpx/openai on plain `import src.brain`.
    if name in ("HallucinationBrain", "risk_assessment"):
        from .hallucination_brain import HallucinationBrain, risk_assessment

        return {"HallucinationBrain": HallucinationBrain, "risk_assessment": risk_assessment}[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
