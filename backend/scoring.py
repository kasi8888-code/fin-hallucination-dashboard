"""
Scoring layer for the Hallucination Risk API.

This module is the integration point for the "Brain" (NLP / Data Scientist)
component: the mathematical core (`src/brain`) is owned by that role and is
authoritative for how hallucination risk is computed. The API/DB engineer
("Plumber") owns this FastAPI application; this scoring layer only adapts the
brain's calibrated outputs into the response contract the API exposes.

The brain implements:
  - SelfCheckGPT: N stochastic samples -> pairwise cosine similarity, token
    dispersion, numeric CV / mismatch, Shannon entropy.
  - Groundedness: each sampled response against a trusted reference document
    (dominant signal), with per-sentence risks for the UI heatmap.
  - Fused risk score (0-1) and decision (Faithful / Suspicious / Hallucinated),
    with weights + thresholds calibrated against the project benchmark.

Module-level imports stay cheap: the brain submodules are imported lazily so
that this file does not pull numpy/httpx at import time until first scoring.
"""

from __future__ import annotations

import os
import re
import sys
from typing import Optional

from llm import call_llm

# The brain lives one level up at <repo>/src/brain. Make the repo root
# importable no matter whether uvicorn is started from backend/ or repo root.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

# ---------------------------------------------------------------------------
# Lazy singleton: the "Brain". The offline embedder (deterministic hashed
# vectors) needs no model download, so the API works out-of-the-box. To use
# real semantic embeddings instead, set the HALLUCINATION_EMBEDDER_BACKEND
# env var to an installed backend (e.g. sentence-transformers) before startup.
# ---------------------------------------------------------------------------
_brain = None


def _get_brain():
    global _brain
    if _brain is None:
        from src.brain.embeddings import EmbeddingModel
        from src.brain.hallucination_brain import HallucinationBrain

        backend = os.getenv("HALLUCINATION_EMBEDDER_BACKEND", "offline").strip().lower()
        try:
            embedder = EmbeddingModel(backend=backend)
        except Exception as exc:  # pragma: no cover - defensive
            print(f"Embedder backend '{backend}' unavailable ({exc!r}); using offline.")
            embedder = EmbeddingModel(backend="offline")
        _brain = HallucinationBrain(embedder=embedder)
    return _brain


def split_sentences(text: str) -> list[str]:
    """
    Split an LLM response into meaningful sentences.
    """
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    cleaned = []
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
        if sentence.startswith("#"):
            continue
        if re.fullmatch(r"[-*]?\s*\d+\.", sentence):
            continue
        cleaned.append(sentence)
    return cleaned


def generate_multiple_answers(
    prompt: str,
    number_of_answers: int = 3,
    temperature: Optional[float] = 0.8,
    model: Optional[str] = None,
) -> list[str]:
    """
    Generate multiple independent answers for the same prompt.

    ``temperature`` is forwarded to the LLM so the samples are genuinely
    varied (required for the SelfCheckGPT consistency signal). When a
    reference document is provided upstream, callers should pass a lower
    temperature for the deterministic/"primary" answer separately.
    """
    if model is None:
        from llm import DEFAULT_MODEL

        model = DEFAULT_MODEL

    answers = []
    for _ in range(number_of_answers):
        answer = call_llm(
            prompt,
            temperature=temperature,
            model=model,
        )
        answers.append(answer)
    return answers


def _consistency_metrics(answers: list[str]) -> dict:
    """
    Return the brain's self-consistency statistics for the N samples.

    Raises ValueError when fewer than 2 answers are provided.
    """
    brain = _get_brain()
    return brain.selfcheck_metrics(answers)


def calculate_consistency(answers: list[str]) -> float:
    """
    Overall self-consistency in [0, 1] across the N sampled answers.

    This is the mean pairwise embedding cosine similarity computed by the
    brain (SelfCheckGPT). The API historically derived risk as
    ``1 - consistency``; that is preserved for backward compatibility, but the
    authoritative fused risk (groundedness + self-check + dispersion) is
    produced by ``score_responses`` and returned in the analysis.
    """
    if len(answers) < 2:
        return 1.0
    brain = _get_brain()
    metrics = brain.selfcheck_metrics(answers)
    return float(metrics["similarity_mean"])


def _threshold_for_sentence_risk() -> float:
    """
    Derive the is_hallucinated cutoff from the brain's decision bands so the
    sentence flag stays consistent with the overall verdict, instead of a
    hard-coded cosine threshold that does not transfer across embedders.
    """
    brain = _get_brain()
    suspicious = brain.detect_bands.get("suspicious", 0.52)
    return float(suspicious)


def calculate_sentence_scores(
    answers: list[str],
    similarity_threshold: Optional[float] = None,
) -> list[dict]:
    """
    Per-sentence hallucination risk.

    For the API's primary/displayed response (``answers[0]``), each sentence is
    compared against every sentence in the OTHER sampled answers; the best
    semantic match per other answer is kept (SelfCheckGPT sentence-level).
    Risk for each sentence is ``1 - best-consistency`` and the boolean flag uses
    the brain's calibrated suspicious band (not a guessed constant) unless an
    explicit ``similarity_threshold`` is passed.

    When a trusted ``reference`` is available the caller should prefer the
    brain's groundedness scoring (see ``score_responses``), which is stronger.
    """
    if len(answers) < 2:
        return []

    threshold = (
        _threshold_for_sentence_risk()
        if similarity_threshold is None
        else similarity_threshold
    )

    brain = _get_brain()
    embed = brain.embedder
    all_sentences = [split_sentences(answer) for answer in answers]

    # Pre-encode every sentence once (the original loop re-encoded per
    # sentence, which was O(n^2 * m) wasted embedding calls).
    encoded = [
        [embed.embed(sentence) for sentence in sentences]
        for sentences in all_sentences
    ]

    results = []
    for answer_index, (sentences, vectors) in enumerate(zip(all_sentences, encoded)):
        for sentence, sentence_vector in zip(sentences, vectors):
            similarities = []
            for other_index, other_vectors in enumerate(encoded):
                if answer_index == other_index:
                    continue
                if not other_vectors:
                    continue
                best = max(
                    _cosine(sentence_vector, other_vector)
                    for other_vector in other_vectors
                )
                similarities.append(best)
            if not similarities:
                continue
            consistency = sum(similarities) / len(similarities)
            risk = 1.0 - consistency
            results.append({
                "sentence": sentence,
                "score": round(float(risk), 4),
                "is_hallucinated": int(risk >= threshold),
            })
    return results


def _cosine(vec_a, vec_b) -> float:
    """Cosine similarity via the brain's embedder (keeps one implementation)."""
    from src.brain.embeddings import cosine_similarity

    return float(cosine_similarity(vec_a, vec_b))


def score_responses(
    question: str,
    answers: list[str],
    reference: Optional[str] = None,
    primary: Optional[str] = None,
) -> dict:
    """
    Full brain assessment for the API.

    Returns a dict shaped like the brain's ``RiskAssessment.to_dict()``:
    decision, risk_score, metrics (selfcheck + groundedness when a reference is
    supplied), sentence_risks (grounded vs the reference when available, else
    self-consistency sentence scores), provider/model.
    """
    brain = _get_brain()
    from src.brain.hallucination_brain import RiskAssessment

    if primary is None:
        primary = answers[0]

    if reference and reference.strip():
        assessment: RiskAssessment = brain.assess(
            question=question,
            responses=answers,
            reference=reference,
            primary=primary,
        )
        return assessment.to_dict()

    # No trusted reference: fuse self-check signals through the brain.
    assessment = brain.assess(
        question=question,
        responses=answers,
        reference=None,
        primary=primary,
    )
    result = assessment.to_dict()
    # Add sentence-level risks from self-consistency so the UI always has a
    # heatmap even without a reference document. Normalise the key to `risk`
    # so both the groundedness and the self-consistency branches emit the same
    # dict shape (sentence / risk / is_hallucinated) for the API layer.
    result["sentence_risks"] = [
        {
            "sentence": item["sentence"],
            "risk": round(item["score"], 4),
            "is_hallucinated": int(item["score"] >= 0.5),
        }
        for item in calculate_sentence_scores(answers)
    ]
    return result
