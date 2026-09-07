"""
calibrate_thresholds.py — the "Brain" role's threshold-tuning tool.

SelfCheckGPT thresholds depend on which LLM you sample from (temperature,
refusal behaviour, tokeniser) and which embedder you score with. This harness
samples a small labelled probe set live from your configured LLM, computes the
self-consistency + grounding features, and prints a data-driven recommendation
for the fusion weights and decision bands in HallucinationBrain.

Design intent: you run this once per (LLM provider, embedder) pair, paste the
recommended numbers into HallucinationBrain(..., fusion=..., thresholds=...)
via the dashboard settings, and you never hand-tune again. The probe set here is
small (10 questions) and costs a handful of Groq/OpenAI tokens.

Usage:
    python calibrate_thresholds.py                 # uses provider="auto" (.env key)
    python calibrate_thresholds.py --provider groq --model qwen/qwen3.8-27b
    python calibrate_thresholds.py --only-report _calib_cache.json   # reuse cached samples

The script writes _calib_cache.json (the raw samples + features) so you can
re-fit the report offline without re-spending API tokens.
"""

from __future__ import annotations

import argparse
import json
import sys
import os
from itertools import product
from typing import Dict, List, Tuple

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.brain.llm_sampler import LLMSampler
from src.brain.embeddings import EmbeddingModel
from src.brain.hallucination_brain import HallucinationBrain

# ---------------------------------------------------------------------------
# Probe set: (question, trusted-reference text, risky?)
#   risky=False -> the model should know the fact; consistent samples expected
#   risky=True  -> obscure/precise fact the model will likely guess/refuse on
# ---------------------------------------------------------------------------
PROBE_ITEMS = [
    ("What was NVIDIA Data Center revenue in Q3 of fiscal 2024?",
     "NVIDIA Data Center revenue was $14.51 billion in Q3 fiscal 2024.", False),
    ("How many vehicles did Tesla deliver in full-year 2023?",
     "Tesla delivered about 1.81 million vehicles in 2023.", False),
    ("What was Apple's total revenue in fiscal Q4 2023?",
     "Apple's fiscal Q4 2023 revenue was about $89.5 billion.", False),
    ("What was the Fed's target federal funds rate range at the end of 2023?",
     "The Fed's target range was 5.25%-5.50% at the end of 2023.", False),
    ("What was Meta Platforms' full-year 2023 revenue?",
     "Meta's 2023 full-year revenue was about $134.9 billion.", False),
    ("What was the exact settlement value in USD of the Fed's overnight reverse repo "
     "facility on 31 December 2021?",
     "Around $1.9 trillion.", True),
    ("What was the exact intraday low in points of the S&P 500 on 12 March 2020?",
     "Around 2,477 points.", True),
    ("What was Goldman Sachs' exact provision for credit losses in Q1 2023?",
     "Around $361 million.", True),
    ("Who was BlackRock's CFO in January 2024 and what was his exact age in years then?",
     "Martin Small was named CFO; exact age is not public knowledge.", True),
    ("What was Airbnb's exact year-over-year revenue growth percentage in Q4 2022?",
     "Around 24% year over year.", True),
]

# self-consistency features whose means we report; keys must exist in
# selfcheck_metrics() output
REPORT_FEATURES = [
    "similarity_mean", "similarity_min", "token_jaccard_distance",
    "numeric_cv", "numeric_mismatch_rate", "shannon_entropy",
]


def _collect(provider: str, model: str, n_samples: int) -> List[Dict]:
    sampler = LLMSampler(provider=provider, model=model)
    brain = HallucinationBrain(embedder=EmbeddingModel(backend="auto"))
    print(f"sampling {len(PROBE_ITEMS)} probes with provider={sampler.provider} "
          f"model={sampler.model}, n={n_samples} stochastic samples each ...")
    rows = []
    for q, ref, risky in PROBE_ITEMS:
        res = sampler.sample_many(q, n=n_samples, temperature=0.8, include_deterministic=False)
        responses = res.sampled_responses
        sc = brain.selfcheck_metrics(responses)
        sims = [brain.embedder.similarity(r, ref) for r in responses]
        sc["min_sim_ref"] = min(sims)
        sc["mean_sim_ref"] = sum(sims) / len(sims)
        rows.append({
            "question": q, "reference": ref, "risky": risky,
            "provider": res.provider, "model": res.model,
            "responses": responses, "selfcheck": sc,
        })
        print(f"  [{'RISKY' if risky else 'FAITH'}] {q[:70]}")
    return rows


def _fit_report(rows: List[Dict]) -> None:
    n = len(rows)
    features = REPORT_FEATURES + ["min_sim_ref", "mean_sim_ref"]
    print(f"\n=== calibration report ({n} probes) ===")
    print("feature means by label:")
    print(f"  {'feature':>22}  {'faithful':>8}  {'risky':>8}")
    for f in features:
        faith = [r["selfcheck"][f] for r in rows if not r["risky"]]
        risky = [r["selfcheck"][f] for r in rows if r["risky"]]
        print(f"  {f:>22}  {sum(faith)/len(faith):8.3f}  {sum(risky)/len(risky):8.3f}")

    # grid-search a weighted grounded+selfcheck score -> recommended band
    best: Tuple = (0.0,)
    candidates = ["min_sim_ref", "similarity_min", "token_jaccard_distance",
                  "numeric_mismatch_rate", "shannon_entropy"]
    weights = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
    for combo in product(weights, repeat=len(candidates)):
        if sum(combo) == 0:
            continue
        scores = [
            sum(combo[j] * (1.0 - r["selfcheck"][candidates[j]]) for j in range(len(candidates)))
            for r in rows
        ]
        # a higher score = more risk
        for lo in range(1, n):
            thresh = (sorted(scores)[lo - 1] + sorted(scores)[lo]) / 2
            tp = tn = fp = fn = 0
            for s, r in zip(scores, rows):
                pred = s > thresh
                if pred and r["risky"]: tp += 1
                elif not pred and not r["risky"]: tn += 1
                elif pred and not r["risky"]: fp += 1
                else: fn += 1
            acc = (tp + tn) / n
            if acc > best[0]:
                best = (acc, combo, thresh, tp, tn, fp, fn)
    acc, combo, thresh, tp, tn, fp, fn = best
    print(f"\nrecommended (weighted risk > {thresh:.3f}): acc={acc:.3f} "
          f"TP={tp} TN={tn} FP={fp} FN={fn}")
    print("suggested fusion weights (risk components, sum to 1.0):")
    total = sum(combo)
    mapping = dict(zip(candidates, combo))
    for name, w in mapping.items():
        if w:
            print(f"  {'grounded_min' if name == 'min_sim_ref' else name}: {w / total:.2f}")
    print("\nPaste those weights into HallucinationBrain(fusion=..., thresholds=...).")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--provider", default="auto", help="auto|openai|groq|ollama")
    parser.add_argument("--model", default=None, help="override model name")
    parser.add_argument("--n-samples", type=int, default=3, help="stochastic samples per probe")
    parser.add_argument("--cache", default="_calib_cache.json")
    parser.add_argument("--only-report", action="store_true",
                        help="reuse cached samples instead of calling the LLM")
    args = parser.parse_args()

    if args.only_report and os.path.isfile(args.cache):
        rows = json.load(open(args.cache, encoding="utf-8"))
    else:
        rows = _collect(args.provider, args.model, args.n_samples)
        with open(args.cache, "w", encoding="utf-8") as f:
            json.dump(rows, f, indent=2, ensure_ascii=False)
        print(f"wrote {args.cache}")
    _fit_report(rows)


if __name__ == "__main__":
    main()
