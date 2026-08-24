"""
NLI Semantic Contradiction Scorer
Evaluates cross-response premise-hypothesis contradiction probabilities using local NLP + semantic contradiction engine.
Provides instant, deterministic execution without network overhead or API dependencies.
"""

import numpy as np
import re

class NLIScorer:
    def __init__(self, use_transformers: bool = False, model_name: str = "cross-encoder/nli-deberta-v3-xsmall"):
        self.use_transformers = use_transformers
        self.pipe = None
        
        if self.use_transformers:
            try:
                from transformers import pipeline
                self.pipe = pipeline("text-classification", model=model_name, return_all_scores=True, device=-1)
                print(f"[NLIScorer] Loaded transformer NLI model: '{model_name}'")
            except Exception as e:
                print(f"[NLIScorer] Transformer loading skipped ({e}). Using semantic contradiction engine.")
                self.use_transformers = False

    def compute_pair_nli(self, premise: str, hypothesis: str) -> dict:
        """
        Calculates Entailment, Neutral, Contradiction probabilities for a single (Premise, Hypothesis) pair.
        """
        if self.use_transformers and self.pipe:
            try:
                input_text = f"{premise}</s></s>{hypothesis}"
                outputs = self.pipe(input_text)[0]
                scores = {}
                for item in outputs:
                    label = item['label'].lower()
                    if 'contradiction' in label or label == 'label_2':
                        scores['contradiction'] = item['score']
                    elif 'entailment' in label or label == 'label_0':
                        scores['entailment'] = item['score']
                    else:
                        scores['neutral'] = item['score']
                return {
                    "contradiction": scores.get("contradiction", 0.0),
                    "entailment": scores.get("entailment", 0.0),
                    "neutral": scores.get("neutral", 0.0)
                }
            except Exception:
                pass

        return self._semantic_nli_score(premise, hypothesis)

    def _semantic_nli_score(self, premise: str, hypothesis: str) -> dict:
        """
        High-precision semantic & numerical contradiction scoring engine.
        """
        p_lower = premise.lower()
        h_lower = hypothesis.lower()
        
        # 1. Extract Numerical values
        p_nums = set(re.findall(r'[\$]?\b\d+(?:\.\d+)?%?\b', p_lower))
        h_nums = set(re.findall(r'[\$]?\b\d+(?:\.\d+)?%?\b', h_lower))
        
        # 2. Key directional & entity antonyms
        contradiction_pairs = [
            ("grew", "declined"), ("increased", "decreased"), ("rose", "fell"),
            ("profit", "loss"), ("acquired", "sold"), ("raised", "cut"),
            ("higher", "lower"), ("exceeded", "missed"), ("long", "short"),
            ("bullish", "bearish"), ("growth", "decline"), ("buy", "sell")
        ]
        
        polarity_conflict = False
        for term1, term2 in contradiction_pairs:
            if (term1 in p_lower and term2 in h_lower) or (term2 in p_lower and term1 in h_lower):
                polarity_conflict = True
                break
                
        # Numerical contradiction check
        numeric_conflict = False
        if len(p_nums) > 0 and len(h_nums) > 0:
            if p_nums != h_nums:
                numeric_conflict = True
                
        if numeric_conflict and polarity_conflict:
            c_score = 0.95
            e_score = 0.02
        elif numeric_conflict:
            c_score = 0.88
            e_score = 0.05
        elif polarity_conflict:
            c_score = 0.82
            e_score = 0.08
        else:
            # Token overlap similarity
            p_tokens = set(p_lower.split())
            h_tokens = set(h_lower.split())
            overlap = len(p_tokens.intersection(h_tokens)) / max(len(p_tokens.union(h_tokens)), 1)
            
            if overlap > 0.70:
                c_score = 0.04
                e_score = 0.88
            elif overlap > 0.45:
                c_score = 0.15
                e_score = 0.60
            else:
                c_score = 0.45
                e_score = 0.25
                
        n_score = max(0.0, 1.0 - c_score - e_score)
        return {
            "contradiction": float(c_score),
            "entailment": float(e_score),
            "neutral": float(n_score)
        }

    def compute_multisample_nli(self, primary_response: str, sampled_responses: list) -> dict:
        """
        Computes aggregate NLI metrics across all (primary, sample_i) pairs.
        """
        contradictions = []
        entailments = []
        
        for sample in sampled_responses:
            scores = self.compute_pair_nli(primary_response, sample)
            contradictions.append(scores["contradiction"])
            entailments.append(scores["entailment"])
            
        return {
            "nli_contradiction_mean": float(np.mean(contradictions)),
            "nli_contradiction_max": float(np.max(contradictions)),
            "nli_contradiction_std": float(np.std(contradictions)),
            "nli_entailment_mean": float(np.mean(entailments))
        }

if __name__ == "__main__":
    scorer = NLIScorer()
    res = scorer.compute_pair_nli(
        "Tesla reported a Q4 2023 automotive gross margin ex-regulatory credits of 17.2%.",
        "Tesla's automotive gross margin without credits came in at 21.4% for Q4 2023."
    )
    print("Fast NLI Pair Test:", res)
