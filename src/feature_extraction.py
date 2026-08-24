"""
Feature Extractor for Multi-Sample Self-Consistency Hallucination Detection
Extracts:
1. Numeric Variance & Mismatch Rate
2. spaCy Named Entity Consistency (Jaccard Distance & Mismatches)
3. Lexical Disagreement (ROUGE-L & Token Jaccard Distance)
4. Response Length Dispersion
5. NLI Semantic Contradiction & Entailment Scores
"""

import re
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import spacy
from typing import List, Dict, Any
from src.nli_scorer import NLIScorer

# Load spaCy model
try:
    nlp = spacy.load("en_core_web_sm")
except Exception:
    nlp = None

def extract_numbers(text: str) -> List[float]:
    """
    Extracts all numerical values (percentages, currency amounts, floats, integers) from text.
    """
    # Matches patterns like $81.8, 17.2%, 14.51 billion -> normalized to floats where applicable
    clean_text = text.replace(",", "")
    matches = re.findall(r'[\$]?\b\d+(?:\.\d+)?%?\b', clean_text)
    
    nums = []
    for m in matches:
        m_clean = m.replace("$", "").replace("%", "")
        try:
            val = float(m_clean)
            nums.append(val)
        except ValueError:
            continue
    return nums

def compute_numeric_dispersion(sampled_responses: List[str]) -> Dict[str, float]:
    """
    Computes statistical dispersion of numerical figures across sampled responses.
    """
    all_num_lists = [extract_numbers(r) for r in sampled_responses]
    all_nums = [n for sublist in all_num_lists for n in sublist]
    
    if not all_nums or len(all_nums) <= 1:
        return {"numeric_cv": 0.0, "numeric_mismatch_rate": 0.0}
    
    mean_val = np.mean(all_nums)
    std_val = np.std(all_nums)
    cv = std_val / mean_val if mean_val != 0 else 0.0
    
    # Calculate pairwise mismatch rate
    n = len(sampled_responses)
    mismatches = 0
    total_pairs = 0
    for i in range(n):
        for j in range(i + 1, n):
            set_i = set(all_num_lists[i])
            set_j = set(all_num_lists[j])
            if set_i != set_j:
                mismatches += 1
            total_pairs += 1
            
    mismatch_rate = mismatches / max(total_pairs, 1)
    return {
        "numeric_cv": float(cv),
        "numeric_mismatch_rate": float(mismatch_rate)
    }

def extract_entities(text: str) -> List[str]:
    """
    Extracts named entities using spaCy.
    """
    if nlp is None:
        # Fallback simple regex capitalization extractor
        words = text.split()
        return [w.strip(".,()") for w in words if w.istitle() and len(w) > 2]
    
    doc = nlp(text)
    # Focus on key financial entity types: ORG, PERSON, GPE, MONEY, DATE, LAW, PERCENT
    target_label_types = {"ORG", "PERSON", "GPE", "MONEY", "DATE", "LAW", "PERCENT", "CARDINAL"}
    entities = [ent.text.strip().lower() for ent in doc.ents if ent.label_ in target_label_types]
    return list(set(entities))

def compute_entity_dispersion(sampled_responses: List[str]) -> Dict[str, float]:
    """
    Computes spaCy Named Entity pairwise Jaccard Distance and entity inconsistency count.
    """
    entity_sets = [set(extract_entities(r)) for r in sampled_responses]
    
    n = len(sampled_responses)
    jaccard_distances = []
    
    for i in range(n):
        for j in range(i + 1, n):
            u = entity_sets[i].union(entity_sets[j])
            inter = entity_sets[i].intersection(entity_sets[j])
            if len(u) == 0:
                jaccard_distances.append(0.0)
            else:
                sim = len(inter) / len(u)
                jaccard_distances.append(1.0 - sim)
                
    all_unique_entities = set().union(*entity_sets)
    mismatch_counts = 0
    for ent in all_unique_entities:
        presence = sum(1 for e_set in entity_sets if ent in e_set)
        if 0 < presence < len(sampled_responses):
            mismatch_counts += 1
            
    return {
        "entity_jaccard_distance": float(np.mean(jaccard_distances)) if jaccard_distances else 0.0,
        "entity_mismatch_count": float(mismatch_counts)
    }

def compute_lexical_dispersion(sampled_responses: List[str]) -> Dict[str, float]:
    """
    Computes token-level Jaccard distance and length dispersion.
    """
    token_sets = [set(r.lower().split()) for r in sampled_responses]
    n = len(sampled_responses)
    jaccard_distances = []
    
    for i in range(n):
        for j in range(i + 1, n):
            u = token_sets[i].union(token_sets[j])
            inter = token_sets[i].intersection(token_sets[j])
            sim = len(inter) / len(u) if len(u) > 0 else 1.0
            jaccard_distances.append(1.0 - sim)
            
    lengths = [len(r.split()) for r in sampled_responses]
    mean_len = np.mean(lengths)
    std_len = np.std(lengths)
    length_dispersion = std_len / mean_len if mean_len > 0 else 0.0
    
    return {
        "token_jaccard_distance": float(np.mean(jaccard_distances)) if jaccard_distances else 0.0,
        "length_dispersion": float(length_dispersion)
    }

class FeatureExtractor:
    def __init__(self):
        self.nli_scorer = NLIScorer()
        
    def extract_features(self, item: Dict[str, Any]) -> Dict[str, float]:
        """
        Processes a benchmark QA item containing deterministic_response and sampled_responses.
        Returns a complete feature dictionary.
        """
        primary_res = item["deterministic_response"]
        samples = item["sampled_responses"]
        all_responses = [primary_res] + samples
        
        # 1. Statistical Dispersion Features
        num_feats = compute_numeric_dispersion(all_responses)
        ent_feats = compute_entity_dispersion(all_responses)
        lex_feats = compute_lexical_dispersion(all_responses)
        
        # 2. NLI Semantic Features
        nli_feats = self.nli_scorer.compute_multisample_nli(primary_res, samples)
        
        # Combine into single feature vector dict
        features = {
            **num_feats,
            **ent_feats,
            **lex_feats,
            **nli_feats
        }
        return features

if __name__ == "__main__":
    import json
    with open("data/financial_qa_dataset.json", "r") as f:
        data = json.load(f)
        
    fe = FeatureExtractor()
    sample_item = data[1] # Tesla gross margin example
    feats = fe.extract_features(sample_item)
    print("Extracted Features for Sample Item:", json.dumps(feats, indent=2))
