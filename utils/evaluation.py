from __future__ import annotations

import math
import re
from typing import List

try:  # pragma: no cover - optional dependency
    import numpy as np
except Exception:  # pragma: no cover
    np = None


def _mean(values):
    return sum(values) / len(values) if values else 0.0

try:  # pragma: no cover - optional dependency
    from rouge_score import rouge_scorer
except Exception:  # pragma: no cover
    rouge_scorer = None

try:  # pragma: no cover - optional dependency
    from bert_score import score as bert_score
except Exception:  # pragma: no cover
    bert_score = None


def _tokens(text: str) -> List[str]:
    return re.findall(r"\w+", text.lower())


def _lcs_length(a: List[str], b: List[str]) -> int:
    table = [[0] * (len(b) + 1) for _ in range(len(a) + 1)]
    for i, token_a in enumerate(a, start=1):
        for j, token_b in enumerate(b, start=1):
            table[i][j] = table[i - 1][j - 1] + 1 if token_a == token_b else max(table[i - 1][j], table[i][j - 1])
    return table[-1][-1]


class MetricsCalculator:
    def __init__(self):
        self.rouge_scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True) if rouge_scorer else None

    def calculate_ranking_metrics(self, rank_list: list[int]) -> dict:
        if not rank_list:
            return {"MRR": 0.0, "H@1": 0.0, "H@3": 0.0, "H@10": 0.0}
        ranks = [max(1, int(rank)) for rank in rank_list]
        return {
            "MRR": float(_mean([1.0 / rank for rank in ranks])),
            "H@1": float(_mean([rank <= 1 for rank in ranks])),
            "H@3": float(_mean([rank <= 3 for rank in ranks])),
            "H@10": float(_mean([rank <= 10 for rank in ranks])),
        }

    def calculate_bleu4(self, generated: str, reference: str) -> float:
        gen = _tokens(generated)
        ref = _tokens(reference)
        if not gen or not ref:
            return 0.0
        overlap = sum(1 for token in gen if token in ref)
        precision = overlap / len(gen)
        brevity = min(1.0, math.exp(1.0 - len(ref) / max(len(gen), 1)))
        return float(precision * brevity)

    def calculate_rouge_l(self, generated: str, reference: str) -> float:
        if self.rouge_scorer:
            return float(self.rouge_scorer.score(reference, generated)["rougeL"].fmeasure)
        gen = _tokens(generated)
        ref = _tokens(reference)
        if not gen or not ref:
            return 0.0
        lcs = _lcs_length(gen, ref)
        precision = lcs / len(gen)
        recall = lcs / len(ref)
        return float(2 * precision * recall / max(precision + recall, 1e-8))

    def calculate_bert_score(self, generated: str, reference: str, lang: str = "en") -> float:
        if bert_score is None:
            return 0.0
        _, _, f1 = bert_score([generated], [reference], lang=lang, verbose=False)
        return float(f1.mean().item())

    def calculate_explanation_metrics(self, generated_list: list[str], reference_list: list[str]) -> dict:
        if len(generated_list) != len(reference_list):
            raise ValueError("Generated and reference lists must have the same length")
        bleu = [self.calculate_bleu4(gen, ref) for gen, ref in zip(generated_list, reference_list)]
        rouge = [self.calculate_rouge_l(gen, ref) for gen, ref in zip(generated_list, reference_list)]
        bert = [self.calculate_bert_score(gen, ref) for gen, ref in zip(generated_list, reference_list)]
        return {
            "BLEU-4": float(_mean(bleu)),
            "ROUGE-L": float(_mean(rouge)),
            "BertScore-F1": float(_mean(bert)),
        }

    def aggregate_metrics(self, ranking_metrics: dict, explanation_metrics: dict) -> dict:
        return {**ranking_metrics, **explanation_metrics}
