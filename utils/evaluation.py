import numpy as np
import nltk
from rouge_score import rouge_scorer
from bert_score import score as bert_score
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import torch

# Download required NLTK resources if missing
try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt")

smoother = SmoothingFunction()


class MetricsCalculator:
    def __init__(self):
        self.rouge_scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)

    def calculate_ranking_metrics(self, rank_list: list[int]) -> dict:
        """
        Calculate MRR, H@1, H@3, H@10 for temporal KG reasoning tasks
        :param rank_list: List of 1-based ranks for each query
        :return: Dictionary of ranking metrics
        """
        if not rank_list:
            return {"MRR": 0.0, "H@1": 0.0, "H@3": 0.0, "H@10": 0.0}

        mrr = np.mean([1.0 / r for r in rank_list])
        h1 = np.mean([1 if r <= 1 else 0 for r in rank_list])
        h3 = np.mean([1 if r <= 3 else 0 for r in rank_list])
        h10 = np.mean([1 if r <= 10 else 0 for r in rank_list])

        return {
            "MRR": mrr,
            "H@1": h1,
            "H@3": h3,
            "H@10": h10
        }

    def calculate_bleu4(self, generated: str, reference: str) -> float:
        """
        Calculate BLEU-4 score for generated explanations
        """
        gen_tokens = nltk.word_tokenize(generated.lower())
        ref_tokens = nltk.word_tokenize(reference.lower())
        return sentence_bleu([ref_tokens], gen_tokens, weights=(0, 0, 0, 1), smoothing_function=smoother.method4)

    def calculate_rouge_l(self, generated: str, reference: str) -> float:
        """
        Calculate ROUGE-L F1 score for explanation quality
        """
        scores = self.rouge_scorer.score(reference, generated)
        return scores["rougeL"].fmeasure

    def calculate_bert_score(self, generated: str, reference: str, lang: str = "en") -> float:
        """
        Calculate BertScore F1 score for semantic similarity
        """
        P, R, F1 = bert_score([generated], [reference], lang=lang, verbose=False)
        return F1.mean().item()

    def calculate_explanation_metrics(self, generated_list: list[str], reference_list: list[str]) -> dict:
        """
        Calculate all explanation quality metrics for a batch
        """
        if len(generated_list) != len(reference_list):
            raise ValueError("Generated and reference lists must have the same length")

        bleu4_scores = []
        rouge_l_scores = []
        bert_score_scores = []

        for gen, ref in zip(generated_list, reference_list):
            bleu4_scores.append(self.calculate_bleu4(gen, ref))
            rouge_l_scores.append(self.calculate_rouge_l(gen, ref))
            bert_score_scores.append(self.calculate_bert_score(gen, ref))

        return {
            "BLEU-4": np.mean(bleu4_scores),
            "ROUGE-L": np.mean(rouge_l_scores),
            "BertScore-F1": np.mean(bert_score_scores)
        }

    def aggregate_metrics(self, ranking_metrics: dict, explanation_metrics: dict) -> dict:
        """
        Combine ranking and explanation metrics into a single dictionary
        """
        return {**ranking_metrics, **explanation_metrics}


if __name__ == "__main__":
    # Test ranking metrics
    rank_list = [1, 3, 5, 10, 20]
    metrics = MetricsCalculator()
    ranking_metrics = metrics.calculate_ranking_metrics(rank_list)
    print("Ranking metrics:", ranking_metrics)

    # Test explanation metrics
    generated = "The entity is China because USA met China in 2023."
    reference = "The target entity is China, as USA had a meeting relation with China at time 2023."
    explanation_metrics = metrics.calculate_explanation_metrics([generated], [reference])
    print("Explanation metrics:", explanation_metrics)
    print("MetricsCalculator test passed successfully!")