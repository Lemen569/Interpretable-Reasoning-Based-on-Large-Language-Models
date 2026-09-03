from __future__ import annotations

from collections import defaultdict
from typing import Dict, Iterable, List, Sequence, Tuple

import torch

try:  # pragma: no cover - optional dependency
    import numpy as np
except Exception:  # pragma: no cover
    np = None

from models.gnn import TemporalGNN
from models.llm import LLMWrapper
from models.transformer import FusionTransformer


class ReasoningEngine:
    """Fuse retained HTIR paths with GNN scores and verify both directions."""

    def __init__(
        self,
        gnn: TemporalGNN | None,
        llm: LLMWrapper | None,
        fusion_transformer: FusionTransformer | None,
        decay_rate: float = 0.08,
        alpha: float = 0.5,
        beta: float = 0.7,
        device: str = "cuda",
        enable_fusion: bool = True,
        enable_bidirectional: bool = True,
    ):
        self.gnn = gnn
        self.llm = llm
        self.transformer = fusion_transformer
        self.decay_rate = decay_rate
        self.alpha = alpha
        self.beta = beta
        self.device = device
        self.enable_fusion = enable_fusion
        self.enable_bidirectional = enable_bidirectional

    def time_decay(self, time_delta: int | float) -> float:
        if np is not None:
            return float(np.exp(-self.decay_rate * abs(float(time_delta))))
        import math
        return math.exp(-self.decay_rate * abs(float(time_delta)))

    def llm_path_scoring(self, paths: Sequence[Sequence[Tuple[int, int, int]]], query_prompt: str) -> float:
        if not paths or self.llm is None:
            return 0.0
        scores = []
        for path in paths:
            path_text = " -> ".join(f"({entity},{relation},{time})" for entity, relation, time in path)
            raw = self.llm.score(f"{query_prompt}\nReasoning path: {path_text}")
            times = [step[2] for step in path if step[2] >= 0]
            decay = self.time_decay(max(times) - min(times)) if len(times) > 1 else 1.0
            scores.append(self.beta * raw + (1.0 - self.beta) * decay)
        return float(sum(scores) / len(scores))

    def compute_llm_scores(self, entity_paths: Dict[int, List[list]], query_prompt: str) -> Dict[int, float]:
        return {
            int(entity): score
            for entity, paths in entity_paths.items()
            if (score := self.llm_path_scoring(paths, query_prompt)) >= self.alpha
        }

    def compute_gnn_scores(
        self,
        graph,
        entity_ids: torch.Tensor,
        rel_ids: torch.Tensor,
        time_ids: torch.Tensor,
        head_id: int,
        relation_id: int,
        query_time: int,
        entities: Iterable[int],
    ) -> Dict[int, float]:
        if self.gnn is None:
            return {int(entity): 0.0 for entity in entities}
        candidate_ids = torch.tensor(list(entities), dtype=torch.long, device=entity_ids.device)
        scores = self.gnn.score_candidates(
            graph, entity_ids, relation_id, query_time, head_id, candidate_ids
        )
        return {int(entity): float(score.item()) for entity, score in zip(candidate_ids.tolist(), scores)}

    def fuse_feature_scores(self, llm_score: float, gnn_score: float) -> float:
        if not self.enable_fusion or self.transformer is None:
            return float(self.beta * llm_score + (1.0 - self.beta) * gnn_score)
        return self.transformer.fuse_features(
            torch.tensor([gnn_score], dtype=torch.float32, device=self.device),
            torch.tensor([llm_score], dtype=torch.float32, device=self.device),
        )

    def batch_fusion(self, llm_scores: Dict[int, float], gnn_scores: Dict[int, float]) -> Dict[int, float]:
        entities = set(llm_scores) | set(gnn_scores)
        return {
            entity: self.fuse_feature_scores(llm_scores.get(entity, 0.0), gnn_scores.get(entity, 0.0))
            for entity in entities
        }

    def bidirectional_verify(
        self,
        forward_scores: Dict[int, float],
        backward_scores: Dict[int, float],
    ) -> Tuple[int, float]:
        if not forward_scores and not backward_scores:
            raise ValueError("At least one direction must provide candidate scores.")
        if not backward_scores or not self.enable_bidirectional:
            source = forward_scores or backward_scores
            entity = max(source, key=source.get)
            return entity, float(source[entity])
        f_entity = max(forward_scores, key=forward_scores.get)
        b_entity = max(backward_scores, key=backward_scores.get)
        if f_entity == b_entity:
            return f_entity, (forward_scores[f_entity] + backward_scores[b_entity]) / 2.0
        # A second verification pass can be performed by the caller; if the
        # predictions remain inconsistent, use the higher fused score.
        return (
            (f_entity, forward_scores[f_entity])
            if forward_scores[f_entity] >= backward_scores[b_entity]
            else (b_entity, backward_scores[b_entity])
        )

    def forward(self, **kwargs):
        forward_scores = kwargs.get("forward_scores", {})
        backward_scores = kwargs.get("backward_scores", {})
        return self.bidirectional_verify(forward_scores, backward_scores)
