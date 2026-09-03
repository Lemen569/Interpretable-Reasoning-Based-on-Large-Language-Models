from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn

from .gnn import TemporalGNN
from .llm import LLMWrapper
from .transformer import FusionTransformer


class TKLXR(nn.Module):
    """TKL-XR model wrapper.

    The wrapper exposes the same stages described in the paper: structural GNN
    scoring, LLM path scoring, adaptive/linear fusion, and bidirectional
    verification.  Feature switches are provided for the ablation scripts.
    """

    def __init__(
        self,
        entity_num: int,
        relation_num: int,
        time_num: int,
        llm_model_name: str = "meta-llama/Llama-2-13b-chat-hf",
        embed_dim: int = 128,
        gnn_layers: int = 3,
        trans_layers: int = 3,
        trans_heads: int = 8,
        decay_rate: float = 0.08,
        dropout: float = 0.1,
        load_4bit: bool = True,
        device: str = "cuda",
        enable_gnn: bool = True,
        enable_llm: bool = True,
        enable_fusion: bool = True,
        enable_htir: bool = True,
        enable_bidirectional: bool = True,
        linear_fusion: bool = False,
        alpha: float = 0.5,
        beta: float = 0.7,
        beam_depth: int = 4,
        beam_width: int = 4,
        use_mock_llm: Optional[bool] = None,
    ):
        super().__init__()
        # Keep the public ``device`` argument convenient for scripts while
        # falling back safely on CPU when CUDA is unavailable.
        self.device_name = (
            device if device != "cuda" or torch.cuda.is_available() else "cpu"
        )
        self.enable_gnn = enable_gnn
        self.enable_llm = enable_llm
        self.enable_fusion = enable_fusion
        self.enable_htir = enable_htir
        self.enable_bidirectional = enable_bidirectional
        self.linear_fusion = linear_fusion
        self.alpha = alpha
        self.beta = beta
        self.beam_depth = beam_depth
        self.beam_width = beam_width
        self.decay_rate = decay_rate
        # Trainable linear-fusion baseline parameters.  They are initialized
        # to the previous beta weighting so existing checkpoints remain
        # numerically well behaved while the ablation is formally learnable.
        self.linear_w_llm = nn.Parameter(torch.tensor(float(beta)))
        self.linear_w_gnn = nn.Parameter(torch.tensor(float(1.0 - beta)))
        self.linear_bias = nn.Parameter(torch.tensor(0.0))

        self.gnn = (
            TemporalGNN(
                entity_num=entity_num,
                relation_num=relation_num,
                time_num=time_num,
                embed_dim=embed_dim,
                gnn_layers=gnn_layers,
                decay_rate=decay_rate,
                dropout=dropout,
            )
            if enable_gnn
            else None
        )
        self.llm = LLMWrapper(
            model_name=llm_model_name,
            load_4bit=load_4bit,
            device=self.device_name,
            use_mock=use_mock_llm,
        ) if enable_llm else None
        self.fusion_network = (
            FusionTransformer(
                d_model=embed_dim,
                n_heads=trans_heads,
                n_layers=trans_layers,
                dropout=dropout,
                device=self.device_name,
            )
            if enable_fusion and enable_gnn and enable_llm and not linear_fusion
            else None
        )

    @staticmethod
    def _mean_or_zero(values: Iterable[float]) -> float:
        values = list(values)
        return float(sum(values) / len(values)) if values else 0.0

    def fuse_scores_tensor(self, llm_score, gnn_score):
        if not torch.is_tensor(llm_score):
            llm_score = torch.tensor([llm_score], dtype=torch.float32, device=self.device_name)
        else:
            llm_score = llm_score.to(self.device_name).reshape(-1)
        if not torch.is_tensor(gnn_score):
            gnn_score = torch.tensor([gnn_score], dtype=torch.float32, device=self.device_name)
        else:
            gnn_score = gnn_score.to(self.device_name).reshape(-1)
        if not self.enable_llm:
            return gnn_score
        if not self.enable_gnn:
            return llm_score
        if not self.enable_fusion:
            return llm_score
        if self.linear_fusion or self.fusion_network is None:
            return self.linear_w_llm * llm_score + self.linear_w_gnn * gnn_score + self.linear_bias

        return self.fusion_network(gnn_score, llm_score)

    def fuse_scores(self, llm_score: float, gnn_score: float) -> float:
        return float(self.fuse_scores_tensor(llm_score, gnn_score).reshape(-1)[0].detach().item())

    def bidirectional_verification(
        self,
        forward_scores: Dict[int, float],
        backward_scores: Optional[Dict[int, float]] = None,
    ) -> Tuple[int, float]:
        if not forward_scores:
            raise ValueError("forward_scores cannot be empty")
        if not self.enable_bidirectional or not backward_scores:
            entity = max(forward_scores, key=forward_scores.get)
            return entity, float(forward_scores[entity])

        forward_entity = max(forward_scores, key=forward_scores.get)
        backward_entity = max(backward_scores, key=backward_scores.get)
        if forward_entity == backward_entity:
            return forward_entity, float(
                (forward_scores[forward_entity] + backward_scores[backward_entity]) / 2.0
            )
        if forward_scores[forward_entity] >= backward_scores[backward_entity]:
            return forward_entity, float(forward_scores[forward_entity])
        return backward_entity, float(backward_scores[backward_entity])

    def candidate_score_tensor(
        self,
        graph,
        entity_ids: torch.Tensor,
        rel_ids: torch.Tensor,
        time_ids: torch.Tensor,
        head_id: int,
        relation_id: int,
        query_time: int,
        candidate_id: int,
        prompt: str,
        path_score: Optional[float] = None,
        use_llm: Optional[bool] = None,
    ):
        if self.enable_gnn and self.gnn is not None:
            candidate = torch.tensor([candidate_id], dtype=torch.long, device=entity_ids.device)
            gnn_score = self.gnn.score_candidates(
                graph,
                entity_ids,
                relation_id,
                query_time,
                head_id,
                candidate,
            )[0]
        else:
            gnn_score = torch.tensor(0.0, dtype=torch.float32, device=self.device_name)
        should_use_llm = use_llm if use_llm is not None else True
        if self.enable_llm and self.llm is not None and should_use_llm:
            llm_score = (
                float(path_score)
                if path_score is not None
                else self.llm.score(f"{prompt}\nCandidate entity: {candidate_id}")
            )
        else:
            llm_score = torch.tensor(0.0, dtype=torch.float32, device=self.device_name)
        return self.fuse_scores_tensor(llm_score, gnn_score)

    def candidate_score(
        self,
        graph,
        entity_ids: torch.Tensor,
        rel_ids: torch.Tensor,
        time_ids: torch.Tensor,
        head_id: int,
        relation_id: int,
        query_time: int,
        candidate_id: int,
        prompt: str,
        path_score: Optional[float] = None,
        use_llm: Optional[bool] = None,
    ) -> float:
        score = self.candidate_score_tensor(
            graph,
            entity_ids,
            rel_ids,
            time_ids,
            head_id,
            relation_id,
            query_time,
            candidate_id,
            prompt,
            path_score,
            use_llm,
        )
        return float(score.reshape(-1)[0].detach().item())

    def rank_candidates(
        self,
        graph,
        entity_ids: torch.Tensor,
        rel_ids: torch.Tensor,
        time_ids: torch.Tensor,
        head_id: int,
        relation_id: int,
        query_time: int,
        candidate_ids: Sequence[int],
        prompt: str,
        path_scores: Optional[Dict[int, float]] = None,
        llm_candidate_ids: Optional[Sequence[int]] = None,
    ) -> Dict[int, float]:
        candidate_list = [int(candidate_id) for candidate_id in candidate_ids]
        if not candidate_list:
            return {}
        llm_candidates = (
            None
            if llm_candidate_ids is None
            else {int(candidate_id) for candidate_id in llm_candidate_ids}
        )
        candidate_tensor = torch.tensor(
            candidate_list, dtype=torch.long, device=entity_ids.device
        )
        if self.enable_gnn and self.gnn is not None:
            gnn_scores = self.gnn.score_candidates(
                graph,
                entity_ids,
                relation_id,
                query_time,
                head_id,
                candidate_tensor,
            )
        else:
            gnn_scores = torch.zeros(
                len(candidate_list), dtype=torch.float32, device=self.device_name
            )

        # LLM scoring remains candidate-specific, but structural propagation is
        # computed once for the complete ranking set.
        llm_values = []
        for candidate_id in candidate_list:
            allowed = llm_candidates is None or candidate_id in llm_candidates
            if self.enable_llm and self.llm is not None and allowed:
                value = (
                    path_scores.get(candidate_id)
                    if path_scores is not None and candidate_id in path_scores
                    else self.llm.score(f"{prompt}\nCandidate entity: {candidate_id}")
                )
            else:
                value = 0.0
            llm_values.append(float(value))
        llm_scores = torch.tensor(
            llm_values, dtype=torch.float32, device=entity_ids.device
        )
        fused = self.fuse_scores_tensor(llm_scores, gnn_scores).detach().reshape(-1)
        scores = {
            candidate_id: float(score.item())
            for candidate_id, score in zip(candidate_list, fused)
        }
        return dict(sorted(scores.items(), key=lambda item: item[1], reverse=True))

    def forward(
        self,
        graph,
        entity_ids,
        rel_ids,
        time_ids,
        forward_prompt,
        backward_prompt,
        target_entity: int,
    ):
        score = self.candidate_score(
            graph,
            entity_ids,
            rel_ids,
            time_ids,
            head_id=target_entity,
            relation_id=0,
            query_time=0,
            candidate_id=target_entity,
            prompt=forward_prompt,
        )
        backward_score = self.candidate_score(
            graph,
            entity_ids,
            rel_ids,
            time_ids,
            head_id=target_entity,
            relation_id=0,
            query_time=0,
            candidate_id=target_entity,
            prompt=backward_prompt,
        )
        if self.enable_bidirectional:
            score = (score + backward_score) / 2.0
        return target_entity, score

    def infer_with_explanation(
        self,
        graph,
        entity_ids,
        rel_ids,
        time_ids,
        query_prompt: str,
        target_entity: int,
    ):
        _, final_score = self.forward(
            graph,
            entity_ids,
            rel_ids,
            time_ids,
            query_prompt,
            query_prompt,
            target_entity,
        )
        explanation = (
            self.llm.generate(query_prompt)
            if self.enable_llm and self.llm is not None
            else f"Prediction is supported by the structural score for entity {target_entity}."
        )
        return final_score, explanation
