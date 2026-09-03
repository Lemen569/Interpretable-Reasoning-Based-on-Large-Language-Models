from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

try:  # pragma: no cover - optional runtime dependency
    import dgl
    import dgl.nn as dgl_nn
except Exception:  # pragma: no cover
    dgl = None
    dgl_nn = None


class TemporalGNN(nn.Module):
    def __init__(
        self,
        entity_num: int,
        relation_num: int,
        time_num: int,
        embed_dim: int = 128,
        gnn_layers: int = 3,
        decay_rate: float = 0.08,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.entity_num = entity_num
        self.relation_num = relation_num
        self.time_num = time_num
        self.embed_dim = embed_dim
        self.decay_rate = decay_rate
        self.entity_embedding = nn.Embedding(entity_num, embed_dim)
        self.relation_embedding = nn.Embedding(relation_num, embed_dim)
        self.time_embedding = nn.Embedding(max(1, time_num), embed_dim)
        self.time_fusion = nn.Linear(2 * embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        self._query_time = None
        self.predictor = nn.Sequential(
            nn.Linear(2 * embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, 1),
        )

        nn.init.xavier_uniform_(self.entity_embedding.weight)
        nn.init.xavier_uniform_(self.relation_embedding.weight)
        nn.init.xavier_uniform_(self.time_embedding.weight)

        self.rgcn_layers = nn.ModuleList()
        self.fallback_layers = nn.ModuleList()
        if dgl_nn is not None:
            for _ in range(gnn_layers):
                self.rgcn_layers.append(
                    dgl_nn.RelGraphConv(
                        embed_dim,
                        embed_dim,
                        relation_num,
                        regularizer="basis",
                        num_bases=min(10, relation_num),
                        self_loop=True,
                    )
                )
        else:
            # Keep graph message passing available when the optional DGL
            # extension cannot be imported on a local CPU/PyTorch build.
            for _ in range(gnn_layers):
                self.fallback_layers.append(nn.Linear(embed_dim, embed_dim))

    def time_decay(self, time_delta: torch.Tensor) -> torch.Tensor:
        return torch.exp(-self.decay_rate * torch.abs(time_delta.float()))

    def forward(self, graph, entity_ids: torch.Tensor, rel_ids: torch.Tensor, time_ids: torch.Tensor) -> torch.Tensor:
        entity_ids = entity_ids.long()
        time_ids = time_ids.long().clamp(min=0, max=max(0, self.time_num - 1))
        e_emb = self.entity_embedding(entity_ids)
        t_emb = self.time_embedding(time_ids)
        h = self.dropout(F.relu(self.time_fusion(torch.cat([e_emb, t_emb], dim=-1))))

        if graph is not None and (self.rgcn_layers or self.fallback_layers):
            edge_rel = graph.edata.get("rel", rel_ids).long()
            if self.rgcn_layers and hasattr(graph, "local_scope"):
                for layer in self.rgcn_layers:
                    h = self.dropout(F.relu(layer(graph, h, edge_rel)))
            elif hasattr(graph, "edges"):
                src, dst = graph.edges()
                src, dst = src.long(), dst.long()
                if edge_rel.numel() != src.numel():
                    edge_rel = torch.zeros_like(src)
                layers = self.fallback_layers or [nn.Identity()]
                for layer in layers:
                    messages = h[src] + self.relation_embedding(
                        edge_rel.clamp(min=0, max=self.relation_num - 1)
                    )
                    edge_time = graph.edata.get("time")
                    if edge_time is not None and edge_time.numel() == src.numel() and self._query_time is not None:
                        decay = self.time_decay(
                            torch.as_tensor(self._query_time, device=h.device) - edge_time.to(h.device)
                        ).unsqueeze(-1)
                        messages = messages * decay
                    aggregate = torch.zeros_like(h)
                    aggregate.index_add_(0, dst, messages)
                    degree = torch.bincount(dst, minlength=h.shape[0]).clamp_min(1)
                    mixed = 0.5 * h + 0.5 * aggregate / degree.unsqueeze(-1)
                    h = mixed if isinstance(layer, nn.Identity) else self.dropout(F.relu(layer(mixed)))
        return h

    def score_candidates(
        self,
        graph,
        entity_ids: torch.Tensor,
        rel_id: int,
        time_id: int,
        head_id: int,
        candidate_ids: torch.Tensor,
    ) -> torch.Tensor:
        time_ids = torch.full_like(entity_ids, int(time_id))
        self._query_time = int(time_id)
        entity_emb = self.forward(graph, entity_ids, torch.empty(0, dtype=torch.long, device=entity_ids.device), time_ids)
        head_emb = entity_emb[int(head_id)].expand(candidate_ids.shape[0], -1)
        tail_emb = entity_emb[candidate_ids.long()]
        relation_emb = self.relation_embedding(
            torch.full((candidate_ids.shape[0],), int(rel_id), dtype=torch.long, device=candidate_ids.device)
        )
        raw = (head_emb * relation_emb * tail_emb).sum(dim=-1)
        return torch.sigmoid(raw)

    def predict_entity_score(
        self,
        graph,
        entity_ids: torch.Tensor,
        rel_ids: torch.Tensor,
        time_ids: torch.Tensor,
        target_entity: int,
        head_entity: Optional[int] = None,
        relation_id: Optional[int] = None,
        time_id: Optional[int] = None,
    ) -> float:
        # Backward-compatible scalar API used by auxiliary scripts.
        if relation_id is not None and head_entity is not None and time_id is not None:
            scores = self.score_candidates(
                graph,
                entity_ids,
                relation_id,
                time_id,
                head_entity,
                torch.tensor([target_entity], device=entity_ids.device),
            )
            return float(scores[0].item())

        time_ids = time_ids.long()
        if time_ids.numel() != entity_ids.numel():
            time_ids = torch.zeros_like(entity_ids)
        entity_emb = self.forward(graph, entity_ids, rel_ids, time_ids)
        target_emb = entity_emb[int(target_entity)]
        return float(torch.sigmoid(self.predictor(torch.cat([target_emb, target_emb]))).item())
