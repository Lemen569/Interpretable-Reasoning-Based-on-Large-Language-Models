from __future__ import annotations

import os
from datetime import datetime
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
from tqdm import tqdm

from utils.logger import setup_logger


logger = setup_logger(name="TKL-XR-Trainer", log_file="./logs/trainer.log")


class TKLXRTrainer:
    """Supervised trainer for temporal entity prediction.

    The training objective is applied to observed quadruples with one sampled
    negative tail.  Validation ranks the observed tail against all entities.
    """

    def __init__(
        self,
        model,
        train_graph=None,
        val_graph=None,
        vocab: Optional[dict] = None,
        train_quads: Optional[Sequence[Tuple[int, int, int, int]]] = None,
        val_quads: Optional[Sequence[Tuple[int, int, int, int]]] = None,
        optimizer=None,
        loss_fn=None,
        epochs: int = 10,
        lr: float = 2e-5,
        batch_size: int = 32,
        device: str = "cuda",
        checkpoint_path: str = "./checkpoints",
        patience: int = 3,
        seed: int = 42,
    ):
        self.model = model.to(device)
        self.train_graph = train_graph
        self.val_graph = val_graph if val_graph is not None else train_graph
        self.vocab = vocab or {}
        self.train_quads = list(train_quads or [])
        self.val_quads = list(val_quads or [])
        self.epochs = int(epochs)
        self.batch_size = int(batch_size)
        self.device = device
        self.checkpoint_path = checkpoint_path
        self.patience = int(patience)
        self.generator = torch.Generator(device="cpu")
        self.generator.manual_seed(int(seed))
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = optimizer
        if self.optimizer is None and trainable_params:
            self.optimizer = torch.optim.AdamW(
                trainable_params,
                lr=lr,
                weight_decay=1e-5,
            )
        # The paper uses a cross-entropy objective with one positive and five
        # randomly sampled negative entities for each query.
        self.loss_fn = loss_fn or nn.CrossEntropyLoss()
        total_steps = max(
            1,
            self.epochs
            * ((len(self.train_quads) + max(self.batch_size, 1) - 1) // max(self.batch_size, 1)),
        )
        warmup_steps = max(1, int(round(0.1 * total_steps)))
        self.scheduler = None
        if self.optimizer is not None:
            final_lr_ratio = min(1.0, 1e-6 / max(float(lr), 1e-12))

            def lr_lambda(step):
                if step < warmup_steps:
                    return max((step + 1) / warmup_steps, 1e-8)
                progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
                return 1.0 - progress * (1.0 - final_lr_ratio)

            self.scheduler = torch.optim.lr_scheduler.LambdaLR(
                self.optimizer, lr_lambda=lr_lambda
            )
        self.best_val_mrr = -1.0
        self.early_stop_counter = 0
        os.makedirs(checkpoint_path, exist_ok=True)

    def _graph_for_time(self, graph, time_id: int):
        if isinstance(graph, dict):
            if time_id in graph:
                return graph[time_id]
            keys = sorted(graph)
            return graph[max((key for key in keys if key <= time_id), default=keys[0])]
        return graph

    def _components(self, quad):
        head_id, relation_id, tail_id, time_id = map(int, quad)
        entity_ids = torch.arange(
            len(self.vocab["entity2id"]),
            dtype=torch.long,
            device=self.device,
        )
        relation_ids = torch.empty(0, dtype=torch.long, device=self.device)
        time_ids = torch.full_like(entity_ids, time_id)
        return head_id, relation_id, tail_id, time_id, entity_ids, relation_ids, time_ids

    def train_one_epoch(self, epoch: int) -> float:
        if not self.train_quads or getattr(self.model, "gnn", None) is None:
            logger.warning("No train quadruples or GNN component; skipping gradient training.")
            return 0.0

        self.model.train()
        order = torch.randperm(len(self.train_quads), generator=self.generator).tolist()
        total_loss = 0.0
        steps = 0
        entity_count = len(self.vocab["entity2id"])

        for start in tqdm(range(0, len(order), self.batch_size), desc=f"Train Epoch {epoch + 1}"):
            batch_indices = order[start : start + self.batch_size]
            if self.optimizer is None:
                return 0.0
            self.optimizer.zero_grad()
            losses = []
            for index in batch_indices:
                quad = self.train_quads[index]
                head_id, relation_id, tail_id, time_id, entity_ids, relation_ids, time_ids = self._components(quad)
                negative_tails = []
                if entity_count > 1:
                    # Sample five true negatives and avoid presenting the
                    # positive tail twice with contradictory labels.
                    sampled = set()
                    while len(sampled) < min(5, entity_count - 1):
                        value = int(
                            torch.randint(
                                entity_count - 1, (1,), generator=self.generator
                            ).item()
                        )
                        if value >= tail_id:
                            value += 1
                        sampled.add(value)
                    negative_tails = sorted(sampled)
                else:
                    negative_tails = [tail_id]
                graph = self._graph_for_time(self.train_graph, time_id)
                candidate_ids = torch.tensor(
                    [tail_id] + negative_tails, dtype=torch.long, device=self.device
                )
                if getattr(self.model, "enable_fusion", False) and getattr(self.model, "enable_llm", False):
                    # The LLM score is a frozen semantic signal, while the
                    # fusion encoder and GNN remain trainable.
                    positive_prompt = f"Temporal query head={head_id} relation={relation_id} time={time_id}"
                    llm_scores = torch.tensor(
                        [
                            float(self.model.llm.score(f"{positive_prompt}\nCandidate entity: {int(candidate)}"))
                            if self.model.llm is not None else 0.0
                            for candidate in candidate_ids.tolist()
                        ],
                        dtype=torch.float32,
                        device=self.device,
                    )
                    gnn_scores = self.model.gnn.score_candidates(
                        graph, entity_ids, relation_id, time_id, head_id, candidate_ids
                    )
                    fused_scores = self.model.fuse_scores_tensor(llm_scores, gnn_scores)
                    logits = torch.logit(fused_scores.clamp(1e-6, 1 - 1e-6))
                else:
                    scores = self.model.gnn.score_candidates(
                        graph, entity_ids, relation_id, time_id, head_id, candidate_ids
                    )
                    logits = torch.logit(scores.clamp(1e-6, 1 - 1e-6))
                # The positive entity is at index zero.
                label = torch.zeros((), dtype=torch.long, device=self.device)
                losses.append(self.loss_fn(logits.unsqueeze(0), label.unsqueeze(0)))
            if not losses:
                continue
            loss = torch.stack(losses).mean()
            loss.backward()
            self.optimizer.step()
            if self.scheduler is not None:
                self.scheduler.step()
            total_loss += float(loss.item())
            steps += 1

        average = total_loss / max(steps, 1)
        logger.info("Train Epoch %s | Avg Loss: %.4f", epoch + 1, average)
        return average

    def _rank_quad(self, quad, graph) -> int:
        head_id, relation_id, tail_id, time_id, entity_ids, _, _ = self._components(quad)
        candidate_ids = entity_ids
        if getattr(self.model, "gnn", None) is None:
            return 1
        scores = self.model.gnn.score_candidates(
            graph,
            entity_ids,
            relation_id,
            time_id,
            head_id,
            candidate_ids,
        )
        target_score = float(scores[tail_id].item())
        return 1 + int((scores > target_score).sum().item())

    def validate(self, epoch: int) -> Dict[str, float]:
        self.model.eval()
        if not self.val_quads:
            return {"MRR": 0.0, "H@1": 0.0, "H@3": 0.0, "H@10": 0.0}
        ranks = []
        with torch.no_grad():
            for quad in tqdm(self.val_quads, desc=f"Val Epoch {epoch + 1}", leave=False):
                ranks.append(self._rank_quad(quad, self._graph_for_time(self.val_graph, int(quad[3]))))
        metrics = {
            "MRR": sum(1.0 / rank for rank in ranks) / len(ranks),
            "H@1": sum(rank <= 1 for rank in ranks) / len(ranks),
            "H@3": sum(rank <= 3 for rank in ranks) / len(ranks),
            "H@10": sum(rank <= 10 for rank in ranks) / len(ranks),
        }
        logger.info("Val Epoch %s | %s", epoch + 1, metrics)
        return metrics

    def save_checkpoint(self, epoch: int, metrics: Dict[str, float], loss: float):
        mrr = float(metrics.get("MRR", 0.0))
        if mrr <= self.best_val_mrr:
            self.early_stop_counter += 1
            return
        self.best_val_mrr = mrr
        self.early_stop_counter = 0
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict() if self.optimizer is not None else {},
                "metrics": metrics,
                "loss": loss,
            },
            os.path.join(self.checkpoint_path, "best_tkl_xr.pth"),
        )

    def train(self):
        logger.info("Starting training at %s", datetime.now())
        for epoch in range(self.epochs):
            loss = self.train_one_epoch(epoch)
            metrics = self.validate(epoch)
            self.save_checkpoint(epoch, metrics, loss)
            if self.early_stop_counter >= self.patience:
                break
        return self.best_val_mrr

    def load_best_checkpoint(self):
        path = os.path.join(self.checkpoint_path, "best_tkl_xr.pth")
        if not os.path.exists(path):
            logger.warning("No checkpoint found at %s; using current weights.", path)
            return self.model
        try:
            checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        except TypeError:  # older PyTorch versions
            checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        if self.optimizer is not None and "optimizer_state_dict" in checkpoint:
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.best_val_mrr = float(checkpoint.get("metrics", {}).get("MRR", checkpoint.get("best_mrr", 0.0)))
        return self.model
