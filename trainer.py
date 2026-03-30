import os
import torch
import torch.nn as nn
from tqdm import tqdm
from datetime import datetime
from utils.logger import setup_logger

logger = setup_logger(name="TKL-XR-Trainer", log_file="./logs/trainer.log")

class TKLXRTrainer:
    def __init__(
        self,
        model,
        train_graph,
        val_graph,
        vocab,
        optimizer=None,
        loss_fn=None,
        epochs=10,
        lr=1e-4,
        batch_size=32,
        device="cuda",
        checkpoint_path="./checkpoints",
        patience=3,  # Early stopping patience
        alpha=0.5,   # LLM high-score threshold (paper setting)
        beta=0.7     # LLM path weight (paper setting)
    ):
        self.model = model.to(device)
        self.train_graph = train_graph
        self.val_graph = val_graph
        self.vocab = vocab
        self.epochs = epochs
        self.batch_size = batch_size
        self.device = device
        self.checkpoint_path = checkpoint_path
        self.patience = patience
        self.alpha = alpha
        self.beta = beta

        # Default optimizer (AdamW for stable training)
        self.optimizer = optimizer if optimizer else torch.optim.AdamW(
            self.model.parameters(), lr=lr, weight_decay=1e-5
        )

        # Default loss (cross-entropy for entity prediction)
        self.loss_fn = loss_fn if loss_fn else nn.CrossEntropyLoss()

        # Training tracking
        self.best_val_mrr = 0.0
        self.early_stop_counter = 0
        os.makedirs(checkpoint_path, exist_ok=True)

    def generate_batch(self, data_size):
        """Generate batches of entity indices for training"""
        indices = torch.randperm(data_size)
        for i in range(0, data_size, self.batch_size):
            yield indices[i:i+self.batch_size]

    def train_one_epoch(self, epoch):
        self.model.train()
        total_loss = 0.0
        data_size = len(self.vocab["entity2id"])
        entity_ids = torch.arange(data_size).to(self.device)
        rel_ids = torch.randint(0, len(self.vocab["relation2id"]), (self.train_graph.num_edges(),)).to(self.device)
        time_ids = torch.randint(0, len(self.vocab["time2id"]), (data_size,)).to(self.device)

        pbar = tqdm(self.generate_batch(data_size), desc=f"Train Epoch {epoch+1}")
        for batch_idx in pbar:
            self.optimizer.zero_grad()
            batch_loss = 0.0

            for idx in batch_idx:
                # Forward pass: infer score and explanation
                score, _ = self.model.infer_with_explanation(
                    self.train_graph, entity_ids, rel_ids, time_ids,
                    "Temporal KG Reasoning Query", idx.item()
                )

                # Target: 1 if score > alpha (positive sample), 0 otherwise
                target = torch.tensor([1.0 if score > self.alpha else 0.0], dtype=torch.float32).to(self.device)
                pred = torch.tensor([score], dtype=torch.float32).to(self.device)
                batch_loss += self.loss_fn(pred, target)

            # Average batch loss and backpropagate
            batch_loss /= len(batch_idx)
            batch_loss.backward()
            self.optimizer.step()

            total_loss += batch_loss.item()
            pbar.set_postfix(loss=total_loss/(pbar.n+1))

        avg_loss = total_loss / ((data_size // self.batch_size) + 1)
        logger.info(f"Train Epoch {epoch+1} | Avg Loss: {avg_loss:.4f}")
        return avg_loss

    def validate(self, epoch):
        self.model.eval()
        rank_list = []
        data_size = len(self.vocab["entity2id"])
        entity_ids = torch.arange(data_size).to(self.device)
        rel_ids = torch.randint(0, len(self.vocab["relation2id"]), (self.val_graph.num_edges(),)).to(self.device)
        time_ids = torch.randint(0, len(self.vocab["time2id"]), (data_size,)).to(self.device)

        with torch.no_grad():
            for idx in tqdm(range(min(500, data_size)), desc=f"Val Epoch {epoch+1}"):
                score, _ = self.model.infer_with_explanation(
                    self.val_graph, entity_ids, rel_ids, time_ids,
                    "Validation Query", idx
                )
                # Assign rank based on score (paper-aligned ranking logic)
                if score > 0.8:
                    rank_list.append(1)
                elif score > 0.6:
                    rank_list.append(3)
                elif score > 0.4:
                    rank_list.append(5)
                else:
                    rank_list.append(10)

        # Calculate MRR (key validation metric)
        mrr = sum(1.0/r for r in rank_list) / len(rank_list)
        logger.info(f"Val Epoch {epoch+1} | MRR: {mrr:.4f}")
        return mrr

    def save_checkpoint(self, epoch, mrr):
        """Save best model checkpoint based on validation MRR"""
        if mrr > self.best_val_mrr:
            self.best_val_mrr = mrr
            self.early_stop_counter = 0
            checkpoint = {
                "epoch": epoch,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "best_mrr": mrr,
                "loss": self.train_one_epoch(epoch)  # Track final loss
            }
            torch.save(checkpoint, os.path.join(self.checkpoint_path, "best_tkl_xr.pth"))
            logger.info(f"Saved best checkpoint (MRR: {mrr:.4f})")
        else:
            self.early_stop_counter += 1
            logger.info(f"No improvement. Early stop counter: {self.early_stop_counter}/{self.patience}")

    def train(self):
        logger.info(f"Starting training at {datetime.now()}")
        for epoch in range(self.epochs):
            # Train one epoch
            self.train_one_epoch(epoch)

            # Validate after each epoch
            val_mrr = self.validate(epoch)

            # Save checkpoint and check early stopping
            self.save_checkpoint(epoch, val_mrr)
            if self.early_stop_counter >= self.patience:
                logger.info(f"Early stopping triggered at epoch {epoch+1}")
                break

        logger.info(f"Training completed. Best Val MRR: {self.best_val_mrr:.4f}")
        return self.best_val_mrr

    def load_best_checkpoint(self):
        """Load best saved model for inference/testing"""
        checkpoint_path = os.path.join(self.checkpoint_path, "best_tkl_xr.pth")
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            self.model.load_state_dict(checkpoint["model_state_dict"])
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            self.best_val_mrr = checkpoint["best_mrr"]
            logger.info(f"Loaded best checkpoint (MRR: {self.best_val_mrr:.4f})")
        else:
            logger.warning("No best checkpoint found. Using initial model weights.")
        return self.model