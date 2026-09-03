from __future__ import annotations

import argparse
import json
import os

import torch

from main import DEVICE, evaluate, set_seed
from models.tkl_xr import TKLXR
from modules.data_process import load_dataset
from utils.logger import setup_logger
from trainer import TKLXRTrainer


logger = setup_logger(name="TKL-XR-Generalization", log_file="./logs/generalization.log")

TARGET_DATASETS = ["ICEWS18", "MIMIC-III", "MIMIC-IV", "WIKI", "YAGO", "GDELT"]


def parse_args():
    parser = argparse.ArgumentParser(description="Cross-domain generalization evaluation")
    parser.add_argument("--source_dataset", default="ICEWS18")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save_path", default="./results/generalization")
    parser.add_argument("--checkpoint_path", default="./checkpoints")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--raw_data_path", default="./data")
    parser.add_argument("--processed_path", default="./data/processed")
    parser.add_argument("--max_eval_samples", type=int, default=200)
    parser.add_argument("--use_mock_llm", action="store_true")
    return parser.parse_args()


def run(args):
    set_seed(args.seed)
    if DEVICE == "cuda":
        torch.cuda.set_device(args.gpu)

    source = load_dataset(args.source_dataset, args.raw_data_path, args.processed_path)
    source_model = TKLXR(
        source.entity_num,
        source.relation_num,
        source.time_num,
        device=DEVICE,
        load_4bit=not args.use_mock_llm,
        use_mock_llm=args.use_mock_llm,
    ).to(DEVICE)
    source_trainer = TKLXRTrainer(
        model=source_model,
        train_graph=source.time_graphs,
        val_graph=source.time_graphs,
        vocab=source.vocab,
        train_quads=source.train,
        val_quads=source.valid,
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        device=DEVICE,
        checkpoint_path=os.path.join(args.checkpoint_path, args.source_dataset),
        seed=args.seed,
    )
    checkpoint_file = os.path.join(
        args.checkpoint_path, args.source_dataset, "best_tkl_xr.pth"
    )
    if os.path.exists(checkpoint_file):
        source_trainer.load_best_checkpoint()
    else:
        source_trainer.train()
        source_trainer.load_best_checkpoint()

    results = {}
    for target_name in TARGET_DATASETS:
        if target_name == args.source_dataset:
            continue
        try:
            target = load_dataset(target_name, args.raw_data_path, args.processed_path)
        except FileNotFoundError:
            logger.warning("Skipping %s because its data is unavailable.", target_name)
            continue
        target_model = TKLXR(
            target.entity_num,
            target.relation_num,
            target.time_num,
            device=DEVICE,
            load_4bit=not args.use_mock_llm,
            use_mock_llm=args.use_mock_llm,
        ).to(DEVICE)
        source_state = source_model.state_dict()
        target_state = target_model.state_dict()
        compatible = {
            key: value
            for key, value in source_state.items()
            if key in target_state and target_state[key].shape == value.shape
        }
        target_state.update(compatible)
        target_model.load_state_dict(target_state)
        results[f"{args.source_dataset}_to_{target_name}"] = evaluate(
            target_model,
            target,
            argparse.Namespace(
                beam_depth=4,
                beam_width=4,
                retrieval_rounds=1,
                top_relations=10,
                top_entities=10,
                candidate_mode="all",
                max_eval_samples=args.max_eval_samples,
            ),
        )

    os.makedirs(args.save_path, exist_ok=True)
    with open(os.path.join(args.save_path, "generalization_results.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    return results


if __name__ == "__main__":
    run(parse_args())
