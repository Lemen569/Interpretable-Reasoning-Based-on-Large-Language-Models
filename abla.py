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


logger = setup_logger(name="TKL-XR-Ablation", log_file="./logs/ablation_study.log")

ABLATION_SETTINGS = {
    "full_model": dict(enable_gnn=True, enable_llm=True, enable_fusion=True, enable_htir=True, enable_bidirectional=True),
    "wo_gnn": dict(enable_gnn=False, enable_llm=True, enable_fusion=True, enable_htir=True, enable_bidirectional=True),
    "wo_llm": dict(enable_gnn=True, enable_llm=False, enable_fusion=True, enable_htir=True, enable_bidirectional=True),
    "wo_fusion": dict(enable_gnn=True, enable_llm=True, enable_fusion=False, enable_htir=True, enable_bidirectional=True),
    "wo_htir": dict(enable_gnn=True, enable_llm=True, enable_fusion=True, enable_htir=False, enable_bidirectional=True),
    "wo_bidirectional": dict(enable_gnn=True, enable_llm=True, enable_fusion=True, enable_htir=True, enable_bidirectional=False),
    "w_linear_fusion": dict(enable_gnn=True, enable_llm=True, enable_fusion=True, enable_htir=True, enable_bidirectional=True, linear_fusion=True),
}


def parse_args():
    parser = argparse.ArgumentParser(description="TKL-XR ablation study")
    parser.add_argument("--dataset", default="ICEWS18")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--save_path", default="./results/ablation")
    parser.add_argument("--checkpoint_path", default="./checkpoints/ablation")
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
    dataset = load_dataset(args.dataset, args.raw_data_path, args.processed_path)
    results = {}

    for name, switches in ABLATION_SETTINGS.items():
        logger.info("Running %s", name)
        model = TKLXR(
            entity_num=dataset.entity_num,
            relation_num=dataset.relation_num,
            time_num=dataset.time_num,
            device=DEVICE,
            load_4bit=not args.use_mock_llm,
            use_mock_llm=args.use_mock_llm,
            **switches,
        ).to(DEVICE)
        trainer = TKLXRTrainer(
            model=model,
            train_graph=dataset.time_graphs,
            val_graph=dataset.time_graphs,
            vocab=dataset.vocab,
            train_quads=dataset.train,
            val_quads=dataset.valid,
            epochs=args.epochs,
            lr=args.lr,
            batch_size=args.batch_size,
            device=DEVICE,
            checkpoint_path=os.path.join(args.checkpoint_path, args.dataset, name),
            seed=args.seed,
        )
        trainer.train()
        trainer.load_best_checkpoint()
        results[name] = evaluate(
            model,
            dataset,
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
    output = os.path.join(args.save_path, f"ablation_results_{args.dataset}.json")
    with open(output, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    return results


if __name__ == "__main__":
    run(parse_args())
