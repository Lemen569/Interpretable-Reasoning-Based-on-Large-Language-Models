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


logger = setup_logger(name="TKL-XR-Sensitivity", log_file="./logs/sensitivity_experiment.log")

PARAMS = {
    "beam_depth": [1, 2, 3, 4, 5, 6],
    "beam_width": [1, 2, 3, 4, 5, 6],
    "decay_rate": [0.01, 0.05, 0.08, 0.12, 0.18],
    "gnn_layers": [1, 2, 3, 4, 5],
    "alpha": [0.3, 0.4, 0.5, 0.6, 0.7],
    "beta": [0.5, 0.6, 0.7, 0.8, 0.9],
}


def parse_args():
    parser = argparse.ArgumentParser(description="TKL-XR sensitivity analysis")
    parser.add_argument("--dataset", default="ICEWS18")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--save_path", default="./results/sensitivity")
    parser.add_argument("--checkpoint_path", default="./checkpoints/sensitivity")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--raw_data_path", default="./data")
    parser.add_argument("--processed_path", default="./data/processed")
    parser.add_argument("--params", nargs="+", default=None)
    parser.add_argument("--max_eval_samples", type=int, default=100)
    parser.add_argument("--use_mock_llm", action="store_true")
    return parser.parse_args()


def run(args):
    set_seed(args.seed)
    if DEVICE == "cuda":
        torch.cuda.set_device(args.gpu)
    dataset = load_dataset(args.dataset, args.raw_data_path, args.processed_path)
    selected = args.params or list(PARAMS)
    results = {}

    for parameter in selected:
        if parameter not in PARAMS:
            logger.warning("Skipping unknown parameter %s", parameter)
            continue
        results[parameter] = {}
        for value in PARAMS[parameter]:
            kwargs = {
                "entity_num": dataset.entity_num,
                "relation_num": dataset.relation_num,
                "time_num": dataset.time_num,
                "device": DEVICE,
                "load_4bit": not args.use_mock_llm,
                "use_mock_llm": args.use_mock_llm,
                "beam_depth": 4,
                "beam_width": 4,
                "decay_rate": 0.08,
                "gnn_layers": 3,
                "alpha": 0.5,
                "beta": 0.7,
            }
            kwargs[parameter] = value
            model = TKLXR(**kwargs).to(DEVICE)
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
                checkpoint_path=os.path.join(
                    args.checkpoint_path, args.dataset, parameter, str(value)
                ),
                seed=args.seed,
            )
            trainer.train()
            trainer.load_best_checkpoint()
            eval_args = argparse.Namespace(
                beam_depth=kwargs["beam_depth"],
                beam_width=kwargs["beam_width"],
                retrieval_rounds=1,
                top_relations=10,
                top_entities=10,
                candidate_mode="all",
                max_eval_samples=args.max_eval_samples,
            )
            results[parameter][str(value)] = evaluate(model, dataset, eval_args)

    os.makedirs(args.save_path, exist_ok=True)
    with open(os.path.join(args.save_path, f"sensitivity_results_{args.dataset}.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    return results


if __name__ == "__main__":
    run(parse_args())
