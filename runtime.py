from __future__ import annotations

import argparse
import json
import os
import time

import torch
from tqdm import tqdm

from models.tkl_xr import TKLXR
from main import _direction_scores
from modules.data_process import load_dataset
from modules.retrieval import HTIRRetriever
from utils.logger import setup_logger


logger = setup_logger(name="TKL-XR-Runtime", log_file="./logs/runtime.log")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# Profile only variants implemented in this repository. External baselines
# require their own released implementations and are not relabeled here.
MODELS = ("TKL-XR", "TKL-XR-linear", "TKL-XR-no-HTIR")


def parse_args():
    parser = argparse.ArgumentParser(description="Runtime profiling")
    parser.add_argument("--dataset", default="ICEWS05-15")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--batch_sizes", nargs="+", type=int, default=[8, 16, 32, 64])
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--save_path", default="./results/runtime")
    parser.add_argument("--raw_data_path", default="./data")
    parser.add_argument("--processed_path", default="./data/processed")
    parser.add_argument("--use_mock_llm", action="store_true")
    return parser.parse_args()


def mem_usage() -> float:
    if not torch.cuda.is_available():
        return 0.0
    return float(torch.cuda.max_memory_allocated() / 1024**3)


def build(name: str, dataset, use_mock_llm: bool):
    switches = {
        "linear_fusion": name == "TKL-XR-linear",
        "enable_htir": name != "TKL-XR-no-HTIR",
    }
    return TKLXR(
        dataset.entity_num,
        dataset.relation_num,
        dataset.time_num,
        device=DEVICE,
        load_4bit=not use_mock_llm,
        use_mock_llm=use_mock_llm,
        **switches,
    ).to(DEVICE)


def test(model, dataset, batch_size: int, repeats: int):
    if not dataset.test:
        return {"latency": 0.0, "throughput": 0.0, "gpu_mem": 0.0}
    quad = dataset.test[0]
    retriever = HTIRRetriever(
        model.llm, beam_depth=model.beam_depth, beam_width=model.beam_width
    )

    for _ in range(2):
        _direction_scores(model, retriever, dataset, quad, candidate_mode="all")
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    with torch.no_grad():
        for _ in tqdm(range(repeats), leave=False):
            for _ in range(batch_size):
                _direction_scores(model, retriever, dataset, quad, candidate_mode="all")
    elapsed = time.perf_counter() - started
    total = repeats * batch_size
    return {
        "latency": elapsed / total,
        "throughput": total / max(elapsed, 1e-12),
        "gpu_mem": mem_usage(),
    }


def run(args):
    if DEVICE == "cuda":
        torch.cuda.set_device(args.gpu)
    dataset = load_dataset(args.dataset, args.raw_data_path, args.processed_path)
    results = {}
    for name in MODELS:
        model = build(name, dataset, args.use_mock_llm)
        results[name] = {
            str(batch_size): test(model, dataset, batch_size, args.repeats)
            for batch_size in args.batch_sizes
        }
    os.makedirs(args.save_path, exist_ok=True)
    with open(os.path.join(args.save_path, "res.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    logger.info("Runtime results written to %s", args.save_path)
    return results


if __name__ == "__main__":
    run(parse_args())
