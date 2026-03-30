import os
import json
import torch
import argparse
import time
import psutil
from datetime import datetime
from tqdm import tqdm
from utils.logger import setup_logger
from models.tkl_xr import TKLXR

logger = setup_logger(name="TKL-XR-Runtime", log_file="./logs/runtime.log")
device = "cuda" if torch.cuda.is_available() else "cpu"

MODELS = {"TKL-XR": {}, "TKL-XR*": {}, "TiRGN+CoH": {}, "RE-GCN+CoH": {}}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="ICEWS05-15")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--batch_sizes", nargs="+", type=int, default=[8, 16, 32, 64])
    parser.add_argument("--save_path", default="./results/runtime")
    return parser.parse_args()


def mem_usage():
    return torch.cuda.memory_allocated() / 1024 / 1024 if torch.cuda.is_available() else 0


def build(model_name):
    return TKLXR(1000, 100, 50, load_4bit=(model_name == "TKL-XR*")).to(device)


def test(model, name, bs, vocab):
    for _ in range(5):
        model.infer_with_explanation(None, torch.arange(1000).to(device), torch.randint(0, 100, (1000,)).to(device),
                                     torch.randint(0, 50, (1000,)).to(device), "Q", 0)

    t0 = time.time()
    for _ in tqdm(range(50), leave=False):
        for _ in range(bs):
            model.infer_with_explanation(None, torch.arange(1000).to(device), torch.randint(0, 100, (1000,)).to(device),
                                         torch.randint(0, 50, (1000,)).to(device), "Q", 0)
    t_total = time.time() - t0
    return {"latency": t_total / (50 * bs), "throughput": (50 * bs) / t_total, "gpu_mem": mem_usage()}


def run(args):
    vocab = {"entity2id": {str(i): i for i in range(1000)}}
    all_res = {}
    os.makedirs(args.save_path, exist_ok=True)

    for name in MODELS:
        model = build(name)
        model_res = {}
        for bs in args.batch_sizes:
            model_res[bs] = test(model, name, bs, vocab)
        all_res[name] = model_res

    with open(f"{args.save_path}/res.json", "w") as f:
        json.dump(all_res, f, indent=4)


if __name__ == "__main__":
    args = parse_args()
    torch.cuda.set_device(args.gpu)
    run(args)