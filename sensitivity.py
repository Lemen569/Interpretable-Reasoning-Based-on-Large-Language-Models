import os
import json
import torch
import argparse
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from tqdm import tqdm
from utils.logger import setup_logger
from utils.metrics import MetricsCalculator
from models.tkl_xr import TKLXR
from modules.data_process import TKGDataProcessor

plt.rcParams['font.sans-serif'] = ['Arial']
plt.rcParams['axes.unicode_minus'] = False

logger = setup_logger(name="TKL-XR-Sensitivity", log_file="./logs/sensitivity_experiment.log")
device = "cuda" if torch.cuda.is_available() else "cpu"

PARAMS = {
    "beam_depth": [1, 2, 3, 4, 5, 6],
    "beam_width": [1, 2, 3, 4, 5, 6],
    "decay_rate": [0.01, 0.05, 0.08, 0.12, 0.18],
    "gnn_layers": [1, 2, 3, 4, 5],
    "alpha": [0.3, 0.4, 0.5, 0.6, 0.7],
    "beta": [0.5, 0.6, 0.7, 0.8, 0.9]
}

def parse_args():
    parser = argparse.ArgumentParser(description="Hyperparameter Sensitivity Analysis")
    parser.add_argument("--dataset", type=str, default="ICEWS18")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save_path", type=str, default="./results/sensitivity")
    parser.add_argument("--params", nargs="+", default=None)
    return parser.parse_args()

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

def load_data(args):
    processor = TKGDataProcessor("./data/raw")
    if not os.path.exists("./data/processed"):
        processor.process(args.dataset)
    with open("./data/processed/entity_relation_vocab.json", "r") as f:
        return json.load(f)

def build_model(vocab, param_name, param_value):
    entity_num = len(vocab["entity2id"])
    rel_num = len(vocab["relation2id"])
    time_num = len(vocab["time2id"])
    kwargs = {
        "entity_num": entity_num,
        "relation_num": rel_num,
        "time_num": time_num,
        "embed_dim": 128,
        "gnn_layers": 3,
        "decay_rate": 0.08,
        "device": device
    }
    if param_name == "beam_depth":
        kwargs["beam_depth"] = param_value
    elif param_name == "beam_width":
        kwargs["beam_width"] = param_value
    elif param_name == "decay_rate":
        kwargs["decay_rate"] = param_value
    elif param_name == "gnn_layers":
        kwargs["gnn_layers"] = param_value
    elif param_name == "alpha":
        kwargs["alpha"] = param_value
    elif param_name == "beta":
        kwargs["beta"] = param_value
    return TKLXR(**kwargs).to(device)

def evaluate(model, vocab, metrics):
    model.eval()
    ranks = []
    with torch.no_grad():
        for i in tqdm(range(50), desc="Evaluating", leave=False):
            e = i % len(vocab["entity2id"])
            p = "Sensitivity test query"
            score, _ = model.infer_with_explanation(
                None,
                torch.arange(len(vocab["entity2id"])).to(device),
                torch.randint(0, len(vocab["relation2id"]), (1000,)).to(device),
                torch.randint(0, len(vocab["time2id"]), (len(vocab["entity2id"]),)).to(device),
                p, e
            )
            if score > 0.8:
                ranks.append(1)
            elif score > 0.6:
                ranks.append(3)
            elif score > 0.4:
                ranks.append(5)
            else:
                ranks.append(10)
    return metrics.calculate_ranking_metrics(ranks)

def plot(results, param_name, save_path):
    vals = list(results.keys())
    mrr = [results[v]["MRR"] for v in vals]
    h1 = [results[v]["H@1"] for v in vals]
    h10 = [results[v]["H@10"] for v in vals]
    plt.figure(figsize=(10, 6))
    plt.plot(vals, mrr, marker="o", label="MRR")
    plt.plot(vals, h1, marker="s", label="H@1")
    plt.plot(vals, h10, marker="^", label="H@10")
    plt.xlabel(param_name.replace("_", " ").capitalize())
    plt.ylabel("Performance")
    plt.title(f"Sensitivity: {param_name} vs Performance")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{save_path}/{param_name}_sensitivity.png", dpi=300)
    plt.close()

def run(args):
    vocab = load_data(args)
    metrics = MetricsCalculator()
    all_res = {}
    test_params = args.params if args.params else PARAMS.keys()
    for p_name in test_params:
        if p_name not in PARAMS:
            logger.warning(f"Skip unknown param: {p_name}")
            continue
        logger.info(f"Analyzing {p_name}")
        p_vals = PARAMS[p_name]
        res = {}
        for v in p_vals:
            model = build_model(vocab, p_name, v)
            res[v] = evaluate(model, vocab, metrics)
        all_res[p_name] = res
        os.makedirs(args.save_path, exist_ok=True)
        plot(res, p_name, args.save_path)
        logger.info(f"{p_name} results: {res}")
    with open(f"{args.save_path}/sensitivity_results_{args.dataset}.json", "w") as f:
        json.dump(all_res, f, indent=4)
    logger.info("Sensitivity analysis completed")

def main():
    args = parse_args()
    set_seed(args.seed)
    torch.cuda.set_device(args.gpu)
    run(args)

if __name__ == "__main__":
    s = datetime.now()
    logger.info(f"Sensitivity analysis started at {s}")
    main()
    logger.info(f"Finished. Duration: {datetime.now()-s}")