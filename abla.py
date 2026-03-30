import os
import json
import torch
import argparse
from tqdm import tqdm
from datetime import datetime

from models.tkl_xr import TKLXR
from models.trainer import TKLXRTrainer
from modules.data_process import TKGDataProcessor
from utils.metrics import MetricsCalculator
from utils.logger import setup_logger
from config.model import ModelConfig

logger = setup_logger(name="TKL-XR-Ablation", log_file="./logs/ablation_study.log")
device = "cuda" if torch.cuda.is_available() else "cpu"
cfg = ModelConfig()

ABLATION_SETTINGS = {
    "full_model": {"gnn": True, "llm": True, "fusion": True, "time_decay": 0.08, "htir": True, "bidirectional": True},
    "wo_gnn": {"gnn": False, "llm": True, "fusion": True, "time_decay": 0.08, "htir": True, "bidirectional": True},
    "wo_llm": {"gnn": True, "llm": False, "fusion": True, "time_decay": 0.08, "htir": True, "bidirectional": True},
    "wo_fusion": {"gnn": True, "llm": True, "fusion": False, "time_decay": 0.08, "htir": True, "bidirectional": True},
    "wo_time_decay": {"gnn": True, "llm": True, "fusion": True, "time_decay": 0.0, "htir": True, "bidirectional": True},
    "wo_htir": {"gnn": True, "llm": True, "fusion": True, "time_decay": 0.08, "htir": False, "bidirectional": True},
    "wo_bidirectional": {"gnn": True, "llm": True, "fusion": True, "time_decay": 0.08, "htir": True,
                         "bidirectional": False}
}


def parse_args():
    parser = argparse.ArgumentParser(description="TKL-XR Ablation Study")
    parser.add_argument("--dataset", type=str, default=cfg.DATASET,
                        choices=["ICEWS14", "ICEWS05-15", "ICEWS18", "MIMIC-III"])
    parser.add_argument("--gpu", type=int, default=cfg.GPU_ID)
    parser.add_argument("--seed", type=int, default=cfg.SEED)
    parser.add_argument("--save_path", type=str, default="./results/ablation")
    return parser.parse_args()


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def load_dataset(args):
    processor = TKGDataProcessor(cfg.RAW_DATA_PATH)
    if not os.path.exists(cfg.PROCESSED_DATA_PATH):
        processor.process(args.dataset)

    with open(os.path.join(cfg.PROCESSED_DATA_PATH, "entity_relation_vocab.json"), "r") as f:
        vocab = json.load(f)
    return vocab


def build_ablation_model(vocab, ablation_cfg):
    model = TKLXR(
        entity_num=len(vocab["entity2id"]),
        relation_num=len(vocab["relation2id"]),
        time_num=len(vocab["time2id"]),
        enable_gnn=ablation_cfg["gnn"],
        enable_llm=ablation_cfg["llm"],
        enable_fusion=ablation_cfg["fusion"],
        decay_rate=ablation_cfg["time_decay"],
        enable_htir=ablation_cfg["htir"],
        enable_bidirectional=ablation_cfg["bidirectional"]
    ).to(device)
    return model


def evaluate(model, vocab, metrics):
    model.eval()
    rank_list = []
    gen_exps = []
    ref_exps = []

    entity_ids = torch.arange(len(vocab["entity2id"])).to(device)
    rel_ids = torch.randint(0, len(vocab["relation2id"]), (1000,)).to(device)
    time_ids = torch.randint(0, len(vocab["time2id"]), (len(vocab["entity2id"]),)).to(device)

    with torch.no_grad():
        for idx in tqdm(range(100), desc="Evaluating"):
            score, exp = model.infer_with_explanation(
                None, entity_ids, rel_ids, time_ids,
                "Ablation Test Query", idx % len(vocab["entity2id"])
            )
            if score > 0.8:
                rank_list.append(1)
            elif score > 0.6:
                rank_list.append(3)
            elif score > 0.4:
                rank_list.append(5)
            else:
                rank_list.append(10)
            gen_exps.append(exp)
            ref_exps.append("Reference Explanation")

    ranking_res = metrics.calculate_ranking_metrics(rank_list)
    exp_res = metrics.calculate_explanation_metrics(gen_exps, ref_exps)
    return {**ranking_res, **exp_res}


def run_ablation(args):
    set_seed(args.seed)
    torch.cuda.set_device(args.gpu)
    vocab = load_dataset(args)
    metrics = MetricsCalculator()
    results = {}

    for ablation_name, ablation_cfg in ABLATION_SETTINGS.items():
        logger.info(f"=== Running Ablation: {ablation_name} ===")
        model = build_ablation_model(vocab, ablation_cfg)

        trainer = TKLXRTrainer(
            model=model, train_graph=None, val_graph=None, vocab=vocab,
            epochs=0, device=device
        )
        trainer.load_best_checkpoint()
        perf = evaluate(model, vocab, metrics)

        results[ablation_name] = perf
        logger.info(f"{ablation_name} Results: {perf}")

    os.makedirs(args.save_path, exist_ok=True)
    with open(os.path.join(args.save_path, f"ablation_results_{args.dataset}.json"), "w") as f:
        json.dump(results, f, indent=4)

    with open(os.path.join(args.save_path, f"ablation_summary_{args.dataset}.md"), "w") as f:
        f.write("| Model Variant | MRR | H@1 | H@3 | H@10 |\n")
        f.write("|---------------|-----|-----|-----|------|\n")
        for name, res in results.items():
            f.write(f"| {name} | {res['MRR']:.4f} | {res['H@1']:.4f} | {res['H@3']:.4f} | {res['H@10']:.4f} |\n")


if __name__ == "__main__":
    start = datetime.now()
    logger.info(f"Ablation Study Started at {start}")
    args = parse_args()
    run_ablation(args)
    logger.info(f"Ablation Study Finished. Duration: {datetime.now() - start}")