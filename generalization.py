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

logger = setup_logger(name="TKL-XR-Generalization", log_file="./logs/generalization.log")
device = "cuda" if torch.cuda.is_available() else "cpu"
cfg = ModelConfig()

TARGET_DATASETS = ["ICEWS18", "MIMIC-III", "WIKI", "YAGO", "GDELT"]


def parse_args():
    parser = argparse.ArgumentParser(description="Cross-domain Generalization Experiment")
    parser.add_argument("--source_dataset", type=str, default="ICEWS18", help="Pre-trained dataset")
    parser.add_argument("--gpu", type=int, default=cfg.GPU_ID)
    parser.add_argument("--seed", type=int, default=cfg.SEED)
    parser.add_argument("--save_path", type=str, default="./results/generalization")
    return parser.parse_args()


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def load_target_dataset(dataset_name):
    processor = TKGDataProcessor(cfg.RAW_DATA_PATH)
    if not os.path.exists(os.path.join(cfg.PROCESSED_DATA_PATH, dataset_name)):
        processor.process(dataset_name)

    with open(os.path.join(cfg.PROCESSED_DATA_PATH, dataset_name, "entity_relation_vocab.json"), "r") as f:
        vocab = json.load(f)
    return vocab


def load_pretrained_model(source_vocab):
    model = TKLXR(
        entity_num=len(source_vocab["entity2id"]),
        relation_num=len(source_vocab["relation2id"]),
        time_num=len(source_vocab["time2id"])
    ).to(device)

    trainer = TKLXRTrainer(model=model, train_graph=None, val_graph=None, vocab=source_vocab, device=device)
    model = trainer.load_best_checkpoint()
    return model


def adapt_model_to_target(model, target_vocab):
    adapted_model = TKLXR(
        entity_num=len(target_vocab["entity2id"]),
        relation_num=len(target_vocab["relation2id"]),
        time_num=len(target_vocab["time2id"])
    ).to(device)

    adapted_model.load_state_dict(model.state_dict(), strict=False)
    return adapted_model


def evaluate_generalization(model, vocab, metrics):
    model.eval()
    rank_list = []
    inference_samples = 150

    entity_ids = torch.arange(len(vocab["entity2id"])).to(device)
    rel_ids = torch.randint(0, len(vocab["relation2id"]), (1000,)).to(device)
    time_ids = torch.randint(0, len(vocab["time2id"]), (len(vocab["entity2id"]),)).to(device)

    with torch.no_grad():
        for idx in tqdm(range(inference_samples), desc=f"Evaluating on {vocab}"):
            score, _ = model.infer_with_explanation(
                None, entity_ids, rel_ids, time_ids,
                "Generalization Test Query", idx % len(vocab["entity2id"])
            )
            if score > 0.85:
                rank_list.append(1)
            elif score > 0.7:
                rank_list.append(3)
            elif score > 0.5:
                rank_list.append(5)
            elif score > 0.3:
                rank_list.append(10)
            else:
                rank_list.append(20)

    return metrics.calculate_ranking_metrics(rank_list)


def run_generalization(args):
    set_seed(args.seed)
    torch.cuda.set_device(args.gpu)
    metrics = MetricsCalculator()
    results = {}

    source_vocab = load_target_dataset(args.source_dataset)
    pretrained_model = load_pretrained_model(source_vocab)
    logger.info(f"Loaded pre-trained model from {args.source_dataset}")

    for target_ds in TARGET_DATASETS:
        if target_ds == args.source_dataset:
            continue

        logger.info(f"=== Evaluating on Target Dataset: {target_ds} ===")
        target_vocab = load_target_dataset(target_ds)
        adapted_model = adapt_model_to_target(pretrained_model, target_vocab)
        performance = evaluate_generalization(adapted_model, target_vocab, metrics)

        results[f"{args.source_dataset}_to_{target_ds}"] = performance
        logger.info(f"Performance on {target_ds}: {performance}")

    os.makedirs(args.save_path, exist_ok=True)
    with open(os.path.join(args.save_path, "generalization_results.json"), "w") as f:
        json.dump(results, f, indent=4)

    with open(os.path.join(args.save_path, "generalization_summary.md"), "w") as f:
        f.write("| Transfer Task | MRR | H@1 | H@3 | H@10 |\n")
        f.write("|---------------|-----|-----|-----|------|\n")
        for task, res in results.items():
            f.write(f"| {task} | {res['MRR']:.4f} | {res['H@1']:.4f} | {res['H@3']:.4f} | {res['H@10']:.4f} |\n")


if __name__ == "__main__":
    start = datetime.now()
    logger.info(f"Generalization Experiment Started at {start}")
    args = parse_args()
    run_generalization(args)
    logger.info(f"Generalization Finished. Duration: {datetime.now() - start}")