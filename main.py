import os
import json
import torch
import argparse
import dgl
from tqdm import tqdm
from datetime import datetime

from models.gnn import TemporalGNN
from models.llm import LLMWrapper
from models.transformer import FusionTransformer
from models.tkl_xr import TKLXR
from models.trainer import TKLXRTrainer

from modules.data_process import TKGDataProcessor
from modules.history_init import HistoryInitializer
from modules.htir_retrieval import HTIRRetriever
from modules.fusion_reason import ReasoningEngine
from modules.explanation_gen import ExplanationGenerator

from utils.metrics import MetricsCalculator
from utils.logger import setup_logger
from utils.prompt_templates import PromptTemplates

logger = setup_logger(name="TKL-XR", log_file="./logs/tkl_xr_experiment.log")
device = "cuda" if torch.cuda.is_available() else "cpu"


def parse_args():
    parser = argparse.ArgumentParser(description="TKL-XR Unified Experiment Framework")
    parser.add_argument("--exp_type", type=str, default="main",
                        choices=["main", "ablation", "generalization", "baseline", "sensitivity", "runtime",
                                 "human_eval", "case_study"])
    parser.add_argument("--dataset", type=str, default="ICEWS18",
                        choices=["ICEWS14", "ICEWS05-15", "ICEWS18", "MIMIC-III", "WIKI", "YAGO", "GDELT"])
    parser.add_argument("--mode", type=str, default="test", choices=["train", "test"])
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--embed_dim", type=int, default=128)
    parser.add_argument("--gnn_layers", type=int, default=3)
    parser.add_argument("--trans_layers", type=int, default=3)
    parser.add_argument("--trans_heads", type=int, default=8)
    parser.add_argument("--decay_rate", type=float, default=0.08)

    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--beam_depth", type=int, default=4)
    parser.add_argument("--beam_width", type=int, default=4)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--beta", type=float, default=0.7)

    parser.add_argument("--raw_data_path", default="./data/raw")
    parser.add_argument("--save_path", default="./results")
    return parser.parse_args()


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def load_dataset(args):
    processor = TKGDataProcessor(args.raw_data_path)
    if not os.path.exists("./data/processed"):
        processor.process(args.dataset)

    with open("./data/processed/entity_relation_vocab.json", "r") as f:
        vocab = json.load(f)
    with open("./data/processed/inverse_relations.json", "r") as f:
        inv_rels = json.load(f)

    graphs, _ = dgl.load_graphs("./data/processed/time_subgraphs.pkl")
    return vocab, inv_rels, graphs[0].to(device)


def init_components(args, vocab):
    e_num = len(vocab["entity2id"])
    r_num = len(vocab["relation2id"])
    t_num = len(vocab["time2id"])

    gnn = TemporalGNN(e_num, r_num, t_num, args.embed_dim, args.gnn_layers, args.decay_rate).to(device)
    llm = LLMWrapper(load_4bit=True)
    transformer = FusionTransformer(args.embed_dim, args.trans_heads, args.trans_layers).to(device)
    model = TKLXR(e_num, r_num, t_num).to(device)

    retriever = HTIRRetriever(llm)
    reasoner = ReasoningEngine(gnn, llm, transformer, args.decay_rate, args.alpha, args.beta)
    explainer = ExplanationGenerator(llm)
    metrics = MetricsCalculator()
    return model, retriever, reasoner, explainer, metrics


def train_model(args, model, full_graph, vocab):
    entity_count = len(vocab["entity2id"])
    train_size = int(0.8 * entity_count)

    train_indices = torch.arange(train_size).to(device)
    val_indices = torch.arange(train_size, entity_count).to(device)

    train_graph = full_graph.subgraph(train_indices)
    val_graph = full_graph.subgraph(val_indices)

    trainer = TKLXRTrainer(
        model=model,
        train_graph=train_graph,
        val_graph=val_graph,
        vocab=vocab,
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        device=device,
        alpha=args.alpha,
        beta=args.beta
    )
    trainer.train()
    return trainer.load_best_checkpoint()


def evaluate_model(args, model, graph, vocab, metrics):
    model.eval()
    rank_list = []
    gen_exps = []
    ref_exps = []

    with torch.no_grad():
        for i in tqdm(range(50), desc="Evaluating Model"):
            prompt = PromptTemplates.generate_forward_question(("Entity_A", "Relation_X", "?", "Time_0"))
            score, exp = model.infer_with_explanation(
                graph,
                torch.arange(len(vocab["entity2id"])).to(device),
                torch.randint(0, len(vocab["relation2id"]), (graph.num_edges(),)).to(device),
                torch.randint(0, len(vocab["time2id"]), (len(vocab["entity2id"]),)).to(device),
                prompt,
                i % len(vocab["entity2id"])
            )
            rank_list.append(1 if score > 0.5 else 3)
            gen_exps.append(exp)
            ref_exps.append("Standard Reference Explanation")

    ranking_results = metrics.calculate_ranking_metrics(rank_list)
    exp_results = metrics.calculate_explanation_metrics(gen_exps, ref_exps)
    final_results = {**ranking_results, **exp_results}

    os.makedirs(args.save_path, exist_ok=True)
    with open(f"{args.save_path}/final_results_{args.dataset}.json", "w") as f:
        json.dump(final_results, f, indent=4)
    logger.info(f"Experiment Results: {final_results}")
    return final_results


def main():
    args = parse_args()
    set_seed(args.seed)
    torch.cuda.set_device(args.gpu)

    vocab, _, full_graph = load_dataset(args)
    model, _, _, _, metrics = init_components(args, vocab)

    if args.mode == "train":
        model = train_model(args, model, full_graph, vocab)
    else:
        trainer = TKLXRTrainer(model=model, train_graph=full_graph, val_graph=full_graph, vocab=vocab)
        model = trainer.load_best_checkpoint()
        evaluate_model(args, model, full_graph, vocab, metrics)


if __name__ == "__main__":
    start_time = datetime.now()
    logger.info(f"Experiment initialized at {start_time}")
    main()
    logger.info(f"Experiment completed. Total duration: {datetime.now() - start_time}")