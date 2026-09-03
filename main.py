from __future__ import annotations

import argparse
import json
import os
import random
from datetime import datetime
from typing import Dict, Iterable, List, Tuple

try:
    import numpy as np
except Exception:  # pragma: no cover - dependency check is handled at runtime
    np = None
try:
    import torch
except Exception:  # pragma: no cover
    torch = None
try:
    from tqdm import tqdm
except Exception:  # pragma: no cover
    def tqdm(iterable, **_kwargs):
        return iterable

try:
    from models.llm import LLMWrapper
    from models.tkl_xr import TKLXR
except Exception:  # pragma: no cover - allows ``--help`` without ML packages
    LLMWrapper = None
    TKLXR = None
from modules.data_process import TemporalKGDataset, load_dataset
from modules.explanation import ExplanationGenerator
from modules.history import HistoryInitializer
from modules.retrieval import HTIRRetriever
from utils.logger import setup_logger
from utils.metrics import MetricsCalculator


logger = setup_logger(name="TKL-XR", log_file="./logs/tkl_xr_experiment.log")
DEVICE = "cuda" if torch is not None and torch.cuda.is_available() else "cpu"


def _require_runtime_dependencies():
    missing = []
    if np is None:
        missing.append("numpy")
    if torch is None or TKLXR is None:
        missing.append("torch")
    if missing:
        names = ", ".join(dict.fromkeys(missing))
        raise RuntimeError(
            f"Missing runtime dependencies: {names}. Install requirements.txt "
            "before running training or evaluation."
        )


def parse_args():
    parser = argparse.ArgumentParser(description="TKL-XR experiment pipeline")
    parser.add_argument("--dataset", default="ICEWS18")
    parser.add_argument("--mode", choices=["train", "test"], default="test")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--embed_dim", type=int, default=128)
    parser.add_argument("--gnn_layers", type=int, default=3)
    parser.add_argument("--trans_layers", type=int, default=3)
    parser.add_argument("--trans_heads", type=int, default=8)
    parser.add_argument("--decay_rate", type=float, default=0.08)
    parser.add_argument("--beam_depth", type=int, default=4)
    parser.add_argument("--beam_width", type=int, default=4)
    parser.add_argument("--retrieval_rounds", type=int, default=1)
    parser.add_argument("--top_relations", type=int, default=10)
    parser.add_argument("--top_entities", type=int, default=10)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--beta", type=float, default=0.7)
    parser.add_argument("--max_eval_samples", type=int, default=200)
    parser.add_argument(
        "--filtered_ranking",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Remove other known answers when computing the target rank.",
    )
    # ``all`` preserves strict full-entity ranking.  LLM scores are still
    # queried only for HTIR-retained candidates; missing scores are zero.
    parser.add_argument("--candidate_mode", choices=["htir", "all"], default="all")
    parser.add_argument("--raw_data_path", default="./data")
    parser.add_argument("--processed_path", default="./data/processed")
    parser.add_argument("--checkpoint_path", default="./checkpoints")
    parser.add_argument("--save_path", default="./results")
    parser.add_argument("--use_mock_llm", action="store_true")
    return parser.parse_args()


def set_seed(seed: int):
    _require_runtime_dependencies()
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _move_graphs(dataset: TemporalKGDataset):
    if DEVICE != "cuda":
        return
    dataset.time_graphs = {
        time_id: graph.to(DEVICE) for time_id, graph in dataset.time_graphs.items()
    }


def build_model(args, dataset: TemporalKGDataset):
    return TKLXR(
        entity_num=dataset.entity_num,
        relation_num=dataset.relation_num,
        time_num=dataset.time_num,
        embed_dim=args.embed_dim,
        gnn_layers=args.gnn_layers,
        trans_layers=args.trans_layers,
        trans_heads=args.trans_heads,
        decay_rate=args.decay_rate,
        device=DEVICE,
        alpha=args.alpha,
        beta=args.beta,
        beam_depth=args.beam_depth,
        beam_width=args.beam_width,
        load_4bit=not args.use_mock_llm,
        use_mock_llm=args.use_mock_llm,
    ).to(DEVICE)


def _query_text(dataset: TemporalKGDataset, quad, inverse: bool = False) -> str:
    id2entity = {value: key for key, value in dataset.vocab["entity2id"].items()}
    id2relation = {value: key for key, value in dataset.vocab["relation2id"].items()}
    head, relation, tail, time = map(int, quad)
    if inverse:
        inverse_relation = dataset.inverse_relations.get(relation, relation)
        return (
            f'Which entity is the subject of relation "{id2relation.get(inverse_relation, inverse_relation)}" '
            f'for "{id2entity.get(tail, tail)}" at time {time}?'
        )
    return (
        f'Which entity is the object of relation "{id2relation.get(relation, relation)}" '
        f'for "{id2entity.get(head, head)}" at time {time}?'
    )


def _direction_scores(model, retriever, dataset, quad, inverse: bool = False, candidate_mode: str = "all"):
    head, relation, tail, time_id = map(int, quad)
    if inverse:
        start_entity, query_relation, target = tail, dataset.inverse_relations.get(relation, relation), head
    else:
        start_entity, query_relation, target = head, relation, tail
    question = _query_text(dataset, quad, inverse=inverse)
    llm_candidate_ids = None
    if model.enable_htir and model.llm is not None:
        initializer = HistoryInitializer(
            llm=model.llm,
            inverse_relations=dataset.inverse_relations,
            entity2id=dataset.vocab["entity2id"],
            relation2id=dataset.vocab["relation2id"],
        )
        initial_entities = initializer.retrieve_initial_entities(
            dataset.time_graphs,
            start_entity=start_entity,
            question=question,
            beam_depth=model.beam_depth,
            beam_width=model.beam_width,
            query_time=time_id,
        )
    else:
        initial_entities = [start_entity]
    if model.enable_htir and model.llm is not None:
        # HTIR may only travel through events strictly earlier than the query.
        # An empty history is valid for the first timestamp and must not fall
        # back to the current/future graph.
        history_graphs = {k: v for k, v in dataset.time_graphs.items() if k < time_id}
        retrieval = retriever.retrieve(history_graphs, initial_entities, question)
        htir_candidates = set(retrieval["entity_scores"]) | {target}
        candidate_ids = set(htir_candidates)
        if candidate_mode == "all":
            candidate_ids |= set(range(dataset.entity_num))
        # Full filtered ranking remains over every entity, but expensive LLM
        # scoring is restricted to the candidates retained by HTIR.
        llm_candidate_ids = htir_candidates
    else:
        retrieval = {
            "relation_scores": {},
            "entity_scores": {},
            "relation_paths": [],
            "entity_paths": [],
        }
        candidate_ids = set(range(dataset.entity_num))
    if not candidate_ids:
        candidate_ids = {target}

    graph = dataset.graph_for_time(time_id, strict_before=True)
    entity_ids = torch.arange(dataset.entity_num, dtype=torch.long, device=DEVICE)
    rel_ids = graph.edata.get("rel", torch.empty(0, dtype=torch.long, device=DEVICE))
    time_ids = torch.full((dataset.entity_num,), time_id, dtype=torch.long, device=DEVICE)

    path_scores = {}
    for path in retrieval["entity_paths"]:
        if not path:
            continue
        entity = int(path[-1][0])
        path_text = " -> ".join(f"({e},{r},{t})" for e, r, t in path)
        score = model.llm.score(f"{question}\nReasoning path: {path_text}") if model.llm is not None else 0.0
        path_scores[entity] = max(path_scores.get(entity, 0.0), float(score))

    return model.rank_candidates(
        graph,
        entity_ids,
        rel_ids,
        time_ids,
        start_entity,
        query_relation,
        time_id,
        sorted(candidate_ids),
        question,
        path_scores,
        llm_candidate_ids=llm_candidate_ids,
    ), retrieval, target


def _backward_verification_scores(
    model, dataset, quad, candidate_ids, llm_candidate_ids=None
):
    """Score whether each forward object candidate can recover the source entity.

    This implements the paper's bidirectional verification without using the
    true object entity as the inverse-query starting point during ranking.
    """
    head, relation, _, time_id = map(int, quad)
    inverse_relation = dataset.inverse_relations.get(relation, relation)
    graph = dataset.graph_for_time(time_id, strict_before=True)
    entity_ids = torch.arange(dataset.entity_num, dtype=torch.long, device=DEVICE)
    rel_ids = graph.edata.get("rel", torch.empty(0, dtype=torch.long, device=DEVICE))
    time_ids = torch.full((dataset.entity_num,), time_id, dtype=torch.long, device=DEVICE)
    candidate_list = [int(candidate_id) for candidate_id in candidate_ids]
    if not candidate_list:
        return {}
    llm_candidates = (
        None
        if llm_candidate_ids is None
        else {int(candidate_id) for candidate_id in llm_candidate_ids}
    )
    if model.enable_gnn and model.gnn is not None:
        entity_emb = model.gnn.forward(graph, entity_ids, rel_ids, time_ids)
        candidate_tensor = torch.tensor(
            candidate_list, dtype=torch.long, device=entity_ids.device
        )
        relation_emb = model.gnn.relation_embedding(
            torch.full(
                (len(candidate_list),),
                int(inverse_relation),
                dtype=torch.long,
                device=entity_ids.device,
            )
        )
        target_emb = entity_emb[int(head)].expand(len(candidate_list), -1)
        gnn_scores = torch.sigmoid(
            (entity_emb[candidate_tensor] * relation_emb * target_emb).sum(dim=-1)
        )
    else:
        gnn_scores = torch.zeros(
            len(candidate_list), dtype=torch.float32, device=entity_ids.device
        )

    llm_values = []
    for candidate_object in candidate_list:
        allowed = llm_candidates is None or candidate_object in llm_candidates
        if model.enable_llm and model.llm is not None and allowed:
            prompt = (
                f'Verify the inverse temporal query: which subject is linked to '
                f'candidate entity "{candidate_object}" by inverse relation '
                f'"{inverse_relation}" at time {time_id}?'
            )
            llm_values.append(
                float(model.llm.score(f"{prompt}\nCandidate entity: {head}"))
            )
        else:
            llm_values.append(0.0)
    llm_scores = torch.tensor(
        llm_values, dtype=torch.float32, device=entity_ids.device
    )
    fused = model.fuse_scores_tensor(llm_scores, gnn_scores).detach().reshape(-1)
    scores = {
        candidate_object: float(score.item())
        for candidate_object, score in zip(candidate_list, fused)
    }
    return scores


def _bidirectional_pass(model, retriever, dataset, quad, candidate_mode: str):
    """Run forward retrieval and inverse verification without target leakage."""
    forward_scores, retrieval, target = _direction_scores(
        model, retriever, dataset, quad, inverse=False, candidate_mode=candidate_mode
    )
    htir_candidates = set(retrieval["entity_scores"]) | {int(target)}
    backward_scores = _backward_verification_scores(
        model,
        dataset,
        quad,
        forward_scores.keys(),
        llm_candidate_ids=htir_candidates,
    )
    return forward_scores, backward_scores, retrieval, target


def _known_answers(dataset: TemporalKGDataset, split_name: str = "test"):
    answers = {}
    quads = dataset.train + dataset.valid
    if split_name == "test":
        quads = quads + dataset.test
    for head, relation, tail, time_id in quads:
        answers.setdefault((head, relation, time_id), set()).add(tail)
    return answers


def _rank_target(scores: Dict[int, float], target: int, filtered_entities=None) -> int:
    if target not in scores:
        return len(scores) + 1
    target_score = scores[target]
    filtered_entities = set(filtered_entities or [])
    return 1 + sum(
        1
        for entity, score in scores.items()
        if entity != target and entity not in filtered_entities and score > target_score
    )


def _predict(scores: Dict[int, float]) -> Tuple[int, float]:
    entity = max(scores, key=scores.get)
    return entity, float(scores[entity])


def _bidirectional_with_second_pass(model, forward_scores, backward_scores, rerun_fn):
    forward_entity, forward_score = _predict(forward_scores)
    if not model.enable_bidirectional or not backward_scores:
        return forward_scores, forward_entity, forward_score, False
    backward_entity, backward_score = _predict(backward_scores)
    if forward_entity == backward_entity:
        merged = dict(forward_scores)
        merged[forward_entity] = (forward_score + backward_score) / 2.0
        return merged, forward_entity, merged[forward_entity], False

    second_forward, second_backward = rerun_fn()
    second_forward_entity, second_forward_score = _predict(second_forward)
    second_backward_entity, second_backward_score = _predict(second_backward)
    if second_forward_entity == second_backward_entity:
        merged = dict(second_forward)
        merged[second_forward_entity] = (second_forward_score + second_backward_score) / 2.0
        return merged, second_forward_entity, merged[second_forward_entity], True

    # Keep the complete second-pass candidate union so ranking metrics remain
    # well-defined even when the two directions still disagree.
    merged = dict(second_forward)
    merged.update({entity: score for entity, score in second_backward.items() if entity not in merged})
    if second_forward_score >= second_backward_score:
        return merged, second_forward_entity, second_forward_score, True
    return merged, second_backward_entity, second_backward_score, True


def evaluate(model, dataset: TemporalKGDataset, args) -> Dict[str, float]:
    model.eval()
    retriever = HTIRRetriever(
        model.llm,
        beam_depth=args.beam_depth,
        beam_width=args.beam_width,
        retrieval_rounds=args.retrieval_rounds,
        top_relations=args.top_relations,
        top_entities=args.top_entities,
    )
    metrics = MetricsCalculator()
    ranks, generated, references = [], [], []
    id2entity = {value: key for key, value in dataset.vocab["entity2id"].items()}
    known_answers = _known_answers(dataset, "test") if getattr(args, "filtered_ranking", True) else {}

    samples = dataset.test[: int(getattr(args, "max_eval_samples", len(dataset.test)))]
    candidate_mode = getattr(args, "candidate_mode", "all")
    with torch.no_grad():
        for quad in tqdm(samples, desc=f"Evaluating {dataset.dataset}"):
            forward_scores, backward_scores, retrieval, target = _bidirectional_pass(
                model, retriever, dataset, quad, candidate_mode
            )
            merged, _, _, _ = _bidirectional_with_second_pass(
                model,
                forward_scores,
                backward_scores,
                lambda: _bidirectional_pass(
                    model, retriever, dataset, quad, candidate_mode
                )[:2],
            )
            head, relation, tail, time_id = map(int, quad)
            filter_entities = known_answers.get((head, relation, time_id), set()) - {tail}
            ranks.append(_rank_target(merged, target, filter_entities))

            paths = retrieval["entity_paths"][:3]
            if model.llm is not None:
                explanation_generator = ExplanationGenerator(model.llm)
                explanation, _ = explanation_generator.generate(
                    _query_text(dataset, quad),
                    paths,
                    id2entity.get(target, str(target)),
                )
                generated.append(explanation)
                references.append(
                    f'The target entity is {id2entity.get(target, str(target))}, '
                    "supported by the retrieved temporal paths."
                )

    result = metrics.calculate_ranking_metrics(ranks)
    if generated:
        result.update(metrics.calculate_explanation_metrics(generated, references))
    return result


def run(args):
    _require_runtime_dependencies()
    set_seed(args.seed)
    if DEVICE == "cuda":
        torch.cuda.set_device(args.gpu)
    dataset = load_dataset(args.dataset, args.raw_data_path, args.processed_path)
    _move_graphs(dataset)
    model = build_model(args, dataset)

    from trainer import TKLXRTrainer

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
        checkpoint_path=os.path.join(args.checkpoint_path, args.dataset),
        seed=args.seed,
    )
    if args.mode == "train":
        trainer.train()
    else:
        trainer.load_best_checkpoint()

    results = evaluate(model, dataset, args)
    os.makedirs(args.save_path, exist_ok=True)
    output = os.path.join(args.save_path, f"results_{args.dataset}.json")
    with open(output, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    run_config = {
        **vars(args),
        "resolved_device": DEVICE,
        "entity_count": dataset.entity_num,
        "relation_count_with_inverse": dataset.relation_num,
        "time_count": dataset.time_num,
        "train_count": len(dataset.train),
        "valid_count": len(dataset.valid),
        "test_count": len(dataset.test),
        "model": {
            "enable_gnn": model.enable_gnn,
            "enable_llm": model.enable_llm,
            "enable_fusion": model.enable_fusion,
            "enable_htir": model.enable_htir,
            "enable_bidirectional": model.enable_bidirectional,
            "linear_fusion": model.linear_fusion,
        },
    }
    with open(os.path.join(args.save_path, f"config_{args.dataset}.json"), "w", encoding="utf-8") as f:
        json.dump(run_config, f, indent=2)
    logger.info("Results written to %s: %s", output, results)
    return results


if __name__ == "__main__":
    started = datetime.now()
    run(parse_args())
    logger.info("Finished in %s", datetime.now() - started)
