from __future__ import annotations

import json
import os
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

try:  # pragma: no cover - optional runtime dependency
    import torch
except Exception:  # pragma: no cover
    torch = None

try:  # pragma: no cover - optional runtime dependency
    import dgl
except Exception:  # pragma: no cover
    dgl = None


Quadruple = Tuple[int, int, int, int]


class SimpleTemporalGraph:
    """Small DGL-compatible graph fallback for CPU environments.

    The experiment pipeline only needs edge iteration, edge attributes, node
    count, and device transfer.  Keeping this fallback here makes preprocessing
    and mock evaluation runnable even when the optional DGL binary is absent or
    incompatible with the installed PyTorch build.
    """

    def __init__(self, src, dst, num_nodes: int, rel=None, time=None):
        self._src = torch.tensor(list(src), dtype=torch.long) if torch is not None else list(src)
        self._dst = torch.tensor(list(dst), dtype=torch.long) if torch is not None else list(dst)
        self._num_nodes = int(num_nodes)
        self.edata = {
            "rel": torch.tensor(list(rel or [0] * len(src)), dtype=torch.long)
            if torch is not None else list(rel or [0] * len(src)),
            "time": torch.tensor(list(time or [0] * len(src)), dtype=torch.long)
            if torch is not None else list(time or [0] * len(src)),
        }

    @property
    def device(self):
        return self._src.device if torch is not None and torch.is_tensor(self._src) else "cpu"

    def edges(self):
        return self._src, self._dst

    def num_nodes(self):
        return self._num_nodes

    def num_edges(self):
        return int(len(self._src))

    def to(self, device):
        if torch is not None:
            self._src = self._src.to(device)
            self._dst = self._dst.to(device)
            self.edata = {key: value.to(device) for key, value in self.edata.items()}
        return self


def _natural_key(value: str):
    try:
        return int(value)
    except Exception:
        return value


def _split_columns(line: str) -> List[str]:
    return [part for part in line.replace("\t", " ").split(" ") if part]


def _parse_mapping_file(path: Path) -> Dict[str, int]:
    mapping: Dict[str, int] = {}
    if not path.exists():
        return mapping
    with path.open("r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line:
                continue
            parts = _split_columns(line)
            if len(parts) < 2:
                continue
            if len(parts) >= 2 and parts[1].lstrip("-").isdigit():
                key, idx = parts[0], int(parts[1])
            elif parts[-1].lstrip("-").isdigit():
                key, idx = " ".join(parts[:-1]), int(parts[-1])
            elif parts[0].lstrip("-").isdigit():
                key, idx = " ".join(parts[1:]), int(parts[0])
            else:
                continue
            mapping[key] = idx
    return mapping


def _read_split_file(path: Path) -> List[Tuple[str, str, str, str]]:
    quads: List[Tuple[str, str, str, str]] = []
    if not path.exists():
        return quads
    with path.open("r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line:
                continue
            parts = _split_columns(line)
            if len(parts) < 4:
                continue
            head, relation, tail, time = parts[:4]
            quads.append((head, relation, tail, time))
    return quads


def _read_mimic_split_file(path: Path, split_name: str) -> List[Tuple[str, str, str, str]]:
    """Convert the supplied MIMIC hyperedge JSONL examples to temporal KG quads.

    Each JSON record is treated as one patient/encounter at a relative time.
    Disease, prescription, and procedure codes become observed entities, while
    the provided binary labels are retained as outcome entities.  This adapter
    keeps the original values and makes the supplied MIMIC files consumable by
    the common TKG pipeline.
    """
    quads: List[Tuple[str, str, str, str]] = []
    if not path.exists():
        return quads
    with path.open("r", encoding="utf-8") as f:
        for record_id, raw_line in enumerate(f):
            line = raw_line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            case = f"{split_name}:case:{record_id}"
            timestamp = str(record_id)
            for category, relation in (
                ("disease", "has_disease"),
                ("prescription", "has_prescription"),
                ("procedure", "has_procedure"),
            ):
                values = record.get(category, {})
                for code in values.get("idxs", []) if isinstance(values, dict) else []:
                    quads.append((case, relation, f"{category}:{code}", timestamp))
            labels = record.get("label", [])
            for label_index, label in enumerate(labels):
                quads.append(
                    (case, "mortality_label", f"outcome:{str(label)}", str(label_index))
                )
    return quads


@dataclass
class TemporalKGDataset:
    dataset: str
    vocab: Dict[str, Dict[str, int]]
    inverse_relations: Dict[int, int]
    train: List[Quadruple]
    valid: List[Quadruple]
    test: List[Quadruple]
    time_graphs: Dict[int, object]

    def graph_for_time(self, time_id: int, strict_before: bool = False):
        if not self.time_graphs:
            raise ValueError("No temporal graphs available.")
        if not strict_before and time_id in self.time_graphs:
            return self.time_graphs[time_id]
        available = sorted(self.time_graphs)
        threshold = time_id if not strict_before else time_id - 1
        previous = [t for t in available if t <= threshold]
        if strict_before and not previous:
            # No event before the first timestamp is available.  Construct an
            # empty graph with the same node count instead of falling back to
            # the first (current-time) graph, which would leak query-time
            # edges into history initialization/evaluation.
            first_graph = self.time_graphs[available[0]]
            if dgl is not None and torch is not None and hasattr(first_graph, "num_nodes"):
                empty = dgl.graph(
                    ([], []),
                    num_nodes=int(first_graph.num_nodes()),
                    device=first_graph.device if hasattr(first_graph, "device") else None,
                )
                empty.edata["rel"] = torch.empty(0, dtype=torch.long)
                empty.edata["time"] = torch.empty(0, dtype=torch.long)
                return empty
            if hasattr(first_graph, "num_nodes"):
                return SimpleTemporalGraph(
                    [], [], int(first_graph.num_nodes()), rel=[], time=[]
                )
            return first_graph
        fallback = max(previous, default=available[0])
        return self.time_graphs[fallback]

    @property
    def entity_num(self) -> int:
        return len(self.vocab["entity2id"])

    @property
    def relation_num(self) -> int:
        return len(self.vocab["relation2id"])

    @property
    def time_num(self) -> int:
        return len(self.vocab["time2id"])

    def all_quads(self) -> List[Quadruple]:
        return list(self.train) + list(self.valid) + list(self.test)


class TKGDataProcessor:
    def __init__(self, raw_data_path: str, save_path: str = "./data/processed"):
        self.raw_data_path = Path(raw_data_path)
        self.save_path = Path(save_path)
        self.save_path.mkdir(parents=True, exist_ok=True)

    def _dataset_dir(self, dataset: str) -> Path:
        direct = self.raw_data_path / dataset
        if direct.exists():
            return direct
        # The supplied MIMIC package is stored in one shared ``mimic`` folder.
        # Both paper aliases therefore resolve to that folder while retaining
        # the requested dataset name in processed artifacts.
        if dataset.lower() in {"mimic", "mimic-iii", "mimic-iv"}:
            shared = self.raw_data_path / "mimic"
            if shared.exists():
                return shared
        return direct

    def _mapping_paths(self, dataset: str) -> Tuple[Path, Path]:
        base = self._dataset_dir(dataset)
        return base / "entity2id.txt", base / "relation2id.txt"

    def load_splits(self, dataset: str) -> Dict[str, List[Tuple[str, str, str, str]]]:
        base = self._dataset_dir(dataset)
        if not base.exists():
            raise FileNotFoundError(f"Dataset directory not found: {base}")
        if dataset.lower() in {"mimic", "mimic-iii", "mimic-iv"}:
            split_files = {
                "train": base / "hyperedges-mimic-text-train-example.jsonl",
                "valid": base / "hyperedges-mimic-text-valid-example.jsonl",
                "test": base / "hyperedges-mimic-text-test-example.jsonl",
            }
            splits = {
                name: _read_mimic_split_file(path, name)
                for name, path in split_files.items()
            }
            if any(splits.values()):
                return splits
        splits = {
            "train": _read_split_file(base / "train.txt"),
            "valid": _read_split_file(base / "valid.txt"),
            "test": _read_split_file(base / "test.txt"),
        }
        if not any(splits.values()):
            raise FileNotFoundError(f"No split files found under {base}")
        return splits

    def _build_vocab(self, splits: Dict[str, List[Tuple[str, str, str, str]]], dataset: str) -> Dict[str, Dict[str, int]]:
        entity_path, relation_path = self._mapping_paths(dataset)
        entity_mapping = _parse_mapping_file(entity_path)
        relation_mapping = _parse_mapping_file(relation_path)

        observed = splits["train"] + splits["valid"] + splits["test"]
        observed_entities = {value for quad in observed for value in (quad[0], quad[2])}
        observed_relations = {quad[1] for quad in observed}
        times = sorted({quad[3] for quad in observed}, key=_natural_key)

        # The supplied benchmark files use integer IDs in the split files,
        # while some mapping files use human-readable names.  Prefer the
        # representation actually present in the split files and remap it to
        # contiguous IDs for embeddings/DGL.
        mapping_entity_ids = {str(value) for value in entity_mapping.values()}
        mapping_relation_ids = {str(value) for value in relation_mapping.values()}
        if observed_entities and observed_entities.issubset(mapping_entity_ids):
            entity_keys = sorted(observed_entities, key=_natural_key)
        elif observed_entities and observed_entities.issubset(set(entity_mapping)):
            entity_keys = sorted(observed_entities, key=_natural_key)
        else:
            entity_keys = sorted(observed_entities, key=_natural_key)

        if observed_relations and observed_relations.issubset(mapping_relation_ids):
            relation_keys = sorted(observed_relations, key=_natural_key)
        elif observed_relations and observed_relations.issubset(set(relation_mapping)):
            relation_keys = sorted(observed_relations, key=_natural_key)
        else:
            relation_keys = sorted(observed_relations, key=_natural_key)

        entity2id = {entity: idx for idx, entity in enumerate(entity_keys)}
        relation2id = {relation: idx for idx, relation in enumerate(relation_keys)}
        time2id = {tm: idx for idx, tm in enumerate(times)}

        inverse_offset = len(relation2id)
        inverse_relations: Dict[int, int] = {}
        extended_relation2id = dict(relation2id)
        for relation, rid in sorted(relation2id.items(), key=lambda item: item[1]):
            inv_name = f"{relation}_inv"
            inv_id = inverse_offset + rid
            extended_relation2id[inv_name] = inv_id
            inverse_relations[rid] = inv_id
            inverse_relations[inv_id] = rid

        return {
            "entity2id": entity2id,
            "relation2id": extended_relation2id,
            "time2id": time2id,
            "inverse_relations": inverse_relations,
        }

    def _map_split(self, split: List[Tuple[str, str, str, str]], vocab: Dict[str, Dict[str, int]]) -> List[Quadruple]:
        entity2id = vocab["entity2id"]
        relation2id = vocab["relation2id"]
        time2id = vocab["time2id"]
        mapped: List[Quadruple] = []
        for h, r, t, tm in split:
            if h not in entity2id or t not in entity2id or r not in relation2id or tm not in time2id:
                continue
            mapped.append((entity2id[h], relation2id[r], entity2id[t], time2id[tm]))
        return mapped

    def _build_temporal_graphs(
        self,
        all_quads: Sequence[Quadruple],
        entity_num: int,
        base_relation_num: int,
        timeline: Optional[Sequence[int]] = None,
    ) -> Dict[int, object]:
        if torch is None:
            raise RuntimeError("torch is required to build temporal graphs.")

        by_time: Dict[int, List[Tuple[int, int, int]]] = defaultdict(list)
        for head_id, rel_id, tail_id, time_id in all_quads:
            by_time[time_id].append((head_id, rel_id, tail_id))

        running_src: List[int] = []
        running_dst: List[int] = []
        running_rel: List[int] = []
        running_time: List[int] = []
        graphs: Dict[int, object] = {}

        timeline_ids = sorted(set(timeline or by_time))
        for time_id in timeline_ids:
            for head_id, rel_id, tail_id in by_time[time_id]:
                inv_rel_id = rel_id + base_relation_num
                running_src.extend([head_id, tail_id])
                running_dst.extend([tail_id, head_id])
                running_rel.extend([rel_id, inv_rel_id])
                running_time.extend([time_id, time_id])

            if dgl is not None:
                graph = dgl.add_self_loop(
                    dgl.graph((running_src, running_dst), num_nodes=entity_num)
                )
                # ``add_self_loop`` appends one edge per node; assign explicit
                # relation/time features for those edges so feature lengths match.
                loop_count = graph.num_edges() - len(running_rel)
                graph.edata["rel"] = torch.tensor(
                    running_rel + [0] * loop_count, dtype=torch.long
                )
                graph.edata["time"] = torch.tensor(
                    running_time + [time_id] * loop_count, dtype=torch.long
                )
            else:
                graph = SimpleTemporalGraph(
                    running_src + list(range(entity_num)),
                    running_dst + list(range(entity_num)),
                    entity_num,
                    rel=running_rel + [0] * entity_num,
                    time=running_time + [time_id] * entity_num,
                )
            graphs[time_id] = graph

        return graphs

    def save_processed_data(self, dataset: str, vocab: Dict[str, Dict[str, int]], splits: Dict[str, List[Quadruple]], graphs: Dict[int, object]):
        processed_dir = self.save_path / dataset
        processed_dir.mkdir(parents=True, exist_ok=True)

        with (processed_dir / "entity_relation_vocab.json").open("w", encoding="utf-8") as f:
            json.dump(
                {
                    "entity2id": vocab["entity2id"],
                    "relation2id": vocab["relation2id"],
                    "time2id": vocab["time2id"],
                },
                f,
                indent=2,
                ensure_ascii=False,
            )

        with (processed_dir / "inverse_relations.json").open("w", encoding="utf-8") as f:
            json.dump({str(k): v for k, v in vocab["inverse_relations"].items()}, f, indent=2)

        with (processed_dir / "splits.json").open("w", encoding="utf-8") as f:
            json.dump({name: quads for name, quads in splits.items()}, f, indent=2)

        with (processed_dir / "metadata.json").open("w", encoding="utf-8") as f:
            json.dump(
                {
                    "dataset": dataset,
                    "entity_count": len(vocab["entity2id"]),
                    "base_relation_count": len(vocab["relation2id"]) // 2,
                    "relation_count_with_inverse": len(vocab["relation2id"]),
                    "time_count": len(vocab["time2id"]),
                    "train_count": len(splits["train"]),
                    "valid_count": len(splits["valid"]),
                    "test_count": len(splits["test"]),
                    "graph_source": "train_only_cumulative_history",
                },
                f,
                indent=2,
            )

        if torch is not None:
            torch.save(graphs, processed_dir / "time_graphs.pt")
            # Backward-compatible artifact name used by earlier scripts.
            torch.save(graphs, processed_dir / "time_subgraphs.pt")

    def process(self, dataset: str) -> TemporalKGDataset:
        splits_raw = self.load_splits(dataset)
        vocab = self._build_vocab(splits_raw, dataset)

        train = self._map_split(splits_raw["train"], vocab)
        valid = self._map_split(splits_raw["valid"], vocab)
        test = self._map_split(splits_raw["test"], vocab)

        base_relation_num = len(vocab["relation2id"]) // 2
        # Build the historical graph from training events only.  Validation and
        # test targets must not be exposed through preprocessing.
        graphs = self._build_temporal_graphs(
            train,
            len(vocab["entity2id"]),
            base_relation_num,
            timeline=[quad[3] for quad in train + valid + test],
        )
        self.save_processed_data(dataset, vocab, {"train": train, "valid": valid, "test": test}, graphs)

        return TemporalKGDataset(
            dataset=dataset,
            vocab={
                "entity2id": vocab["entity2id"],
                "relation2id": vocab["relation2id"],
                "time2id": vocab["time2id"],
            },
            inverse_relations=vocab["inverse_relations"],
            train=train,
            valid=valid,
            test=test,
            time_graphs=graphs,
        )

    def load_processed(self, dataset: str) -> Optional[TemporalKGDataset]:
        processed_dir = self.save_path / dataset
        vocab_path = processed_dir / "entity_relation_vocab.json"
        graphs_path = processed_dir / "time_graphs.pt"
        legacy_graphs_path = processed_dir / "time_subgraphs.pt"
        splits_path = processed_dir / "splits.json"
        inverse_path = processed_dir / "inverse_relations.json"

        if not (vocab_path.exists() and (graphs_path.exists() or legacy_graphs_path.exists()) and splits_path.exists()):
            return None

        with vocab_path.open("r", encoding="utf-8") as f:
            vocab = json.load(f)
        with inverse_path.open("r", encoding="utf-8") as f:
            inverse_relations = {int(k): int(v) for k, v in json.load(f).items()}
        with splits_path.open("r", encoding="utf-8") as f:
            split_json = json.load(f)

        graph_file = graphs_path if graphs_path.exists() else legacy_graphs_path
        if torch is not None:
            try:
                try:
                    graphs = torch.load(graph_file, map_location="cpu", weights_only=False)
                except TypeError:  # older PyTorch versions
                    graphs = torch.load(graph_file, map_location="cpu")
            except Exception:
                # A processed graph may have been serialized by an unavailable
                # DGL binary.  Rebuild it from the raw split files using the
                # portable graph fallback instead of failing at startup.
                return None
        else:
            graphs = {}
        return TemporalKGDataset(
            dataset=dataset,
            vocab=vocab,
            inverse_relations=inverse_relations,
            train=[tuple(item) for item in split_json.get("train", [])],
            valid=[tuple(item) for item in split_json.get("valid", [])],
            test=[tuple(item) for item in split_json.get("test", [])],
            time_graphs=graphs,
        )


def load_dataset(dataset: str, raw_data_path: str = "./data", processed_path: str = "./data/processed") -> TemporalKGDataset:
    processor = TKGDataProcessor(raw_data_path=raw_data_path, save_path=processed_path)
    loaded = processor.load_processed(dataset)
    if loaded is not None:
        return loaded
    return processor.process(dataset)


if __name__ == "__main__":
    processor = TKGDataProcessor(raw_data_path="./data")
    ds = processor.process("ICEWS18")
    print(ds.dataset, ds.entity_num, ds.relation_num, ds.time_num)
