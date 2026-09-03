from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

try:  # pragma: no cover - optional runtime dependency
    import torch
except Exception:  # pragma: no cover
    torch = None


PathStep = Tuple[int, int, int]
Path = List[PathStep]


@dataclass
class ScoredPath:
    path: Path
    score: float


class BeamSearch:
    """Structural beam search used by the Travel phase of HTIR.

    Each path step is represented as ``(entity_id, relation_id, time_id)``.
    The implementation is intentionally deterministic when the LLM wrapper is
    in mock mode, which makes preprocessing and smoke tests reproducible.
    """

    def __init__(
        self,
        graph: Dict[int, object],
        llm=None,
        depth: int = 4,
        width: int = 4,
    ):
        self.graph = graph
        self.llm = llm
        self.depth = depth
        self.width = width
        self._outgoing: Dict[int, List[PathStep]] = {}
        self._incoming: Dict[int, List[PathStep]] = {}
        self._build_index()

    def _build_index(self):
        outgoing = {}
        incoming = {}
        for _, graph in sorted(self.graph.items(), key=lambda item: item[0]):
            src, dst, rel_ids, time_ids = self._edge_lists(graph)
            for head, tail, rel_id, edge_time in zip(src, dst, rel_ids, time_ids):
                outgoing.setdefault(int(head), []).append((int(tail), int(rel_id), int(edge_time)))
                incoming.setdefault(int(tail), []).append((int(head), int(rel_id), int(edge_time)))
        self._outgoing = {
            key: list(dict.fromkeys(value)) for key, value in outgoing.items()
        }
        self._incoming = {
            key: list(dict.fromkeys(value)) for key, value in incoming.items()
        }

    @staticmethod
    def _edge_lists(graph):
        src, dst = graph.edges()
        src = src.tolist() if hasattr(src, "tolist") else list(src)
        dst = dst.tolist() if hasattr(dst, "tolist") else list(dst)
        rel = graph.edata.get("rel")
        rel = rel.tolist() if rel is not None and hasattr(rel, "tolist") else list(rel or [0] * len(src))
        time = graph.edata.get("time")
        time = time.tolist() if time is not None and hasattr(time, "tolist") else list(time or [0] * len(src))
        return src, dst, rel, time

    def _neighbors(self, entity: int, relation: Optional[int], direction: str) -> List[PathStep]:
        index = self._outgoing if direction == "out" else self._incoming
        neighbors = index.get(int(entity), [])
        if relation is None:
            return list(neighbors)
        return [item for item in neighbors if item[1] == int(relation)]

    def _score_path(self, path: Path, question: str) -> float:
        if self.llm is None:
            return 0.0
        text = " -> ".join(f"e={e},r={r},t={t}" for e, r, t in path)
        return float(self.llm.score(f"Question: {question}\nPath: {text}"))

    def search(
        self,
        start_entity: int,
        question: str = "",
        relation: Optional[int] = None,
        direction: str = "out",
        depth: Optional[int] = None,
        width: Optional[int] = None,
    ) -> List[Path]:
        depth = self.depth if depth is None else max(1, int(depth))
        width = self.width if width is None else max(1, int(width))
        if direction not in {"out", "in"}:
            raise ValueError("direction must be 'out' or 'in'")

        beam: List[ScoredPath] = [ScoredPath([(int(start_entity), -1, -1)], 0.0)]
        for _ in range(depth):
            candidates: List[ScoredPath] = []
            for item in beam:
                current = item.path[-1][0]
                for next_entity, rel_id, time_id in self._neighbors(current, relation, direction):
                    if next_entity in [step[0] for step in item.path]:
                        continue
                    new_path = item.path + [(next_entity, rel_id, time_id)]
                    local_score = self._score_path(new_path, question)
                    candidates.append(ScoredPath(new_path, item.score + local_score))
            candidates.sort(key=lambda item: item.score, reverse=True)
            beam = candidates[:width]
            if not beam:
                break
        return [item.path for item in beam]
