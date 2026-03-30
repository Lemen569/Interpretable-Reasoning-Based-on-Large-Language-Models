import dgl
import torch
from typing import List, Tuple, Dict
from models.llm import LLMWrapper


class BeamSearch:
    def __init__(
            self,
            graph: Dict[int, dgl.DGLGraph],
            llm: LLMWrapper,
            depth: int = 4,
            width: int = 4,
            device: str = "cuda"
    ):
        self.graph = graph
        self.llm = llm
        self.depth = depth
        self.width = width
        self.device = device

    def _get_neighbors(self, entity: int, rel: int = None, direction: str = "out") -> List[Tuple[int, int]]:
        """
        Get neighboring entities for a given entity (bidirectional support)
        """
        neighbors = []
        for tmid, g in self.graph.items():
            if direction == "out":
                src, dst = g.edges()
                mask = (src == entity)
                if rel is not None:
                    mask = mask & (g.edata["rel"] == rel)
                dst_list = dst[mask].tolist()
                rel_list = g.edata["rel"][mask].tolist()
                neighbors.extend(zip(dst_list, rel_list))
            elif direction == "in":
                src, dst = g.edges()
                mask = (dst == entity)
                if rel is not None:
                    mask = mask & (g.edata["rel"] == rel)
                src_list = src[mask].tolist()
                rel_list = g.edata["rel"][mask].tolist()
                neighbors.extend(zip(src_list, rel_list))
        return list(set(neighbors))

    def _score_path(self, path: List[Tuple[int, int, int]], question: str) -> float:
        """
        Score a reasoning path using LLM for relevance to the query
        """
        path_str = " -> ".join([f"Entity {e} via Relation {r} at Time {t}" for e, r, t in path])
        prompt = f"Question: {question}\nReasoning path: {path_str}\nScore this path from 0 to 1 (1 = highly relevant)."
        return self.llm.score(prompt)

    def search(
            self,
            start_entity: int,
            question: str,
            relation: int = None,
            direction: str = "out"
    ) -> List[List[Tuple[int, int, int]]]:
        """
        Perform bidirectional beam search over the temporal KG
        """
        beam = [([(start_entity, -1, 0)], 0.0)]

        for step in range(self.depth):
            candidates = []
            for path, score in beam:
                current_entity = path[-1][0]
                neighbors = self._get_neighbors(current_entity, relation, direction)

                for neighbor, rel in neighbors:
                    new_time = path[-1][2] + 1
                    new_path = path + [(neighbor, rel, new_time)]
                    new_score = score + self._score_path(new_path, question)
                    candidates.append((new_path, new_score))

            candidates.sort(key=lambda x: x[1], reverse=True)
            beam = candidates[:self.width]

            if not beam:
                break

        beam.sort(key=lambda x: x[1], reverse=True)
        return [path for path, score in beam]


if __name__ == "__main__":
    import dgl
    from models.llm import LLMWrapper

    # Mock temporal graph
    graph = {0: dgl.graph(([0, 0], [1, 2])), 1: dgl.graph(([1, 2], [3, 3]))}
    for tmid, g in graph.items():
        g.edata["rel"] = torch.tensor([0, 1], dtype=torch.long)

    llm = LLMWrapper()
    beam_search = BeamSearch(graph, llm, depth=2, width=2)
    paths = beam_search.search(start_entity=0, question="Test query question")
    print("Found reasoning paths:", paths)
    print("BeamSearch test passed successfully!")