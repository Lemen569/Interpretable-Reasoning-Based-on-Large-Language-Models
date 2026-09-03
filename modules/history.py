from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Tuple

from models.llm import LLMWrapper
from utils.beam_search import beam_search


class HistoryInitializer:
    def __init__(
        self,
        llm: Optional[LLMWrapper] = None,
        inverse_relations: Optional[Dict[int, int]] = None,
        entity2id: Optional[Dict[str, int]] = None,
        relation2id: Optional[Dict[str, int]] = None,
    ):
        self.llm = llm
        self.inverse_relations = inverse_relations or {}
        self.entity2id = entity2id or {}
        self.relation2id = relation2id or {}
        self.id2entity = {value: key for key, value in self.entity2id.items()}
        self.id2relation = {value: key for key, value in self.relation2id.items()}

    def generate_question(self, quadruple: tuple, inverse: bool = False) -> str:
        head, relation, tail, time = quadruple
        if not inverse:
            return (
                f'Which entity is most likely to be the object of relation '
                f'"{relation}" for "{head}" at time {time}?'
            )
        inverse_relation = self.inverse_relations.get(relation, relation)
        if isinstance(inverse_relation, int):
            inverse_relation = self.id2relation.get(inverse_relation, inverse_relation)
        return (
            f'Which entity is most likely to be the subject of relation '
            f'"{inverse_relation}" for "{tail}" at time {time}?'
        )

    def retrieve_initial_entities(
        self,
        time_subgraphs: Dict[int, object],
        start_entity: int | str,
        question: str,
        beam_depth: int = 4,
        beam_width: int = 4,
        query_time: Optional[int] = None,
    ) -> List[int]:
        if isinstance(start_entity, str):
            if start_entity not in self.entity2id:
                raise KeyError(f"Unknown entity: {start_entity}")
            start_entity = self.entity2id[start_entity]
        # History initialization follows the paper's time-ordered procedure:
        # search each historical subgraph in turn, retain the top-K paths with
        # LLM scores, and propagate their tail entities to the next time.
        available_times = sorted(time_subgraphs)
        if query_time is not None:
            available_times = [time_id for time_id in available_times if time_id < int(query_time)]
        active_entities = {int(start_entity)}
        retained_paths = []
        for time_id in available_times:
            graph = {time_id: time_subgraphs[time_id]}
            candidates = []
            for entity in sorted(active_entities):
                candidates.extend(
                    beam_search(
                        graph=graph,
                        start_entity=entity,
                        depth=beam_depth,
                        width=beam_width,
                        llm_question=question,
                        llm=self.llm,
                    )
                )
            if not candidates:
                continue
            if self.llm is not None:
                scored = [
                    (
                        self.llm.score(
                            f"{question}\nHistorical path: "
                            + " -> ".join(f"({e},{r},{t})" for e, r, t in path)
                        ),
                        path,
                    )
                    for path in candidates
                ]
                scored.sort(key=lambda item: item[0], reverse=True)
                candidates = [path for _, path in scored[:beam_width]]
            retained_paths.extend(candidates)
            active_entities = {path[-1][0] for path in candidates if path}
            if not active_entities:
                break

        entities = {path[-1][0] for path in retained_paths if path}
        entities.add(int(start_entity))
        return sorted(entities)
