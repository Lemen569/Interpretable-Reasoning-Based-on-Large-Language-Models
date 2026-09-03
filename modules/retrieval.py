from __future__ import annotations

from collections import defaultdict
from typing import Dict, Iterable, List, Optional, Tuple

from models.llm import LLMWrapper
from utils.beam_search import beam_search
from utils.prompt_templates import PromptTemplates


class HTIRRetriever:
    """Two-stage Travel--Prune retrieval used by TKL-XR."""

    def __init__(
        self,
        llm: LLMWrapper,
        beam_depth: int = 4,
        beam_width: int = 4,
        retrieval_rounds: int = 1,
        top_relations: int = 10,
        top_entities: int = 10,
    ):
        self.llm = llm
        self.beam_depth = beam_depth
        self.beam_width = beam_width
        self.retrieval_rounds = retrieval_rounds
        self.top_relations = top_relations
        self.top_entities = top_entities

    def _prune(self, question: str, candidates: Iterable[int], task: str, top_k: int) -> Dict[int, float]:
        candidates = list(dict.fromkeys(int(candidate) for candidate in candidates))
        if not candidates:
            return {}
        prompt = (
            f"Question: {question}\n"
            f"Task: {task} pruning\n"
            f"Candidates: {candidates}\n"
            f"Return a relevance score in [0,1] for each candidate."
        )
        scores = self.llm.score_candidates(prompt, [str(candidate) for candidate in candidates])
        raw = {int(key): max(0.0, float(value)) for key, value in scores.items()}
        total = sum(raw.values())
        normalized = {key: value / total for key, value in raw.items()} if total > 0 else raw
        return dict(sorted(normalized.items(), key=lambda item: item[1], reverse=True)[:top_k])

    def relation_retrieval(
        self,
        time_subgraphs: Dict[int, object],
        initial_entities: Iterable[int],
        question: str,
        I: Optional[int] = None,
        D: Optional[int] = None,
        K: Optional[int] = None,
        M: Optional[int] = None,
    ) -> Tuple[Dict[int, float], List[List[Tuple[int, int, int]]]]:
        rounds = self.retrieval_rounds if I is None else int(I)
        depth = self.beam_depth if D is None else int(D)
        width = self.beam_width if K is None else int(K)
        top_k = self.top_relations if M is None else int(M)

        paths: List[List[Tuple[int, int, int]]] = []
        for _ in range(max(1, rounds)):
            for entity in initial_entities:
                paths.extend(
                    beam_search(
                        time_subgraphs,
                        start_entity=int(entity),
                        direction="out",
                        depth=depth,
                        width=width,
                        llm_question=question,
                        llm=self.llm,
                    )
                )
                paths.extend(
                    beam_search(
                        time_subgraphs,
                        start_entity=int(entity),
                        direction="in",
                        depth=depth,
                        width=width,
                        llm_question=question,
                        llm=self.llm,
                    )
                )

        candidate_relations = [step[1] for path in paths for step in path if step[1] >= 0]
        relation_scores = self._prune(question, candidate_relations, "relation", top_k)
        return relation_scores, paths

    def entity_retrieval(
        self,
        time_subgraphs: Dict[int, object],
        top_relations: Dict[int, float],
        question: str,
        start_entities: Optional[Iterable[int]] = None,
        I: Optional[int] = None,
        D: Optional[int] = None,
        K: Optional[int] = None,
        M: Optional[int] = None,
    ) -> Tuple[Dict[int, float], List[List[Tuple[int, int, int]]]]:
        rounds = self.retrieval_rounds if I is None else int(I)
        depth = self.beam_depth if D is None else int(D)
        width = self.beam_width if K is None else int(K)
        top_k = self.top_entities if M is None else int(M)
        starts = list(start_entities or [0])

        paths: List[List[Tuple[int, int, int]]] = []
        for _ in range(max(1, rounds)):
            for start_entity in starts:
                for relation in top_relations:
                    paths.extend(
                        beam_search(
                            time_subgraphs,
                            start_entity=int(start_entity),
                            relation=int(relation),
                            direction="out",
                            depth=depth,
                            width=width,
                            llm_question=question,
                            llm=self.llm,
                        )
                    )

        candidate_entities = [path[-1][0] for path in paths if path]
        entity_scores = self._prune(question, candidate_entities, "entity", top_k)
        retained = set(entity_scores)
        final_paths = [path for path in paths if path and path[-1][0] in retained]
        return entity_scores, final_paths

    def retrieve(
        self,
        time_subgraphs: Dict[int, object],
        initial_entities: Iterable[int],
        question: str,
    ) -> Dict[str, object]:
        relation_scores, relation_paths = self.relation_retrieval(
            time_subgraphs, initial_entities, question
        )
        entity_scores, entity_paths = self.entity_retrieval(
            time_subgraphs,
            relation_scores,
            question,
            start_entities=initial_entities,
        )
        return {
            "relation_scores": relation_scores,
            "entity_scores": entity_scores,
            "relation_paths": relation_paths,
            "entity_paths": entity_paths,
        }
