import torch
from models.llm import LLMWrapper
from utils.beam_search import beam_search


class HTIRRetriever:
    def __init__(self, llm: LLMWrapper):
        self.llm = llm

    def relation_retrieval(
            self,
            time_subgraphs: dict,
            initial_entities: list,
            question: str,
            I: int = 5,
            D: int = 4,
            K: int = 4,
            M: int = 10
    ) -> dict:
        candidate_paths = []
        for e in initial_entities:
            for _ in range(I):
                out_paths = beam_search(time_subgraphs, e, direction="out", depth=D, width=K)
                in_paths = beam_search(time_subgraphs, e, direction="in", depth=D, width=K)
                candidate_paths.extend(out_paths + in_paths)

        candidate_rels = list({p[1] for p in candidate_paths})
        prompt = self._build_prune_prompt(question, initial_entities, candidate_rels, task="relation", top_k=M)
        rel_scores = self.llm.generate_with_score(prompt)[0]
        rel_scores = eval(rel_scores)
        top_rels = dict(sorted(rel_scores.items(), key=lambda x: x[1], reverse=True)[:M])
        return top_rels

    def entity_retrieval(
            self,
            time_subgraphs: dict,
            top_relations: dict,
            question: str,
            I: int = 5,
            D: int = 4,
            K: int = 4,
            M: int = 10
    ) -> tuple[dict, list]:
        candidate_paths = []
        for r, _ in top_relations.items():
            for _ in range(I):
                paths = beam_search(time_subgraphs, relation=r, depth=D, width=K)
                candidate_paths.extend(paths)

        candidate_entities = list({p[-1] for p in candidate_paths})
        prompt = self._build_prune_prompt(question, top_relations, candidate_entities, task="entity", top_k=M)
        ent_scores = self.llm.generate_with_score(prompt)[0]
        ent_scores = eval(ent_scores)
        top_ents = dict(sorted(ent_scores.items(), key=lambda x: x[1], reverse=True)[:M])
        final_paths = [p for p in candidate_paths if p[-1] in top_ents]
        return top_ents, final_paths

    def _build_prune_prompt(self, question: str, context: dict, candidates: list, task: str, top_k: int) -> str:
        return f"""
        Question: {question}
        Context: {context}
        Candidates: {candidates}
        Task: {task}
        Return top {top_k} candidates with scores in JSON format.
        """


if __name__ == "__main__":
    llm = LLMWrapper()
    retriever = HTIRRetriever(llm)
    print("HTIR retrieval module test passed!")