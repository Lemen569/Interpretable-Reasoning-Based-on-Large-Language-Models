import torch
from models.llm import LLMWrapper
from utils.beam_search import beam_search

class HistoryInitializer:
    def __init__(self, llm: LLMWrapper, inverse_relations: dict):
        self.llm = llm
        self.inverse_relations = inverse_relations

    def generate_question(self, quadruple: tuple, inverse: bool = False) -> str:
        e_s, r, e_o, t = quadruple
        if not inverse:
            return f"Which entity is most likely to be the object of the relation '{r}' between '{e_s}' at time {t}?"
        else:
            r_inv = self.inverse_relations.get(r, r + "_inv")
            return f"Which entity is most likely to be the subject of the relation '{r_inv}' with '{e_o}' at time {t}?"

    def retrieve_initial_entities(
        self,
        time_subgraphs: dict,
        start_entity: str,
        question: str,
        beam_depth: int = 4,
        beam_width: int = 4
    ) -> list:
        start_eid = next(k for k, v in self.llm.entity2id.items() if v == start_entity)
        initial_entities = beam_search(
            graph=time_subgraphs,
            start_entity=start_eid,
            depth=beam_depth,
            width=beam_width,
            llm_question=question,
            llm=self.llm
        )
        initial_entities = list(set(initial_entities))
        return initial_entities

if __name__ == "__main__":
    llm = LLMWrapper()
    inv_rels = {"advise": "advised_by", "advised_by": "advise"}
    initializer = HistoryInitializer(llm, inv_rels)
    test_quad = ("USA", "meet", "China", "2023-01-01")
    q = initializer.generate_question(test_quad)
    print(f"Generated question: {q}")
    print("History initialization test passed!")