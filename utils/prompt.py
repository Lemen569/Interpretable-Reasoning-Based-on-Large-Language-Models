from typing import List, Tuple


class PromptTemplates:
    """
    All prompt templates used in TKL-XR, strictly aligned with the paper's prompt design
    """

    @staticmethod
    def generate_forward_question(quadruple: Tuple[str, str, str, str]) -> str:
        e_s, r, e_o, t = quadruple
        return f"""
Question: Given the temporal knowledge graph, which entity is most likely to be the object of the relation "{r}" between "{e_s}" at time {t}?
Provide only the entity name as the answer.
""".strip()

    @staticmethod
    def generate_backward_question(quadruple: Tuple[str, str, str, str], inverse_rel: str) -> str:
        e_s, r, e_o, t = quadruple
        return f"""
Question: Given the temporal knowledge graph, which entity is most likely to be the subject of the relation "{inverse_rel}" with "{e_o}" at time {t}?
Provide only the entity name as the answer.
""".strip()

    @staticmethod
    def prune_relation_candidates(question: str, entities: List[str], relations: List[str], top_k: int) -> str:
        return f"""
Task: Relation Pruning
Question: {question}
Context entities: {entities}
Candidate relations: {relations}
Select the top {top_k} most relevant relations for answering the question.
Return a JSON object where keys are relation names and values are relevance scores (0-1).
Example: {{"meet": 0.92, "consult": 0.78}}
""".strip()

    @staticmethod
    def prune_entity_candidates(question: str, relations: List[str], entities: List[str], top_k: int) -> str:
        return f"""
Task: Entity Pruning
Question: {question}
Context relations: {relations}
Candidate entities: {entities}
Select the top {top_k} most relevant entities for answering the question.
Return a JSON object where keys are entity names and values are relevance scores (0-1).
Example: {{"China": 0.95, "UK": 0.81}}
""".strip()

    @staticmethod
    def generate_initial_explanation(question: str, paths: List[List[Tuple]], target_entity: str) -> str:
        path_str = "\n".join([f"Path {i + 1}: {' -> '.join([f'{e} via {r} at {t}' for e, r, t in path])}" for i, path in
                              enumerate(paths)])
        return f"""
Task: Explanation Generation
Question: {question}
Reasoning paths:
{path_str}
Target entity: {target_entity}
Generate a clear, concise explanation for why this target entity is the answer, based on the reasoning paths.
""".strip()

    @staticmethod
    def verify_explanation(explanation: str, paths: List[List[Tuple]]) -> str:
        path_str = "\n".join([f"Path {i + 1}: {' -> '.join([f'{e} via {r} at {t}' for e, r, t in path])}" for i, path in
                              enumerate(paths)])
        return f"""
Task: Explanation Verification
Explanation: {explanation}
Reasoning paths:
{path_str}
Is this explanation consistent with the reasoning paths? Answer with "correct" or "incorrect", followed by a brief reason.
""".strip()

    @staticmethod
    def refine_explanation(explanation: str, paths: List[List[Tuple]]) -> str:
        path_str = "\n".join([f"Path {i + 1}: {' -> '.join([f'{e} via {r} at {t}' for e, r, t in path])}" for i, path in
                              enumerate(paths)])
        return f"""
Task: Explanation Refinement
Current explanation: {explanation}
Reasoning paths:
{path_str}
Refine this explanation to be more natural, detailed, and aligned with the reasoning paths. Keep it concise.
""".strip()

    @staticmethod
    def correct_explanation(explanation: str, paths: List[List[Tuple]], question: str) -> str:
        path_str = "\n".join([f"Path {i + 1}: {' -> '.join([f'{e} via {r} at {t}' for e, r, t in path])}" for i, path in
                              enumerate(paths)])
        return f"""
Task: Explanation Correction
Current explanation: {explanation}
Reasoning paths:
{path_str}
Question: {question}
Correct this explanation to align with the reasoning paths and answer the question accurately.
""".strip()


if __name__ == "__main__":
    quadruple = ("USA", "meet", "China", "2023-01-01")
    forward_q = PromptTemplates.generate_forward_question(quadruple)
    print("Forward question prompt:\n", forward_q)

    backward_q = PromptTemplates.generate_backward_question(quadruple, "met_by")
    print("\nBackward question prompt:\n", backward_q)

    prune_prompt = PromptTemplates.prune_relation_candidates("Test question", ["USA"], ["meet", "consult"], 2)
    print("\nRelation pruning prompt:\n", prune_prompt)
    print("PromptTemplates test passed successfully!")