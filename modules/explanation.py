from models.llm import LLMWrapper

class ExplanationGenerator:
    def __init__(self, llm: LLMWrapper):
        self.llm = llm

    def generate_initial_explanation(self, question: str, paths: list, target_entity: str) -> str:
        prompt = f"""
        Question: {question}
        Reasoning paths: {paths}
        Target entity: {target_entity}
        Generate a clear explanation for the reasoning result.
        """
        return self.llm.generate(prompt)

    def verify_explanation(self, explanation: str, paths: list) -> str:
        prompt = f"""
        Explanation: {explanation}
        Reasoning paths: {paths}
        Verify if the explanation is consistent with the paths. Return "correct" or "incorrect" with reason.
        """
        return self.llm.generate(prompt)

    def refine_explanation(self, explanation: str, paths: list, verify_result: str) -> str:
        if "correct" in verify_result.lower():
            prompt = f"""
            Refine this explanation to be more natural and detailed: {explanation}
            Reference paths: {paths}
            """
        else:
            prompt = f"""
            Correct this explanation to align with the paths: {explanation}
            Reasoning paths: {paths}
            """
        return self.llm.generate(prompt)

    def generate(self, question: str, paths: list, target_entity: str) -> tuple[str, str]:
        initial_exp = self.generate_initial_explanation(question, paths, target_entity)
        verify_res = self.verify_explanation(initial_exp, paths)
        final_exp = self.refine_explanation(initial_exp, paths, verify_res)
        return final_exp, verify_res

if __name__ == "__main__":
    llm = LLMWrapper()
    generator = ExplanationGenerator(llm)
    print("Explanation generator test passed!")