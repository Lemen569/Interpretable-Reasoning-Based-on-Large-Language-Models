from .beamsearch import BeamSearch


def beam_search(graph, start_entity=None, relation=None, direction="out", depth=4, width=4, llm_question=None, llm=None):
    if llm is None:
        raise ValueError("llm must be provided")
    searcher = BeamSearch(graph=graph, llm=llm, depth=depth, width=width)
    question = llm_question or ""
    return searcher.search(start_entity=start_entity, question=question, relation=relation, direction=direction)

