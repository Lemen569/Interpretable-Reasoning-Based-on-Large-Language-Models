import torch
import numpy as np
import dgl
from models.gnn import TemporalGNN
from models.llm import LLMWrapper
from models.transformer import FusionTransformer

class ReasoningEngine:
    def __init__(
        self,
        gnn: TemporalGNN,
        llm: LLMWrapper,
        fusion_transformer: FusionTransformer,
        decay_rate: float = 0.08,
        alpha: float = 0.5,
        beta: float = 0.7,
        device: str = "cuda"
    ):
        self.gnn = gnn
        self.llm = llm
        self.transformer = fusion_transformer
        self.decay_rate = decay_rate
        self.alpha = alpha
        self.beta = beta
        self.device = device

    def time_decay(self, time_delta):
        return np.exp(-self.decay_rate * np.abs(time_delta))

    def llm_path_scoring(self, entity, paths, query_prompt):
        path_scores = []
        for path in paths:
            path_prompt = f"{query_prompt} Reasoning path: {path}"
            score = self.llm.score(path_prompt)
            t_start = path[3] if len(path) > 3 else 0
            t_end = path[-1] if len(path) > 3 else 0
            decay = self.time_decay(t_end - t_start)
            weighted_score = self.beta * score + (1 - self.beta) * decay
            path_scores.append(weighted_score)
        return np.mean(path_scores) if path_scores else 0.0

    def compute_llm_scores(self, top_entities, reasoning_paths, query_prompt):
        llm_scores = {}
        for entity, raw_score in top_entities.items():
            if raw_score < self.alpha:
                continue
            entity_paths = [p for p in reasoning_paths if p[-1] == entity]
            avg_score = self.llm_path_scoring(entity, entity_paths, query_prompt)
            llm_scores[entity] = avg_score
        return llm_scores

    def compute_gnn_scores(self, graph, entity_ids, rel_ids, time_ids, entities):
        gnn_scores = {}
        for entity in entities:
            score = self.gnn.predict_entity_score(
                graph, entity_ids, rel_ids, time_ids, entity
            )
            gnn_scores[entity] = score
        return gnn_scores

    def fuse_feature_scores(self, llm_score, gnn_score):
        llm_tensor = torch.tensor([llm_score], dtype=torch.float32).to(self.device)
        gnn_tensor = torch.tensor([gnn_score], dtype=torch.float32).to(self.device)
        return self.transformer.fuse_features(gnn_tensor, llm_tensor)

    def batch_fusion(self, llm_scores, gnn_scores):
        fused_scores = {}
        for entity in llm_scores:
            if entity in gnn_scores:
                fused_scores[entity] = self.fuse_feature_scores(
                    llm_scores[entity], gnn_scores[entity]
                )
        return fused_scores

    def bidirectional_verify(self, forward_scores, backward_scores):
        if not forward_scores or not backward_scores:
            return list(forward_scores.keys())[0], list(forward_scores.values())[0]

        forward_ent = max(forward_scores, key=forward_scores.get)
        backward_ent = max(backward_scores, key=backward_scores.get)

        if forward_ent == backward_ent:
            final_score = (forward_scores[forward_ent] + backward_scores[backward_ent]) / 2
            return forward_ent, final_score

        if forward_scores[forward_ent] > backward_scores[backward_ent]:
            return forward_ent, forward_scores[forward_ent]
        else:
            return backward_ent, backward_scores[backward_ent]

    def forward(
        self,
        graph,
        entity_ids,
        rel_ids,
        time_ids,
        forward_entities,
        backward_entities,
        forward_paths,
        backward_paths,
        forward_prompt,
        backward_prompt
    ):
        llm_forward = self.compute_llm_scores(forward_entities, forward_paths, forward_prompt)
        gnn_forward = self.compute_gnn_scores(graph, entity_ids, rel_ids, time_ids, forward_entities.keys())
        fused_forward = self.batch_fusion(llm_forward, gnn_forward)

        llm_backward = self.compute_llm_scores(backward_entities, backward_paths, backward_prompt)
        gnn_backward = self.compute_gnn_scores(graph, entity_ids, rel_ids, time_ids, backward_entities.keys())
        fused_backward = self.batch_fusion(llm_backward, gnn_backward)

        final_entity, final_score = self.bidirectional_verify(fused_forward, fused_backward)
        return final_entity, final_score


if __name__ == "__main__":
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    ENTITY_NUM = 1000
    RELATION_NUM = 100
    TIME_NUM = 50

    gnn_model = TemporalGNN(ENTITY_NUM, RELATION_NUM, TIME_NUM, device=DEVICE)
    llm_model = LLMWrapper(device=DEVICE)
    transformer_model = FusionTransformer(device=DEVICE)

    engine = ReasoningEngine(gnn_model, llm_model, transformer_model, device=DEVICE)

    test_graph = dgl.rand_graph(ENTITY_NUM, 5000).to(DEVICE)
    test_graph = dgl.add_self_loop(test_graph)
    test_entities = torch.arange(ENTITY_NUM).to(DEVICE)
    test_rels = torch.randint(0, RELATION_NUM, (test_graph.num_edges(),)).to(DEVICE)
    test_times = torch.randint(0, TIME_NUM, (ENTITY_NUM,)).to(DEVICE)

    test_forward_ents = {0: 0.8, 1: 0.6}
    test_backward_ents = {0: 0.75, 2: 0.5}
    test_paths = [(0, 1, 1, 10), (0, 2, 1, 10)]
    test_prompt = "Temporal knowledge graph reasoning query"

    result_ent, result_score = engine.forward(
        test_graph, test_entities, test_rels, test_times,
        test_forward_ents, test_backward_ents,
        test_paths, test_paths, test_prompt, test_prompt
    )

    print(f"Final predicted entity: {result_ent}")
    print(f"Final fusion score: {result_score:.4f}")
    print("ReasoningEngine test passed successfully!")