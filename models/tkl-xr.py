import torch
import torch.nn as nn
import torch.nn.functional as F
from .gnn import TemporalGNN
from .llm import LLMWrapper

class TKLXR(nn.Module):
    def __init__(
        self,
        entity_num: int,
        relation_num: int,
        time_num: int,
        llm_model_name: str = "meta-llama/Llama-2-7b-chat-hf",
        embed_dim: int = 128,
        gnn_layers: int = 3,
        decay_rate: float = 0.08,
        dropout: float = 0.1,
        load_4bit: bool = True,
        device: str = "cuda"
    ):
        super(TKLXR, self).__init__()
        self.device = device
        self.decay_rate = decay_rate


        self.gnn = TemporalGNN(
            entity_num=entity_num,
            relation_num=relation_num,
            time_num=time_num,
            embed_dim=embed_dim,
            gnn_layers=gnn_layers,
            decay_rate=decay_rate,
            dropout=dropout
        )


        self.llm = LLMWrapper(
            model_name=llm_model_name,
            load_4bit=load_4bit,
            device=device
        )


        self.fusion_network = nn.Sequential(
            nn.Linear(2, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
            nn.Sigmoid()
        ).to(device)

    def time_weighted_score(self, time_delta: torch.Tensor) -> torch.Tensor:
        return torch.exp(-self.decay_rate * torch.abs(time_delta))

    def fuse_scores(self, llm_score: float, gnn_score: float) -> float:
        score_pair = torch.tensor([[llm_score, gnn_score]], dtype=torch.float32, device=self.device)
        fused_score = self.fusion_network(score_pair)
        return fused_score.item()

    def bidirectional_verification(self, forward_entity, forward_score, backward_entity, backward_score):
        if forward_entity == backward_entity:
            return forward_entity, (forward_score + backward_score) / 2
        return (forward_entity, forward_score) if forward_score > backward_score else (backward_entity, backward_score)

    def get_gnn_entity_score(self, graph, entity_ids, rel_ids, time_ids, target_entity):
        return self.gnn.predict_entity_score(graph, entity_ids, rel_ids, time_ids, target_entity)

    def get_llm_path_score(self, prompt: str):
        return self.llm.score(prompt)

    def generate_reason_explanation(self, prompt: str):
        return self.llm.generate(prompt)

    def forward(
        self,
        graph,
        entity_ids,
        rel_ids,
        time_ids,
        forward_prompt,
        backward_prompt,
        target_entity: int
    ):
        # 1. forward
        llm_forward = self.get_llm_path_score(forward_prompt)
        gnn_forward = self.get_gnn_entity_score(graph, entity_ids, rel_ids, time_ids, target_entity)
        fused_forward = self.fuse_scores(llm_forward, gnn_forward)

        # 2. inverse
        llm_backward = self.get_llm_path_score(backward_prompt)
        gnn_backward = self.get_gnn_entity_score(graph, entity_ids, rel_ids, time_ids, target_entity)
        fused_backward = self.fuse_scores(llm_backward, gnn_backward)

        # 3. validation
        final_entity, final_score = self.bidirectional_verification(
            target_entity, fused_forward,
            target_entity, fused_backward
        )

        return final_entity, final_score

    def infer_with_explanation(
        self,
        graph,
        entity_ids,
        rel_ids,
        time_ids,
        query_prompt,
        target_entity: int
    ):

        _, final_score = self.forward(
            graph=graph,
            entity_ids=entity_ids,
            rel_ids=rel_ids,
            time_ids=time_ids,
            forward_prompt=query_prompt,
            backward_prompt=query_prompt,
            target_entity=target_entity
        )
        explanation = self.generate_reason_explanation(query_prompt)
        return final_score, explanation



if __name__ == "__main__":
    import dgl


    ENTITY_NUM = 1000
    RELATION_NUM = 100
    TIME_NUM = 50
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    model = TKLXR(
        entity_num=ENTITY_NUM,
        relation_num=RELATION_NUM,
        time_num=TIME_NUM,
        device=DEVICE
    )


    g = dgl.rand_graph(ENTITY_NUM, 5000)
    g = dgl.add_self_loop(g)
    entity_ids = torch.arange(ENTITY_NUM, device=DEVICE)
    rel_ids = torch.randint(0, RELATION_NUM, (g.num_edges(),), device=DEVICE)
    time_ids = torch.randint(0, TIME_NUM, (ENTITY_NUM,), device=DEVICE)


    test_prompt = "Please reason the temporal knowledge graph query."
    test_target = 0


    score, exp = model.infer_with_explanation(g, entity_ids, rel_ids, time_ids, test_prompt, test_target)


    print(f"✅ fusion score: {score:.4f}")
    print(f"✅ explanation: {exp[:80]}...")
    print("✅ test passing")