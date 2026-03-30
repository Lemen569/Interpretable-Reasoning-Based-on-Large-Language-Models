import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
import dgl.nn as dgl_nn
import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

class TemporalGNN(nn.Module):
    def __init__(
        self,
        entity_num: int,
        relation_num: int,
        time_num: int,
        embed_dim: int = 128,
        gnn_layers: int = 3,
        decay_rate: float = 0.08,
        dropout: float = 0.1
    ):
        super(TemporalGNN, self).__init__()
        self.entity_num = entity_num
        self.relation_num = relation_num
        self.time_num = time_num
        self.embed_dim = embed_dim
        self.decay_rate = decay_rate
        self.dropout = nn.Dropout(dropout)

        # 嵌入层
        self.entity_embedding = nn.Embedding(entity_num, embed_dim)
        self.relation_embedding = nn.Embedding(relation_num, embed_dim)
        self.time_embedding = nn.Embedding(time_num, embed_dim)

        # 嵌入初始化
        nn.init.xavier_uniform_(self.entity_embedding.weight)
        nn.init.xavier_uniform_(self.relation_embedding.weight)
        nn.init.xavier_uniform_(self.time_embedding.weight)

        # 多层RGCN卷积层
        self.rgcn_layers = nn.ModuleList()
        for i in range(gnn_layers):
            self.rgcn_layers.append(
                dgl_nn.RGCNConv(
                    in_feat=embed_dim,
                    out_feat=embed_dim,
                    num_rels=relation_num,
                    regularizer="basis",
                    num_bases=10
                )
            )

        # 特征融合与预测层
        self.time_fusion = nn.Linear(embed_dim * 2, embed_dim)
        self.predictor = nn.Linear(embed_dim, 1)

    # 时间衰减函数 - 论文公式2: exp(-λ * |t_x - t_y|)
    def time_decay(self, time_delta: torch.Tensor) -> torch.Tensor:
        return torch.exp(-self.decay_rate * torch.abs(time_delta))

    def forward(
        self,
        g: dgl.DGLGraph,
        entity_ids: torch.Tensor,
        rel_ids: torch.Tensor,
        time_ids: torch.Tensor
    ) -> torch.Tensor:
        # 基础嵌入获取
        e_emb = self.entity_embedding(entity_ids)
        t_emb = self.time_embedding(time_ids)

        # 时间-实体特征融合
        et_emb = torch.cat([e_emb, t_emb], dim=-1)
        et_emb = self.time_fusion(et_emb)
        et_emb = self.dropout(F.relu(et_emb))

        # RGCN特征传播
        h = et_emb
        for layer in self.rgcn_layers:
            h = layer(g, h, rel_ids)
            h = self.dropout(F.relu(h))
        return h

    # GNN打分函数 - 论文4.3节
    def compute_score(
        self,
        g: dgl.DGLGraph,
        entity_ids: torch.Tensor,
        rel_ids: torch.Tensor,
        time_ids: torch.Tensor,
        head_ids: torch.Tensor,
        tail_ids: torch.Tensor
    ) -> torch.Tensor:
        entity_emb = self.forward(g, entity_ids, rel_ids, time_ids)
        h_emb = entity_emb[head_ids]
        t_emb = entity_emb[tail_ids]
        r_emb = self.relation_embedding(rel_ids)

        # 计算时间衰减权重
        time_diff = time_ids.unsqueeze(0) - time_ids.unsqueeze(1)
        decay_weight = self.time_decay(time_diff)

        # 时序感知打分
        score = torch.sum(h_emb * r_emb * t_emb * decay_weight, dim=-1, keepdim=True)
        return torch.sigmoid(score)

    # 单实体打分 - 供融合推理模块调用
    def predict_entity_score(
        self,
        g: dgl.DGLGraph,
        entity_ids: torch.Tensor,
        rel_ids: torch.Tensor,
        time_ids: torch.Tensor,
        target_entity: int
    ) -> float:
        entity_emb = self.forward(g, entity_ids, rel_ids, time_ids)
        target_emb = entity_emb[target_entity]
        score = self.predictor(target_emb).item()
        return sigmoid(score)


if __name__ == "__main__":

    ENTITY_NUM = 1000
    RELATION_NUM = 100
    TIME_NUM = 50
    EMBED_DIM = 128

    # 模型初始化
    model = TemporalGNN(ENTITY_NUM, RELATION_NUM, TIME_NUM, EMBED_DIM)

    g = dgl.rand_graph(ENTITY_NUM, 5000)
    g = dgl.add_self_loop(g)
    entity_ids = torch.arange(ENTITY_NUM)
    rel_ids = torch.randint(0, RELATION_NUM, (g.num_edges(),))
    time_ids = torch.randint(0, TIME_NUM, (ENTITY_NUM,))


    entity_embeddings = model(g, entity_ids, rel_ids, time_ids)
    print(f"实体嵌入形状: {entity_embeddings.shape}")

    test_score = model.compute_score(g, entity_ids, rel_ids, time_ids, torch.tensor([0]), torch.tensor([1]))
    print(f"实体对打分: {test_score.item():.4f}")
    print("✅ TemporalGNN 模型测试通过！")