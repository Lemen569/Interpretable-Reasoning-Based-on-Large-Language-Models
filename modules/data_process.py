import os
import json
import pickle
import torch
import dgl
import pandas as pd
from collections import defaultdict


class TKGDataProcessor:
    def __init__(self, raw_data_path: str, save_path: str = "./data/processed"):
        self.raw_data_path = raw_data_path
        self.save_path = save_path
        os.makedirs(save_path, exist_ok=True)

        self.entity2id = {}
        self.relation2id = {}
        self.time2id = {}
        self.inverse_relations = {}
        self.time_subgraphs = defaultdict(dgl.DGLGraph)

    def load_raw_data(self, dataset: str = "ICEWS18"):
        if dataset in ["ICEWS14", "ICEWS05-15", "ICEWS18"]:
            file_path = os.path.join(self.raw_data_path, f"{dataset}.txt")
            df = pd.read_csv(file_path, sep="\t", header=None, names=["head", "relation", "tail", "time"])
            return df.values.tolist()
        else:
            raise ValueError("Unsupported dataset. Choose from ICEWS14/ICEWS05-15/ICEWS18.")

    def build_vocab(self, quadruples):
        entities = set()
        relations = set()
        times = set()
        for h, r, t, tm in quadruples:
            entities.add(h)
            entities.add(t)
            relations.add(r)
            times.add(tm)

        self.entity2id = {e: i for i, e in enumerate(sorted(entities))}
        self.relation2id = {r: i for i, r in enumerate(sorted(relations))}
        self.time2id = {tm: i for i, tm in enumerate(sorted(times))}

    def build_inverse_relations(self):
        for rel, rid in self.relation2id.items():
            if "_inv" not in rel:
                inv_rel = rel + "_inv"
                if inv_rel in self.relation2id:
                    self.inverse_relations[rel] = inv_rel
                    self.inverse_relations[inv_rel] = rel

    def build_time_subgraphs(self, quadruples):
        time_edges = defaultdict(lambda: defaultdict(list))
        for h, r, t, tm in quadruples:
            hid = self.entity2id[h]
            rid = self.relation2id[r]
            tid = self.entity2id[t]
            tmid = self.time2id[tm]
            time_edges[tmid]["h"].append(hid)
            time_edges[tmid]["r"].append(rid)
            time_edges[tmid]["t"].append(tid)

        for tmid, edges in time_edges.items():
            g = dgl.graph((edges["h"], edges["t"]))
            g.edata["rel"] = torch.tensor(edges["r"], dtype=torch.long)
            self.time_subgraphs[tmid] = dgl.add_self_loop(g)

    def save_processed_data(self):
        with open(os.path.join(self.save_path, "entity_relation_vocab.json"), "w") as f:
            json.dump({
                "entity2id": self.entity2id,
                "relation2id": self.relation2id,
                "time2id": self.time2id
            }, f)

        with open(os.path.join(self.save_path, "inverse_relations.json"), "w") as f:
            json.dump(self.inverse_relations, f)

        with open(os.path.join(self.save_path, "time_subgraphs.pkl"), "wb") as f:
            pickle.dump(self.time_subgraphs, f)

    def process(self, dataset: str = "ICEWS18"):
        quadruples = self.load_raw_data(dataset)
        self.build_vocab(quadruples)
        self.build_inverse_relations()
        self.build_time_subgraphs(quadruples)
        self.save_processed_data()
        print("Data processing completed successfully!")


if __name__ == "__main__":
    processor = TKGDataProcessor(raw_data_path="./data/raw")
    processor.process(dataset="ICEWS18")