# Interpretable Reasoning Based on Large Language Models

This repository contains the implementation of TKL-XR, including history
initialization, HTIR Travel--Prune retrieval, GNN/LLM fusion, bidirectional
verification, and explanation generation.

## Data layout

Place each benchmark under `data/<dataset>/` with:

```text
train.txt
valid.txt
test.txt
entity2id.txt        # optional when split files already contain IDs
relation2id.txt      # optional when split files already contain IDs
```

The preprocessing code builds contiguous IDs, adds inverse relations, builds
cumulative temporal graphs, and writes the following files to
`data/processed/<dataset>/`:

```text
entity_relation_vocab.json
inverse_relations.json
splits.json
time_graphs.pt
time_subgraphs.pt       # backward-compatible alias
```

The supplied MIMIC examples are read directly from
`data/mimic/hyperedges-mimic-text-{train,valid,test}-example.jsonl`.
They can be selected as either `MIMIC-III` or `MIMIC-IV`; the loader maps both
paper aliases to the bundled MIMIC files and stores separate processed outputs.

To run preprocessing explicitly instead of relying on the automatic first-run
step:

```bash
python preprocess.py ICEWS14 ICEWS05-15 ICEWS18 WIKI YAGO GDELT
```

## Running

Run commands from this directory. Install the CPU-compatible core dependencies
from `requirements.txt`. The first run automatically preprocesses the selected
dataset and stores the generated vocabulary, inverse-relation map, cumulative
training-history graphs, and split files under `data/processed/<dataset>/`.
The scripts require Python 3.9 or newer.

Create and activate a local virtual environment before running:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
$env:DGLBACKEND = "pytorch"
```
For a local smoke test without downloading Llama weights, use the deterministic
mock LLM fallback:

```bash
python main.py --dataset ICEWS18 --mode test --use_mock_llm --candidate_mode all --max_eval_samples 20
python main.py --dataset MIMIC-III --mode test --use_mock_llm --candidate_mode htir --max_eval_samples 20
python main.py --dataset ICEWS18 --mode train --use_mock_llm --epochs 10
python abla.py --dataset ICEWS18 --use_mock_llm
python generalization.py --source_dataset ICEWS18 --use_mock_llm
python sensitivity.py --dataset ICEWS18 --params beam_depth beam_width --use_mock_llm
python runtime.py --dataset ICEWS05-15 --use_mock_llm
```

The runtime script reports the implemented `TKL-XR`, linear-fusion, and
no-HTIR variants. It does not fabricate measurements for external baselines
whose source code is not included in this repository.

For the paper configuration, omit `--use_mock_llm`, install the optional LLM
dependencies listed in `requirements-llm.txt`, and provide access to the
configured Llama-2-13B checkpoint. The exact defaults used by the paper
(`D=4`, `K=4`, three GNN layers, decay rate `0.08`, and ten training epochs)
are exposed as command-line arguments in `main.py`.

## Evaluation and reproducibility

The default ranking metrics are MRR, Hit@1, Hit@3, and Hit@10. Explanation
metrics are BLEU-4, ROUGE-L, and BERTScore-F1 when the optional `bert-score`
package is installed. Beam depth, beam width, retrieval rounds, pruning limits,
decay rate, and fusion weights are explicit command-line arguments in
`main.py`.

Seeds are set explicitly. Checkpoints are stored per dataset at
`checkpoints/<dataset>/best_tkl_xr.pth`, and results are written to the
`results/` directory.

The `--candidate_mode all` setting evaluates every entity for strict filtered
ranking. GNN scores are computed for the full entity set, while LLM scores are
queried only for candidates retained by HTIR and are set to zero for the
remaining entities. Use `--candidate_mode htir` only for a faster, approximate
candidate-only evaluation.
