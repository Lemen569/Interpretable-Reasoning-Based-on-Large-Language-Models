"""Preprocess one or more temporal knowledge graph benchmarks.

The training/evaluation entry point performs this step automatically, but this
small CLI makes the reproducibility pipeline explicit and independently
inspectable.
"""

from __future__ import annotations

import argparse

from modules.data_process import TKGDataProcessor


def parse_args():
    parser = argparse.ArgumentParser(description="Preprocess TKG benchmark files")
    parser.add_argument("datasets", nargs="+", help="Names of folders under --raw_data_path")
    parser.add_argument("--raw_data_path", default="./data")
    parser.add_argument("--processed_path", default="./data/processed")
    return parser.parse_args()


def main():
    args = parse_args()
    processor = TKGDataProcessor(args.raw_data_path, args.processed_path)
    for dataset_name in args.datasets:
        dataset = processor.process(dataset_name)
        print(
            f"{dataset_name}: entities={dataset.entity_num}, "
            f"relations={dataset.relation_num}, times={dataset.time_num}, "
            f"train={len(dataset.train)}, valid={len(dataset.valid)}, "
            f"test={len(dataset.test)}"
        )


if __name__ == "__main__":
    main()
