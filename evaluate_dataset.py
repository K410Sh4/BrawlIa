"""Compare configurações nas mesmas imagens brutas revisadas, sem treinar modelo."""

import argparse
import json
from brawlla.config import load_config
from brawlla.evaluation import evaluate_dataset
from brawlla.provenance import provenance_snapshot
from brawlla.storage import write_json

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset")
    parser.add_argument("--config")
    parser.add_argument("--compare-config")
    parser.add_argument("--output", default="evaluation.json")
    parser.add_argument("--iou", type=float, default=0.5)
    args = parser.parse_args()
    result = {}
    for label, path in [("candidate", args.config), ("baseline", args.compare_config)]:
        if label == "baseline" and path is None:
            continue
        config = load_config(path)
        result[label] = {
            "provenance": provenance_snapshot(config),
            **evaluate_dataset(args.dataset, config, args.iou),
        }
    write_json(args.output, result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
