import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from speculative_prefill.api_eval.aggregate import collect_shards, summarize_shards


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="local/deepseek_accuracy")
    parser.add_argument("--requested-datasets", default="all")
    parser.add_argument("--output-dir", default="local/deepseek_accuracy/aggregates")
    parser.add_argument("--block-size", type=int, default=25)
    return parser.parse_args()


def main():
    args = parse_args()
    root = Path(args.root)
    shards = collect_shards(root, requested_datasets=args.requested_datasets)
    summary = summarize_shards(shards, block_size=args.block_size)

    timestamp = time.strftime("%Y%m%dT%H%M%S")
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"deepseek_accuracy_aggregate_{timestamp}.json"
    out_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    latest_path = out_dir / "latest.json"
    latest_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Wrote {out_path}")
    print(f"Shards: {summary['shard_count']}/{summary['expected_shards']}")
    print(f"Rows: {summary['row_count']}/{summary['expected_rows']}")
    print(json.dumps(summary["scores_by_method_model"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
