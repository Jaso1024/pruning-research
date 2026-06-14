from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any


def row_pair_counts(row: dict[str, Any]) -> tuple[int, int]:
    pair_delta = row.get("pair_delta", {})
    return int(pair_delta.get("lost_pair_count", 0)), int(pair_delta.get("gained_pair_count", 0))


def write_ranked_csv(rows: list[dict[str, Any]], output_path: Path) -> None:
    with output_path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "rank",
                "name",
                "kind",
                "module",
                "correct_drop",
                "accuracy_drop",
                "mean_margin_drop",
                "lost_pairs",
                "gained_pairs",
                "param_estimate",
                "drop_per_1k_params",
            ]
        )
        for rank, row in enumerate(rows, start=1):
            params = max(int(row.get("parameter_estimate", 0)), 1)
            lost, gained = row_pair_counts(row)
            writer.writerow(
                [
                    rank,
                    row["name"],
                    row["kind"],
                    row["module_name"],
                    row["correct_drop"],
                    row["accuracy_drop"],
                    row["mean_margin_drop"],
                    lost,
                    gained,
                    row.get("parameter_estimate", 0),
                    float(row["correct_drop"]) * 1000.0 / params,
                ]
            )


def attention_component_table(rows: list[dict[str, Any]]) -> list[str]:
    components = {
        "attn_q_head": "q",
        "attn_k_head": "k",
        "attn_v_head": "v",
        "attn_route_head": "route",
    }
    heads: dict[tuple[int, int], dict[str, int]] = defaultdict(dict)
    for row in rows:
        kind = row["kind"]
        if kind not in components:
            continue
        parts = row["name"].split("_")
        block = int(parts[1])
        head = int(parts[-1])
        heads[(block, head)][components[kind]] = int(row["correct_drop"])

    strongest = sorted(heads.items(), key=lambda item: -max(item[1].values()))[:20]
    lines = [
        "## Attention Head Decomposition",
        "",
        "| block | head | q | k | route | v | max |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for (block, head), values in strongest:
        maximum = max(values.values())
        lines.append(
            f"| {block} | {head} | {values.get('q', 0)} | {values.get('k', 0)} | "
            f"{values.get('route', 0)} | {values.get('v', 0)} | {maximum} |"
        )
    return lines


def pair_examples(rows: list[dict[str, Any]]) -> list[str]:
    lines = ["## Pair-Flip Examples"]
    selected = [row for row in rows if row["name"] != "depth_head_scratch_output_conv2_2_out_group_00_0_1"][:8]
    for row in selected:
        pair_delta = row.get("pair_delta", {})
        lines.extend(["", f"### `{row['name']}` ({row['kind']}, drop {row['correct_drop']})"])
        lost = pair_delta.get("top_lost_pairs", [])[:4]
        gained = pair_delta.get("top_gained_pairs", [])[:4]
        if lost:
            lines.append("Lost pairs:")
            for item in lost:
                lines.append(
                    f"- `{item['pair_id']}` margin {item['baseline_margin']:.4f} -> "
                    f"{item['ablated_margin']:.4f} (drop {item['margin_drop']:.4f})"
                )
        if gained:
            lines.append("Gained pairs:")
            for item in gained:
                lines.append(
                    f"- `{item['pair_id']}` margin {item['baseline_margin']:.4f} -> "
                    f"{item['ablated_margin']:.4f} (drop {item['margin_drop']:.4f})"
                )
    return lines


def build_report(summary: dict[str, Any], output_dir: Path) -> str:
    rows = summary["rows_by_accuracy_drop"]
    by_kind: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_kind[row["kind"]].append(row)

    baseline = summary["baseline"]["overall"]
    lines = [
        "# Fine Subcircuit Ablation Report",
        "",
        f"Output dir: `{output_dir}`",
        (
            f"Baseline: `{baseline['larger_correct']}/{baseline['pairs']} = "
            f"{baseline['larger_is_closer_accuracy']:.4f}` on `{summary['image_count']}` images"
        ),
        f"Nodes: `{summary['node_count']}`",
        "",
        "## By-Kind Distribution",
        "",
        "| kind | n | drop>0 | drop=0 | drop<0 | mean_drop | max_drop | min_drop |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for kind in sorted(by_kind):
        drops = [int(row["correct_drop"]) for row in by_kind[kind]]
        lines.append(
            f"| {kind} | {len(drops)} | {sum(drop > 0 for drop in drops)} | "
            f"{sum(drop == 0 for drop in drops)} | {sum(drop < 0 for drop in drops)} | "
            f"{statistics.mean(drops):.3f} | {max(drops)} | {min(drops)} |"
        )

    lines.extend(
        [
            "",
            "## Top Damaging Nodes",
            "",
            "| rank | node | kind | correct_drop | margin_drop | lost/gained pairs |",
            "|---:|---|---|---:|---:|---:|",
        ]
    )
    for rank, row in enumerate(rows[:20], start=1):
        lost, gained = row_pair_counts(row)
        lines.append(
            f"| {rank} | `{row['name']}` | {row['kind']} | {row['correct_drop']} | "
            f"{row['mean_margin_drop']:.4f} | {lost}/{gained} |"
        )

    lines.extend(
        [
            "",
            "## Safest / Helpful Nodes",
            "",
            "| rank | node | kind | correct_drop | margin_drop | lost/gained pairs |",
            "|---:|---|---|---:|---:|---:|",
        ]
    )
    for rank, row in enumerate(summary["rows_by_safe"][:20], start=1):
        lost, gained = row_pair_counts(row)
        lines.append(
            f"| {rank} | `{row['name']}` | {row['kind']} | {row['correct_drop']} | "
            f"{row['mean_margin_drop']:.4f} | {lost}/{gained} |"
        )

    lines.extend(["", *attention_component_table(rows), "", "## Top By Kind"])
    for kind in sorted(by_kind):
        lines.extend(["", f"### {kind}", "| node | correct_drop | margin_drop | lost/gained |", "|---|---:|---:|---:|"])
        top_rows = sorted(
            by_kind[kind],
            key=lambda row: (-int(row["correct_drop"]), -float(row["mean_margin_drop"]), row["name"]),
        )[:10]
        for row in top_rows:
            lost, gained = row_pair_counts(row)
            lines.append(f"| `{row['name']}` | {row['correct_drop']} | {row['mean_margin_drop']:.4f} | {lost}/{gained} |")

    lines.extend(["", *pair_examples(rows)])
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize subcircuit ablation JSON into markdown and CSV.")
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()

    summary_path = args.output_dir / "summary.json"
    summary = json.loads(summary_path.read_text())
    write_ranked_csv(summary["rows_by_accuracy_drop"], args.output_dir / "ranked_nodes.csv")
    report = build_report(summary, args.output_dir)
    report_path = args.output_dir / "analysis_report.md"
    report_path.write_text(report)

    by_kind: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in summary["rows_by_accuracy_drop"]:
        by_kind[row["kind"]].append(row)
    baseline = summary["baseline"]["overall"]
    print(f"report={report_path}")
    print(f"csv={args.output_dir / 'ranked_nodes.csv'}")
    print(f"baseline={baseline['larger_correct']}/{baseline['pairs']} acc={baseline['larger_is_closer_accuracy']:.4f}")
    for kind in sorted(by_kind):
        drops = [int(row["correct_drop"]) for row in by_kind[kind]]
        print(
            kind,
            "n",
            len(drops),
            "pos",
            sum(drop > 0 for drop in drops),
            "zero",
            sum(drop == 0 for drop in drops),
            "neg",
            sum(drop < 0 for drop in drops),
            "max",
            max(drops),
            "min",
            min(drops),
        )
    print("top20")
    for row in summary["rows_by_accuracy_drop"][:20]:
        lost, gained = row_pair_counts(row)
        print(row["name"], row["kind"], row["correct_drop"], f"margin={row['mean_margin_drop']:.4f}", f"lost/gained={lost}/{gained}")


if __name__ == "__main__":
    main()
