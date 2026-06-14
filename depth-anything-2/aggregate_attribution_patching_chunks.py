from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def empty_counts() -> dict[str, int]:
    return {"pairs": 0, "smaller_correct": 0, "larger_correct": 0, "ties": 0}


def add_counts(dst: dict[str, int], src: dict[str, Any]) -> None:
    for key in dst:
        dst[key] += int(src[key])


def finalize_counts(counts: dict[str, int]) -> dict[str, float | int | str]:
    pairs = max(counts["pairs"], 1)
    smaller = counts["smaller_correct"] / pairs
    larger = counts["larger_correct"] / pairs
    return {
        **counts,
        "smaller_is_closer_accuracy": smaller,
        "larger_is_closer_accuracy": larger,
        "best_direction": "smaller" if smaller >= larger else "larger",
        "best_accuracy": max(smaller, larger),
        "tie_fraction": counts["ties"] / pairs,
    }


def aggregate_node_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    accum: dict[str, dict[str, Any]] = {}
    for row in rows:
        key = row["component"]
        item = accum.setdefault(
            key,
            {
                "component": row["component"],
                "kind": row["kind"],
                "module_name": row["module_name"],
                "layer_index": row["layer_index"],
                "images": 0,
                "attribution": 0.0,
                "abs_attribution": 0.0,
                "positive_attribution": 0.0,
                "negative_attribution": 0.0,
                "delta_l2": 0.0,
                "grad_l2": 0.0,
                "numel": 0,
            },
        )
        item["images"] += 1
        for field in ("attribution", "abs_attribution", "positive_attribution", "negative_attribution", "delta_l2", "grad_l2"):
            item[field] += float(row[field])
        item["numel"] += int(row["numel"])
    output = []
    for item in accum.values():
        images = max(int(item["images"]), 1)
        item["mean_attribution_per_image"] = item["attribution"] / images
        item["mean_abs_attribution_per_image"] = item["abs_attribution"] / images
        item["mean_positive_attribution_per_image"] = item["positive_attribution"] / images
        item["mean_negative_attribution_per_image"] = item["negative_attribution"] / images
        item["mean_attribution_per_value"] = item["attribution"] / max(int(item["numel"]), 1)
        output.append(item)
    return output


def layer_key(row: dict[str, Any]) -> str:
    if row["layer_index"] is None:
        return "head"
    return f"block_{int(row['layer_index']):02d}"


def build_layer_rollup(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rollup: dict[str, dict[str, Any]] = {}
    for row in rows:
        key = layer_key(row)
        item = rollup.setdefault(
            key,
            {
                "layer": key,
                "layer_index": row["layer_index"],
                "nodes": 0,
                "attribution": 0.0,
                "abs_attribution": 0.0,
                "positive_attribution": 0.0,
                "negative_attribution": 0.0,
            },
        )
        item["nodes"] += 1
        for field in ("attribution", "abs_attribution", "positive_attribution", "negative_attribution"):
            item[field] += float(row[field])
    return sorted(rollup.values(), key=lambda row: (-float(row["positive_attribution"]), str(row["layer"])))


def build_covers(rows_by_positive: list[dict[str, Any]]) -> dict[str, Any]:
    positive_rows = [row for row in rows_by_positive if float(row["positive_attribution"]) > 0.0]
    total = sum(float(row["positive_attribution"]) for row in positive_rows)
    covers = {}
    for fraction in (0.5, 0.8, 0.9, 0.95):
        threshold = total * fraction
        running = 0.0
        components: list[str] = []
        for row in positive_rows:
            running += float(row["positive_attribution"])
            components.append(str(row["component"]))
            if running >= threshold:
                break
        covers[f"{int(round(fraction * 100))}pct_positive_mass"] = {
            "node_count": len(components),
            "positive_mass": running,
            "components": components,
        }
    return {"total_positive_mass": total, "covers": covers}


def write_markdown(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# Aggregated Attribution Patching Circuit Search",
        "",
        f"Images: `{summary['image_count']}`",
        f"Clean larger-is-closer: `{summary['clean_overall']['larger_is_closer_accuracy']:.6f}`",
        f"Mean-corrupt larger-is-closer: `{summary['corrupted_overall']['larger_is_closer_accuracy']:.6f}`",
        "",
        "## Top Positive Circuit Nodes",
        "",
        "| rank | component | kind | layer | positive | attribution | abs |",
        "| ---: | --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for rank, row in enumerate(summary["rows_by_positive"][:30], start=1):
        layer = "" if row["layer_index"] is None else str(row["layer_index"])
        lines.append(
            f"| {rank} | {row['component']} | {row['kind']} | {layer} | "
            f"{float(row['positive_attribution']):.6g} | {float(row['attribution']):.6g} | "
            f"{float(row['abs_attribution']):.6g} |"
        )
    lines.extend(
        [
            "",
            "## Layer Rollup",
            "",
            "| rank | layer | nodes | positive | attribution | abs |",
            "| ---: | --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for rank, row in enumerate(summary["layer_rollup"], start=1):
        lines.append(
            f"| {rank} | {row['layer']} | {row['nodes']} | "
            f"{float(row['positive_attribution']):.6g} | {float(row['attribution']):.6g} | "
            f"{float(row['abs_attribution']):.6g} |"
        )
    path.write_text("\n".join(lines) + "\n")


def run(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    clean = empty_counts()
    corrupted = empty_counts()
    image_count = 0
    node_rows: list[dict[str, Any]] = []
    config: dict[str, Any] | None = None
    source_dirs: list[str] = []

    for chunk_dir in args.chunk_dirs:
        summary = json.loads((chunk_dir / "summary.json").read_text())
        config = config or summary.get("config", {})
        source_dirs.append(str(chunk_dir))
        add_counts(clean, summary["clean_overall"])
        add_counts(corrupted, summary["corrupted_overall"])
        image_count += int(summary["image_count"])
        for line in (chunk_dir / "node_rows.jsonl").read_text().splitlines():
            if line.strip():
                node_rows.append(json.loads(line))

    aggregated = aggregate_node_rows(node_rows)
    rows_by_positive = sorted(
        aggregated,
        key=lambda row: (
            -float(row["positive_attribution"]),
            -float(row["attribution"]),
            999 if row["layer_index"] is None else int(row["layer_index"]),
            str(row["component"]),
        ),
    )
    rows_by_signed = sorted(
        aggregated,
        key=lambda row: (
            -float(row["attribution"]),
            999 if row["layer_index"] is None else int(row["layer_index"]),
            str(row["component"]),
        ),
    )
    rows_by_abs = sorted(
        aggregated,
        key=lambda row: (
            -float(row["abs_attribution"]),
            999 if row["layer_index"] is None else int(row["layer_index"]),
            str(row["component"]),
        ),
    )
    summary = {
        "config": config or {},
        "aggregate_source_dirs": source_dirs,
        "node_count": len(aggregated),
        "image_count": image_count,
        "clean_overall": finalize_counts(clean),
        "corrupted_overall": finalize_counts(corrupted),
        "rows_by_positive": rows_by_positive,
        "rows_by_signed": rows_by_signed,
        "rows_by_abs": rows_by_abs,
        "layer_rollup": build_layer_rollup(aggregated),
        "circuit_candidates": build_covers(rows_by_positive),
        "metadata": {
            "method": "aggregate of finalized attribution-patching chunks",
            "note": args.note,
        },
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    write_markdown(output_dir / "summary.md", summary)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Aggregate attribution-patching chunk outputs.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--note", default="")
    parser.add_argument("chunk_dirs", type=Path, nargs="+")
    return parser


def main() -> None:
    summary = run(build_parser().parse_args())
    print(
        json.dumps(
            {
                "output_dir": str(Path(summary["aggregate_source_dirs"][0]).parent / "summary.json"),
                "image_count": summary["image_count"],
                "clean_overall": summary["clean_overall"],
                "corrupted_overall": summary["corrupted_overall"],
                "top_circuit_nodes": [
                    {
                        "component": row["component"],
                        "kind": row["kind"],
                        "layer_index": row["layer_index"],
                        "positive_attribution": row["positive_attribution"],
                        "attribution": row["attribution"],
                    }
                    for row in summary["rows_by_positive"][:12]
                ],
                "cover50": summary["circuit_candidates"]["covers"]["50pct_positive_mass"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
