from __future__ import annotations

import argparse
import json
import math
import time
from contextlib import ExitStack
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from tqdm.auto import tqdm

from eval_component_ablation_da2k import (
    ComponentSpec,
    ablate_component,
    build_component_specs,
    evaluate_da2k_model,
    parse_component_types,
    parse_components,
    score_overall,
    selected_annotations,
)
from eval_da2k import MODEL_CONFIGS, load_model, resolve_device


@dataclass(frozen=True)
class BeamAblationConfig:
    dataset_root: Path
    checkpoint: Path
    output_dir: Path
    encoder: str = "vits"
    input_size: int = 518
    device: str = "auto"
    component_types: tuple[str, ...] = ("block", "attn", "mlp")
    components: tuple[str, ...] = ()
    beam_width: int = 4
    max_depth: int = 4
    scene_type: str = ""
    max_images: int = 100
    score_direction: str = "larger"
    log_every: int = 0

    def __post_init__(self) -> None:
        object.__setattr__(self, "dataset_root", Path(self.dataset_root))
        object.__setattr__(self, "checkpoint", Path(self.checkpoint))
        object.__setattr__(self, "output_dir", Path(self.output_dir))
        if self.encoder not in MODEL_CONFIGS:
            raise ValueError(f"unknown encoder: {self.encoder}")
        if self.beam_width <= 0:
            raise ValueError("beam_width must be positive")
        if self.max_depth < 0:
            raise ValueError("max_depth must be non-negative")
        if self.max_images < 0:
            raise ValueError("max_images must be non-negative")
        if self.score_direction not in {"larger", "smaller", "best"}:
            raise ValueError("score_direction must be larger, smaller, or best")
        if self.log_every < 0:
            raise ValueError("log_every must be non-negative")


def state_name(state: tuple[int, ...], specs: list[ComponentSpec]) -> str:
    if not state:
        return "baseline"
    return "__".join(specs[index].name for index in state)


def parse_eval_states(value: str) -> tuple[tuple[str, ...], ...]:
    if not value.strip():
        return ()
    states: list[tuple[str, ...]] = []
    for state_text in value.split(";"):
        names = tuple(part.strip() for part in state_text.split(",") if part.strip())
        if names:
            states.append(names)
    return tuple(states)


def filter_specs(specs: list[ComponentSpec], wanted: tuple[str, ...]) -> list[ComponentSpec]:
    if not wanted:
        return specs
    wanted_set = set(wanted)
    filtered = [spec for spec in specs if spec.name in wanted_set or spec.module_name in wanted_set]
    found = {spec.name for spec in filtered} | {spec.module_name for spec in filtered}
    missing = sorted(wanted_set - found)
    if missing:
        raise ValueError(f"requested component(s) not found: {missing}")
    return filtered


def resolve_state_indices(specs: list[ComponentSpec], state_names: tuple[str, ...]) -> tuple[int, ...]:
    by_name = {spec.name: index for index, spec in enumerate(specs)}
    by_module = {spec.module_name: index for index, spec in enumerate(specs)}
    indices: list[int] = []
    missing: list[str] = []
    for name in state_names:
        if name in by_name:
            indices.append(by_name[name])
        elif name in by_module:
            indices.append(by_module[name])
        else:
            missing.append(name)
    if missing:
        raise ValueError(f"requested component(s) not found: {missing}")
    state = tuple(sorted(set(indices)))
    if len(state) != len(indices):
        raise ValueError(f"state contains duplicate component(s): {state_names}")
    if not is_valid_state(state, specs):
        raise ValueError(f"state contains redundant whole-block and child ablations: {state_names}")
    return state


def is_valid_state(state: tuple[int, ...], specs: list[ComponentSpec]) -> bool:
    whole_block_layers = {specs[index].layer_index for index in state if specs[index].kind == "block"}
    for index in state:
        spec = specs[index]
        if spec.kind != "block" and spec.layer_index in whole_block_layers:
            return False
    return True


def ablate_components(model: torch.nn.Module, specs: list[ComponentSpec], state: tuple[int, ...]):
    stack = ExitStack()
    for index in state:
        stack.enter_context(ablate_component(model, specs[index]))
    return stack


def read_state_record(output_dir: Path, state: tuple[int, ...], specs: list[ComponentSpec]) -> dict[str, Any] | None:
    path = output_dir / "states" / state_name(state, specs) / "summary.json"
    if not path.exists():
        return None
    try:
        record = json.loads(path.read_text())
    except json.JSONDecodeError:
        return None
    if tuple(int(index) for index in record.get("state_indices", [])) != state:
        return None
    return record


def write_state_record(output_dir: Path, record: dict[str, Any]) -> None:
    state_dir = output_dir / "states" / str(record["state_name"])
    state_dir.mkdir(parents=True, exist_ok=True)
    (state_dir / "summary.json").write_text(json.dumps(record, indent=2, sort_keys=True, default=str) + "\n")


def evaluate_state(
    *,
    model: torch.nn.Module,
    specs: list[ComponentSpec],
    state: tuple[int, ...],
    baseline_score: float,
    config: BeamAblationConfig,
    items: list[tuple[str, list[dict[str, Any]]]],
    device: torch.device,
) -> dict[str, Any]:
    existing = read_state_record(config.output_dir, state, specs)
    if existing is not None:
        return existing

    started = time.monotonic()
    if state:
        with ablate_components(model, specs, state):
            result = evaluate_da2k_model(
                model=model,
                dataset_root=config.dataset_root,
                items=items,
                input_size=config.input_size,
                device=device,
                log_every=0,
            )
    else:
        result = evaluate_da2k_model(
            model=model,
            dataset_root=config.dataset_root,
            items=items,
            input_size=config.input_size,
            device=device,
            log_every=config.log_every,
        )
    score = score_overall(result["overall"], config.score_direction)
    score_delta = 0.0 if not state and baseline_score == 0.0 else score - baseline_score
    record = {
        "state_name": state_name(state, specs),
        "state_indices": list(state),
        "components": [specs[index].name for index in state],
        "component_kinds": [specs[index].kind for index in state],
        "layer_indices": [specs[index].layer_index for index in state],
        "component_count": len(state),
        "score_direction": config.score_direction,
        "score": score,
        "baseline_score": baseline_score,
        "score_delta": score_delta,
        "elapsed_seconds": time.monotonic() - started,
        **result,
    }
    write_state_record(config.output_dir, record)
    print(
        json.dumps(
            {
                "state": record["components"],
                "score": record["score"],
                "score_delta": record["score_delta"],
                "overall": record["overall"],
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return record


def rank_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(
        records,
        key=lambda row: (
            -float(row["score"]),
            int(row["component_count"]),
            list(row["components"]),
        ),
    )


def append_jsonl(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True, default=str) + "\n")
        handle.flush()


def write_markdown(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        f"Ranking metric: `{summary['config']['score_direction']}` DA-2K accuracy.",
        "",
        "| depth | rank | components | score | delta | larger acc | smaller acc | pairs |",
        "| ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    baseline = summary["baseline"]
    rows = [{"depth": 0, "rank": 1, **baseline}]
    for depth_row in summary["beam"]:
        for rank, record in enumerate(depth_row["states"], start=1):
            rows.append({"depth": depth_row["depth"], "rank": rank, **record})
    for row in rows:
        overall = row["overall"]
        components = ", ".join(row["components"]) if row["components"] else "baseline"
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["depth"]),
                    str(row["rank"]),
                    components,
                    f"{float(row['score']):.6f}",
                    f"{float(row['score_delta']):.6f}",
                    f"{float(overall['larger_is_closer_accuracy']):.6f}",
                    f"{float(overall['smaller_is_closer_accuracy']):.6f}",
                    str(overall["pairs"]),
                ]
            )
            + " |"
        )
    path.write_text("\n".join(lines) + "\n")


def run_beam_search(config: BeamAblationConfig) -> dict[str, Any]:
    torch.manual_seed(17)
    torch.set_float32_matmul_precision("high")
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    device = resolve_device(config.device)
    config.output_dir.mkdir(parents=True, exist_ok=True)
    (config.output_dir / "config.json").write_text(
        json.dumps(asdict(config), indent=2, sort_keys=True, default=str) + "\n"
    )

    items = selected_annotations(
        config.dataset_root,
        scene_type=config.scene_type,
        max_images=config.max_images,
    )
    if not items:
        raise RuntimeError("no DA-2K annotations selected")

    model = load_model(config.encoder, config.checkpoint, device)
    for param in model.parameters():
        param.requires_grad_(False)

    specs = filter_specs(build_component_specs(model, component_types=config.component_types), config.components)
    if not specs:
        raise RuntimeError("no components selected for beam search")
    (config.output_dir / "components.json").write_text(
        json.dumps([asdict(spec) for spec in specs], indent=2, sort_keys=True) + "\n"
    )

    baseline = evaluate_state(
        model=model,
        specs=specs,
        state=(),
        baseline_score=0.0,
        config=config,
        items=items,
        device=device,
    )
    baseline["baseline_score"] = baseline["score"]
    baseline["score_delta"] = 0.0
    write_state_record(config.output_dir, baseline)
    append_jsonl(config.output_dir / "states.jsonl", baseline)
    baseline_score = float(baseline["score"])

    beam_states: list[tuple[int, ...]] = [()]
    beam_records: list[dict[str, Any]] = []
    all_records: dict[tuple[int, ...], dict[str, Any]] = {(): baseline}
    for depth in range(1, min(config.max_depth, len(specs)) + 1):
        candidates: list[tuple[int, ...]] = []
        seen: set[tuple[int, ...]] = set()
        for state in beam_states:
            for index in range(len(specs)):
                if index in state:
                    continue
                candidate = tuple(sorted((*state, index)))
                if candidate in seen or not is_valid_state(candidate, specs):
                    continue
                seen.add(candidate)
                candidates.append(candidate)

        depth_records: list[dict[str, Any]] = []
        for state in tqdm(candidates, desc=f"beam depth {depth}", unit="state"):
            record = evaluate_state(
                model=model,
                specs=specs,
                state=state,
                baseline_score=baseline_score,
                config=config,
                items=items,
                device=device,
            )
            depth_records.append(record)
            all_records[state] = record
            append_jsonl(config.output_dir / "states.jsonl", record)

        ranked = rank_records(depth_records)
        depth_beam = ranked[: config.beam_width]
        beam_states = [tuple(int(index) for index in row["state_indices"]) for row in depth_beam]
        depth_summary = {
            "depth": depth,
            "candidate_count": len(depth_records),
            "states": depth_beam,
        }
        beam_records.append(depth_summary)
        append_jsonl(config.output_dir / "beam.jsonl", depth_summary)
        if not beam_states:
            break

    records = list(all_records.values())
    best_record = rank_records(records)[0]
    ablated_records = [record for record in records if int(record["component_count"]) > 0]
    best_ablated_record = rank_records(ablated_records)[0] if ablated_records else None
    summary = {
        "config": asdict(config),
        "device": str(device),
        "component_count": len(specs),
        "baseline": baseline,
        "beam": beam_records,
        "best_record": best_record,
        "best_ablated_record": best_ablated_record,
        "records": rank_records(records),
        "rule": (
            "Beam search over compounded component ablations. A state disables all listed components "
            "simultaneously via temporary hooks; redundant states that combine a whole block with one of "
            "its child branches are skipped."
        ),
    }
    (config.output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True, default=str) + "\n")
    write_markdown(config.output_dir / "summary.md", summary)
    return summary


def run_fixed_states(config: BeamAblationConfig, eval_states: tuple[tuple[str, ...], ...]) -> dict[str, Any]:
    torch.manual_seed(17)
    torch.set_float32_matmul_precision("high")
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    device = resolve_device(config.device)
    config.output_dir.mkdir(parents=True, exist_ok=True)
    (config.output_dir / "config.json").write_text(
        json.dumps({**asdict(config), "eval_states": eval_states}, indent=2, sort_keys=True, default=str) + "\n"
    )

    items = selected_annotations(
        config.dataset_root,
        scene_type=config.scene_type,
        max_images=config.max_images,
    )
    if not items:
        raise RuntimeError("no DA-2K annotations selected")

    model = load_model(config.encoder, config.checkpoint, device)
    for param in model.parameters():
        param.requires_grad_(False)

    specs = filter_specs(build_component_specs(model, component_types=config.component_types), config.components)
    if not specs:
        raise RuntimeError("no components selected for fixed-state evaluation")
    (config.output_dir / "components.json").write_text(
        json.dumps([asdict(spec) for spec in specs], indent=2, sort_keys=True) + "\n"
    )

    states = [resolve_state_indices(specs, state_names) for state_names in eval_states]
    baseline = evaluate_state(
        model=model,
        specs=specs,
        state=(),
        baseline_score=0.0,
        config=config,
        items=items,
        device=device,
    )
    baseline["baseline_score"] = baseline["score"]
    baseline["score_delta"] = 0.0
    write_state_record(config.output_dir, baseline)
    append_jsonl(config.output_dir / "states.jsonl", baseline)
    baseline_score = float(baseline["score"])

    records: list[dict[str, Any]] = []
    for state in tqdm(states, desc="fixed states", unit="state"):
        record = evaluate_state(
            model=model,
            specs=specs,
            state=state,
            baseline_score=baseline_score,
            config=config,
            items=items,
            device=device,
        )
        records.append(record)
        append_jsonl(config.output_dir / "states.jsonl", record)

    summary = {
        "config": {**asdict(config), "eval_states": eval_states},
        "device": str(device),
        "component_count": len(specs),
        "baseline": baseline,
        "records": rank_records(records),
        "rule": (
            "Fixed-state compounded component ablation evaluation. Each state disables all listed components "
            "simultaneously via temporary hooks; hooks are removed before the next state."
        ),
    }
    (config.output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True, default=str) + "\n")
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Beam search compounded component ablations on DA-2K.")
    parser.add_argument("--dataset-root", type=Path, default=Path("datasets/DA-2K/extracted/DA-2K"))
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/depth_anything_v2_vits.pth"))
    parser.add_argument("--output-dir", type=Path, default=Path("eval_outputs/da2k_vits_component_ablation_beam"))
    parser.add_argument("--encoder", choices=sorted(MODEL_CONFIGS), default="vits")
    parser.add_argument("--input-size", type=int, default=518)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--component-types", default="block,attn,mlp")
    parser.add_argument("--components", default="", help="Comma-separated component names or module names to search.")
    parser.add_argument(
        "--eval-states",
        default="",
        help="Semicolon-separated compounded states to evaluate directly, with comma-separated components per state.",
    )
    parser.add_argument("--beam-width", type=int, default=4)
    parser.add_argument("--max-depth", type=int, default=4)
    parser.add_argument(
        "--scene-type",
        default="",
        choices=["", "indoor", "outdoor", "non_real", "transparent_reflective", "adverse_style", "aerial", "underwater", "object"],
    )
    parser.add_argument("--max-images", type=int, default=100)
    parser.add_argument("--score-direction", choices=["larger", "smaller", "best"], default="larger")
    parser.add_argument("--log-every", type=int, default=0)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    config = BeamAblationConfig(
        dataset_root=args.dataset_root,
        checkpoint=args.checkpoint,
        output_dir=args.output_dir,
        encoder=args.encoder,
        input_size=args.input_size,
        device=args.device,
        component_types=parse_component_types(args.component_types),
        components=parse_components(args.components),
        beam_width=args.beam_width,
        max_depth=args.max_depth,
        scene_type=args.scene_type,
        max_images=args.max_images,
        score_direction=args.score_direction,
        log_every=args.log_every,
    )
    eval_states = parse_eval_states(args.eval_states)
    if eval_states:
        summary = run_fixed_states(config, eval_states)
        print(
            json.dumps(
                {
                    "baseline": summary["baseline"]["overall"],
                    "records": [
                        {
                            "components": row["components"],
                            "score": row["score"],
                            "score_delta": row["score_delta"],
                            "overall": row["overall"],
                        }
                        for row in summary["records"]
                    ],
                    "output_dir": str(config.output_dir),
                },
                indent=2,
                sort_keys=True,
                default=str,
            )
        )
        return
    summary = run_beam_search(config)
    print(
        json.dumps(
            {
                "baseline": summary["baseline"]["overall"],
                "best_record": summary["best_record"],
                "best_ablated_record": summary["best_ablated_record"],
                "output_dir": str(config.output_dir),
            },
            indent=2,
            sort_keys=True,
            default=str,
        )
    )


if __name__ == "__main__":
    main()
