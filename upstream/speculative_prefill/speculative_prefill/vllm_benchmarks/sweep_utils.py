import csv
import html
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Sequence


@dataclass(frozen=True)
class BenchmarkRow:
    mode: str
    model: str
    spec_model: str
    config: str
    input_len: int
    output_len: int
    batch_size: int
    warmup_iters: int
    iters: int
    avg_latency: float
    p50_latency: float
    p90_latency: float


def parse_int_csv(value: str) -> list[int]:
    parsed = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not parsed:
        raise ValueError("expected at least one integer")
    if any(item <= 0 for item in parsed):
        raise ValueError("all values must be positive")
    return parsed


def rows_from_json_payload(payload: dict) -> list[BenchmarkRow]:
    rows = []
    for result in payload.get("results", []):
        percentiles = result["percentiles"]
        rows.append(
            BenchmarkRow(
                mode=payload["mode"],
                model=payload["model"],
                spec_model=payload.get("spec_model", ""),
                config=payload.get("config", ""),
                input_len=int(result["input_len"]),
                output_len=int(result["output_len"]),
                batch_size=int(result["batch_size"]),
                warmup_iters=int(payload["warmup_iters"]),
                iters=int(payload["iters"]),
                avg_latency=float(result["avg_latency"]),
                p50_latency=float(percentiles["50"]),
                p90_latency=float(percentiles["90"]),
            )
        )
    return rows


def rows_to_csv(rows: Sequence[BenchmarkRow], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(asdict(rows[0]).keys()) if rows else [
        field.name for field in BenchmarkRow.__dataclass_fields__.values()
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


def rows_to_jsonl(rows: Sequence[BenchmarkRow], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(asdict(row), sort_keys=True) + "\n")


def _points_for(
    rows: Iterable[BenchmarkRow],
    *,
    batch_size: int,
) -> dict[str, list[tuple[int, float]]]:
    series: dict[str, list[tuple[int, float]]] = {}
    for row in rows:
        if row.batch_size != batch_size:
            continue
        series.setdefault(row.mode, []).append((row.input_len, row.avg_latency))
    for points in series.values():
        points.sort()
    return series


def svg_latency_by_input_len(rows: Sequence[BenchmarkRow], batch_size: int) -> str:
    series = _points_for(rows, batch_size=batch_size)
    width, height = 920, 520
    left, right, top, bottom = 78, 28, 36, 72
    plot_w = width - left - right
    plot_h = height - top - bottom
    colors = {
        "baseline": "#111827",
        "embedding_norm": "#0f766e",
        "spec_prefill": "#b45309",
    }
    all_points = [point for points in series.values() for point in points]
    if not all_points:
        raise ValueError(f"no rows for batch_size={batch_size}")
    xs = [point[0] for point in all_points]
    ys = [point[1] for point in all_points]
    min_x, max_x = min(xs), max(xs)
    max_y = max(ys) * 1.08
    min_y = 0.0

    def sx(x: int) -> float:
        if max_x == min_x:
            return left + plot_w / 2
        return left + (x - min_x) / (max_x - min_x) * plot_w

    def sy(y: float) -> float:
        return top + (max_y - y) / (max_y - min_y) * plot_h

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{left}" y="24" font-family="Arial" font-size="18" font-weight="700">Latency by input length, batch {batch_size}</text>',
        f'<line x1="{left}" y1="{top + plot_h}" x2="{left + plot_w}" y2="{top + plot_h}" stroke="#374151"/>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_h}" stroke="#374151"/>',
        f'<text x="{width / 2 - 70}" y="{height - 18}" font-family="Arial" font-size="13">input length</text>',
        f'<text x="16" y="{height / 2 + 60}" font-family="Arial" font-size="13" transform="rotate(-90 16,{height / 2 + 60})">avg latency seconds</text>',
    ]

    for i in range(5):
        y = min_y + (max_y - min_y) * i / 4
        py = sy(y)
        parts.append(f'<line x1="{left}" y1="{py:.1f}" x2="{left + plot_w}" y2="{py:.1f}" stroke="#e5e7eb"/>')
        parts.append(f'<text x="{left - 10}" y="{py + 4:.1f}" text-anchor="end" font-family="Arial" font-size="11">{y:.3f}</text>')

    for x in sorted(set(xs)):
        px = sx(x)
        parts.append(f'<text x="{px:.1f}" y="{top + plot_h + 20}" text-anchor="middle" font-family="Arial" font-size="11">{x}</text>')

    legend_x = left + plot_w - 210
    legend_y = top + 12
    for idx, (mode, points) in enumerate(sorted(series.items())):
        color = colors.get(mode, "#4b5563")
        path = " ".join(
            f"{'M' if i == 0 else 'L'} {sx(x):.1f} {sy(y):.1f}"
            for i, (x, y) in enumerate(points)
        )
        parts.append(f'<path d="{path}" fill="none" stroke="{color}" stroke-width="2.5"/>')
        for x, y in points:
            parts.append(f'<circle cx="{sx(x):.1f}" cy="{sy(y):.1f}" r="4" fill="{color}"/>')
        ly = legend_y + idx * 22
        parts.append(f'<line x1="{legend_x}" y1="{ly}" x2="{legend_x + 24}" y2="{ly}" stroke="{color}" stroke-width="3"/>')
        parts.append(f'<text x="{legend_x + 32}" y="{ly + 4}" font-family="Arial" font-size="12">{html.escape(mode)}</text>')

    parts.append("</svg>")
    return "\n".join(parts)


def write_svg_latency_charts(rows: Sequence[BenchmarkRow], output_dir: Path) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for batch_size in sorted({row.batch_size for row in rows}):
        path = output_dir / f"latency_by_input_len_b{batch_size}.svg"
        path.write_text(svg_latency_by_input_len(rows, batch_size), encoding="utf-8")
        paths.append(path)
    return paths
