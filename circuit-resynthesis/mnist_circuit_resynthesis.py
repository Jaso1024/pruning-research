from __future__ import annotations

import argparse
import json
import math
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

try:
    from sklearn.cluster import KMeans
except ModuleNotFoundError:
    KMeans = None


@dataclass
class Config:
    output_dir: Path = Path("outputs/mnist_resynthesis")
    data_dir: Path = Path("data")
    seed: int = 123
    batch_size: int = 1024
    source_hidden: int = 512
    source_epochs: int = 6
    student_epochs: int = 4
    distill_epochs: int = 4
    lr: float = 2e-3
    student_lr: float = 2e-3
    train_subset: int = 0
    test_subset: int = 0
    conv_filters: int = 32
    patch_size: int = 7
    local_windows: tuple[int, ...] = (3, 5, 7, 9, 11, 15)
    masked_windows: tuple[int, ...] = (7, 11, 15)
    prune_fracs: tuple[float, ...] = (0.5, 0.8, 0.9, 0.95)
    run_conv: bool = True
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


class OneHiddenMLP(nn.Module):
    def __init__(self, hidden: int = 512):
        super().__init__()
        self.fc1 = nn.Linear(28 * 28, hidden)
        self.fc2 = nn.Linear(hidden, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.reshape(x.shape[0], -1)
        hidden = F.relu(self.fc1(x))
        return self.fc2(hidden)

    def hidden(self, x: torch.Tensor) -> torch.Tensor:
        x = x.reshape(x.shape[0], -1)
        return F.relu(self.fc1(x))


class TinyConv(nn.Module):
    def __init__(self, filters: int = 32, patch_size: int = 7):
        super().__init__()
        self.conv = nn.Conv2d(1, filters, patch_size, padding=patch_size // 2)
        self.head = nn.Sequential(
            nn.Linear(filters * 2, 64),
            nn.ReLU(),
            nn.Linear(64, 10),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = F.relu(self.conv(x))
        avg = h.mean(dim=(2, 3))
        mx = h.amax(dim=(2, 3))
        return self.head(torch.cat([avg, mx], dim=1))


class FixedStrokeBank(nn.Module):
    def __init__(self, filters: torch.Tensor):
        super().__init__()
        if filters.ndim != 4:
            raise ValueError("filters must be [K,1,k,k]")
        self.register_buffer("filters", filters.float())
        k = int(filters.shape[0])
        self.head = nn.Sequential(
            nn.Linear(k * 2, 64),
            nn.ReLU(),
            nn.Linear(64, 10),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = F.conv2d(x, self.filters, padding=int(self.filters.shape[-1] // 2))
        h = F.relu(h)
        avg = h.mean(dim=(2, 3))
        mx = h.amax(dim=(2, 3))
        return self.head(torch.cat([avg, mx], dim=1))


class MaskedMLP(OneHiddenMLP):
    def __init__(self, hidden: int, fc1_mask: torch.Tensor, fc2_mask: torch.Tensor | None = None):
        super().__init__(hidden)
        if fc1_mask.shape != self.fc1.weight.shape:
            raise ValueError(f"fc1 mask shape {fc1_mask.shape} != {self.fc1.weight.shape}")
        if fc2_mask is None:
            fc2_mask = torch.ones_like(self.fc2.weight, dtype=torch.bool)
        if fc2_mask.shape != self.fc2.weight.shape:
            raise ValueError(f"fc2 mask shape {fc2_mask.shape} != {self.fc2.weight.shape}")
        self.register_buffer("fc1_mask", fc1_mask.bool())
        self.register_buffer("fc2_mask", fc2_mask.bool())

    @torch.no_grad()
    def apply_masks_(self) -> None:
        self.fc1.weight.masked_fill_(~self.fc1_mask, 0)
        self.fc2.weight.masked_fill_(~self.fc2_mask, 0)


def apply_model_masks(model: nn.Module) -> None:
    apply_masks = getattr(model, "apply_masks_", None)
    if callable(apply_masks):
        apply_masks()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_loaders(config: Config) -> tuple[DataLoader, DataLoader]:
    tfm = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    train_ds = datasets.MNIST(config.data_dir, train=True, download=True, transform=tfm)
    test_ds = datasets.MNIST(config.data_dir, train=False, download=True, transform=tfm)
    if config.train_subset > 0:
        train_ds = Subset(train_ds, list(range(min(config.train_subset, len(train_ds)))))
    if config.test_subset > 0:
        test_ds = Subset(test_ds, list(range(min(config.test_subset, len(test_ds)))))
    train = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    test = DataLoader(test_ds, batch_size=config.batch_size, shuffle=False, num_workers=4, pin_memory=True)
    return train, test


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> dict[str, float | int]:
    model.eval()
    correct = 0
    total = 0
    loss_sum = 0.0
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        logits = model(x)
        loss_sum += float(F.cross_entropy(logits, y, reduction="sum").item())
        correct += int((logits.argmax(dim=1) == y).sum().item())
        total += int(y.numel())
    return {"accuracy": correct / max(total, 1), "loss": loss_sum / max(total, 1), "correct": correct, "total": total}


def train_supervised(
    model: nn.Module,
    train_loader: DataLoader,
    test_loader: DataLoader,
    device: torch.device,
    *,
    epochs: int,
    lr: float,
    label: str,
) -> list[dict[str, Any]]:
    model.to(device)
    apply_model_masks(model)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    rows: list[dict[str, Any]] = []
    for epoch in range(1, epochs + 1):
        model.train()
        seen = 0
        loss_sum = 0.0
        for x, y in train_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            loss = F.cross_entropy(model(x), y)
            loss.backward()
            opt.step()
            apply_model_masks(model)
            seen += int(y.numel())
            loss_sum += float(loss.item()) * int(y.numel())
        ev = evaluate(model, test_loader, device)
        row = {"label": label, "epoch": epoch, "train_loss": loss_sum / seen, **ev}
        rows.append(row)
        print(json.dumps(row, sort_keys=True), flush=True)
    return rows


def train_distilled(
    student: nn.Module,
    teacher: nn.Module,
    train_loader: DataLoader,
    test_loader: DataLoader,
    device: torch.device,
    *,
    epochs: int,
    lr: float,
    label: str,
    alpha: float = 0.5,
    temperature: float = 3.0,
) -> list[dict[str, Any]]:
    student.to(device)
    apply_model_masks(student)
    teacher.to(device).eval()
    opt = torch.optim.AdamW(student.parameters(), lr=lr, weight_decay=1e-4)
    rows: list[dict[str, Any]] = []
    for epoch in range(1, epochs + 1):
        student.train()
        seen = 0
        loss_sum = 0.0
        for x, y in train_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            with torch.no_grad():
                t_logits = teacher(x)
            s_logits = student(x)
            ce = F.cross_entropy(s_logits, y)
            kd = F.kl_div(
                F.log_softmax(s_logits / temperature, dim=1),
                F.softmax(t_logits / temperature, dim=1),
                reduction="batchmean",
            ) * (temperature * temperature)
            loss = alpha * ce + (1.0 - alpha) * kd
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            apply_model_masks(student)
            seen += int(y.numel())
            loss_sum += float(loss.item()) * int(y.numel())
        ev = evaluate(student, test_loader, device)
        row = {"label": label, "epoch": epoch, "train_loss": loss_sum / seen, **ev}
        rows.append(row)
        print(json.dumps(row, sort_keys=True), flush=True)
    return rows


def param_count(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def max_window_mask(weight_images: torch.Tensor, window: int) -> torch.Tensor:
    if window >= 28:
        return torch.ones_like(weight_images, dtype=torch.bool)
    n = int(weight_images.shape[0])
    sq = weight_images.square().unsqueeze(1)
    energy = F.avg_pool2d(sq, kernel_size=window, stride=1) * (window * window)
    flat = energy.reshape(n, -1)
    idx = flat.argmax(dim=1)
    y = idx // (28 - window + 1)
    x = idx % (28 - window + 1)
    mask = torch.zeros_like(weight_images, dtype=torch.bool)
    for i in range(n):
        mask[i, y[i] : y[i] + window, x[i] : x[i] + window] = True
    return mask


def source_local_fc1_mask(source: OneHiddenMLP, window: int) -> torch.Tensor:
    images = source.fc1.weight.detach().cpu().reshape(source.fc1.out_features, 28, 28)
    return max_window_mask(images, window).reshape(source.fc1.weight.shape)


def random_local_fc1_mask(hidden: int, window: int, *, seed: int) -> torch.Tensor:
    if window >= 28:
        return torch.ones(hidden, 28 * 28, dtype=torch.bool)
    gen = torch.Generator().manual_seed(seed)
    starts = 28 - window + 1
    ys = torch.randint(0, starts, (hidden,), generator=gen)
    xs = torch.randint(0, starts, (hidden,), generator=gen)
    mask = torch.zeros(hidden, 28, 28, dtype=torch.bool)
    for i in range(hidden):
        mask[i, ys[i] : ys[i] + window, xs[i] : xs[i] + window] = True
    return mask.reshape(hidden, 28 * 28)


def per_neuron_topk_fc1_mask(source: OneHiddenMLP, kept_per_neuron: int) -> torch.Tensor:
    weight = source.fc1.weight.detach().cpu().abs()
    kept = min(max(1, kept_per_neuron), weight.shape[1])
    idx = weight.topk(kept, dim=1, largest=True).indices
    mask = torch.zeros_like(weight, dtype=torch.bool)
    mask.scatter_(1, idx, True)
    return mask


@torch.no_grad()
def localized_source_eval(
    source: OneHiddenMLP,
    test_loader: DataLoader,
    device: torch.device,
    windows: tuple[int, ...],
) -> list[dict[str, Any]]:
    original = source.fc1.weight.detach().clone()
    images = original.reshape(original.shape[0], 28, 28).cpu()
    rows: list[dict[str, Any]] = []
    for window in windows:
        mask = max_window_mask(images, window).reshape(original.shape).to(original.device)
        source.fc1.weight.copy_(original * mask)
        ev = evaluate(source, test_loader, device)
        kept = float(mask.float().mean().item())
        row = {"method": f"source_fc1_local_window_{window}", "window": window, "fc1_weight_fraction_kept": kept, **ev}
        rows.append(row)
        print(json.dumps(row, sort_keys=True), flush=True)
    source.fc1.weight.copy_(original)
    return rows


def init_masked_from_source(student: MaskedMLP, source: OneHiddenMLP) -> None:
    with torch.no_grad():
        student.fc1.weight.copy_(source.fc1.weight.detach().cpu())
        student.fc1.bias.copy_(source.fc1.bias.detach().cpu())
        student.fc2.weight.copy_(source.fc2.weight.detach().cpu())
        student.fc2.bias.copy_(source.fc2.bias.detach().cpu())
        student.apply_masks_()


def masked_resynthesis_eval(
    source: OneHiddenMLP,
    train_loader: DataLoader,
    test_loader: DataLoader,
    device: torch.device,
    *,
    windows: tuple[int, ...],
    epochs: int,
    lr: float,
    seed: int,
) -> tuple[list[dict[str, Any]], dict[str, list[dict[str, Any]]]]:
    rows: list[dict[str, Any]] = []
    histories: dict[str, list[dict[str, Any]]] = {}
    hidden = source.fc1.out_features
    for window in windows:
        local_mask = source_local_fc1_mask(source, window)
        random_mask = random_local_fc1_mask(hidden, window, seed=seed + window)
        variants = (
            (f"masked_mlp_local{window}_source_init_distilled", local_mask, True, "decompiled local fc1 windows"),
            (f"masked_mlp_randomlocal{window}_source_init_distilled", random_mask, True, "random local fc1 windows, source weights"),
            (f"masked_mlp_local{window}_random_init_distilled", local_mask, False, "decompiled local fc1 mask, random weights"),
            (
                f"masked_mlp_topk{window * window}_source_init_distilled",
                per_neuron_topk_fc1_mask(source, window * window),
                True,
                "per-neuron top-k fc1 weights, same density as local window",
            ),
        )
        for label, mask, source_init, note in variants:
            model = MaskedMLP(hidden, mask)
            if source_init:
                init_masked_from_source(model, source)
            hist = train_distilled(
                model,
                source,
                train_loader,
                test_loader,
                device,
                epochs=epochs,
                lr=lr,
                label=label,
                alpha=0.5,
                temperature=3.0,
            )
            nonzero = int(model.fc1_mask.sum().item() + model.fc2_mask.sum().item())
            total = int(model.fc1_mask.numel() + model.fc2_mask.numel())
            row = {
                "method": label,
                "window": window,
                "accuracy": hist[-1]["accuracy"],
                "params": nonzero + model.fc1.bias.numel() + model.fc2.bias.numel(),
                "active_weight_params": nonzero,
                "total_weight_params": total,
                "weight_fraction_left": nonzero / total,
                "note": note,
            }
            rows.append(row)
            histories[label] = hist
    return rows, histories


def activation_rms(model: OneHiddenMLP, loader: DataLoader, device: torch.device, max_batches: int = 20) -> tuple[torch.Tensor, torch.Tensor]:
    x_sumsq = torch.zeros(28 * 28, dtype=torch.float64)
    h_sumsq = torch.zeros(model.fc1.out_features, dtype=torch.float64)
    x_count = 0
    h_count = 0
    model.eval()
    with torch.no_grad():
        for i, (x, _y) in enumerate(loader):
            if i >= max_batches:
                break
            x = x.to(device, non_blocking=True)
            flat = x.reshape(x.shape[0], -1)
            h = model.hidden(x)
            x_sumsq += flat.float().square().sum(dim=0).double().cpu()
            h_sumsq += h.float().square().sum(dim=0).double().cpu()
            x_count += int(flat.shape[0])
            h_count += int(h.shape[0])
    return (x_sumsq / max(x_count, 1)).sqrt().float(), (h_sumsq / max(h_count, 1)).sqrt().float()


@torch.no_grad()
def pruned_copy_eval(
    source: OneHiddenMLP,
    train_loader: DataLoader,
    test_loader: DataLoader,
    device: torch.device,
    fracs: tuple[float, ...],
) -> list[dict[str, Any]]:
    x_rms, h_rms = activation_rms(source, train_loader, device)
    rows: list[dict[str, Any]] = []
    for score_name in ("magnitude", "wanda"):
        for frac in fracs:
            model = OneHiddenMLP(source.fc1.out_features).to(device)
            model.load_state_dict(source.state_dict())
            scores = {}
            if score_name == "magnitude":
                scores["fc1.weight"] = model.fc1.weight.detach().abs().cpu()
                scores["fc2.weight"] = model.fc2.weight.detach().abs().cpu()
            else:
                scores["fc1.weight"] = model.fc1.weight.detach().abs().cpu() * x_rms.reshape(1, -1)
                scores["fc2.weight"] = model.fc2.weight.detach().abs().cpu() * h_rms.reshape(1, -1)
            all_scores = torch.cat([scores["fc1.weight"].flatten(), scores["fc2.weight"].flatten()])
            k = int(all_scores.numel() * frac)
            threshold = torch.topk(all_scores, k=max(1, k), largest=False).values.max()
            model.fc1.weight.masked_fill_(scores["fc1.weight"].to(device) <= threshold.to(device), 0)
            model.fc2.weight.masked_fill_(scores["fc2.weight"].to(device) <= threshold.to(device), 0)
            ev = evaluate(model, test_loader, device)
            nonzero = int((model.fc1.weight != 0).sum().item() + (model.fc2.weight != 0).sum().item())
            total = int(model.fc1.weight.numel() + model.fc2.weight.numel())
            row = {
                "method": f"{score_name}_source_prune_{frac:.2f}",
                "score": score_name,
                "prune_fraction": frac,
                "nonzero_weight_params": nonzero,
                "total_weight_params": total,
                "weight_fraction_left": nonzero / total,
                **ev,
            }
            rows.append(row)
            print(json.dumps(row, sort_keys=True), flush=True)
    return rows


def decompile_patches(source: OneHiddenMLP, *, patch_size: int, filters: int) -> tuple[torch.Tensor, list[dict[str, Any]]]:
    w = source.fc1.weight.detach().cpu().reshape(source.fc1.out_features, 28, 28)
    out = source.fc2.weight.detach().cpu()
    class_strength = out.abs().amax(dim=0)
    local_mask = max_window_mask(w, patch_size)
    local_energy = (w.square() * local_mask).sum(dim=(1, 2))
    total_energy = w.square().sum(dim=(1, 2)).clamp_min(1e-12)
    locality = local_energy / total_energy
    score = class_strength * locality * total_energy.sqrt()
    ranked = torch.argsort(score, descending=True)

    patches: list[torch.Tensor] = []
    rows: list[dict[str, Any]] = []
    used = 0
    for neuron in ranked.tolist():
        if used >= filters:
            break
        image = w[neuron]
        mask = max_window_mask(image.unsqueeze(0), patch_size)[0]
        ys, xs = torch.where(mask)
        y0, y1 = int(ys.min().item()), int(ys.max().item()) + 1
        x0, x1 = int(xs.min().item()), int(xs.max().item()) + 1
        patch = image[y0:y1, x0:x1].clone()
        if patch.shape != (patch_size, patch_size):
            continue
        patch = patch - patch.mean()
        patch = patch / patch.norm().clamp_min(1e-6)
        patches.append(patch.reshape(1, patch_size, patch_size))
        rows.append(
            {
                "rank": used,
                "source_neuron": neuron,
                "y": y0,
                "x": x0,
                "locality": float(locality[neuron].item()),
                "class_strength": float(class_strength[neuron].item()),
                "score": float(score[neuron].item()),
                "preferred_class": int(out[:, neuron].abs().argmax().item()),
            }
        )
        used += 1
    filters_tensor = torch.stack(patches, dim=0)
    return filters_tensor, rows


def init_conv_from_patches(model: TinyConv, filters: torch.Tensor) -> None:
    with torch.no_grad():
        count = min(model.conv.weight.shape[0], filters.shape[0])
        model.conv.weight[:count].copy_(filters[:count])
        model.conv.bias.zero_()


def decomp_metrics(source: OneHiddenMLP, filters: torch.Tensor, patch_rows: list[dict[str, Any]]) -> dict[str, Any]:
    w = source.fc1.weight.detach().cpu().reshape(source.fc1.out_features, 28, 28)
    metrics: dict[str, Any] = {}
    for window in (3, 5, 7, 9, 11, 15):
        mask = max_window_mask(w, window)
        frac = (w.square() * mask).sum(dim=(1, 2)) / w.square().sum(dim=(1, 2)).clamp_min(1e-12)
        metrics[f"local_energy_top_{window}x{window}_mean"] = float(frac.mean().item())
        metrics[f"local_energy_top_{window}x{window}_p75"] = float(frac.quantile(0.75).item())

    flat = filters.reshape(filters.shape[0], -1).numpy()
    k = min(8, filters.shape[0])
    if KMeans is not None and k >= 2:
        labels = KMeans(n_clusters=k, random_state=0, n_init=10).fit_predict(flat)
        counts = np.bincount(labels, minlength=k)
        metrics["patch_cluster_counts"] = counts.tolist()
    elif k >= 2:
        metrics["patch_cluster_counts"] = None
        metrics["patch_cluster_counts_note"] = "skipped because scikit-learn is not installed"
    metrics["patch_rows"] = patch_rows
    return metrics


def write_report(path: Path, summary: dict[str, Any]) -> None:
    lines = ["# MNIST Circuit Resynthesis", ""]
    lines.append(f"Source accuracy: `{summary['source_final']['accuracy']:.4f}` with `{summary['params']['source_mlp']}` params.")
    lines.append("")
    lines.append("## Final Results")
    lines.append("")
    lines.append("| method | accuracy | params | note |")
    lines.append("| --- | ---: | ---: | --- |")
    for row in summary["final_table"]:
        lines.append(
            f"| `{row['method']}` | {row['accuracy']:.4f} | {row.get('params', '')} | {row.get('note', '')} |"
        )
    lines.append("")
    lines.append("## Localizing Source FC1")
    lines.append("")
    lines.append("| window | fc1 fraction kept | accuracy |")
    lines.append("| ---: | ---: | ---: |")
    for row in summary["localized_source"]:
        lines.append(f"| {row['window']} | {row['fc1_weight_fraction_kept']:.4f} | {row['accuracy']:.4f} |")
    lines.append("")
    lines.append("## Masked Local Resynthesis")
    lines.append("")
    lines.append("| method | window | weight fraction left | active params | accuracy |")
    lines.append("| --- | ---: | ---: | ---: | ---: |")
    for row in summary["masked_resynthesis"]:
        lines.append(
            f"| `{row['method']}` | {row['window']} | {row['weight_fraction_left']:.4f} | {row['active_weight_params']} | {row['accuracy']:.4f} |"
        )
    lines.append("")
    lines.append("## Read")
    lines.extend(summary["read"])
    path.write_text("\n".join(lines) + "\n")


def run(config: Config) -> dict[str, Any]:
    set_seed(config.seed)
    torch.set_float32_matmul_precision("high")
    device = torch.device(config.device)
    config.output_dir.mkdir(parents=True, exist_ok=True)
    train_loader, test_loader = get_loaders(config)

    started = time.monotonic()
    source = OneHiddenMLP(config.source_hidden)
    source_hist = train_supervised(
        source,
        train_loader,
        test_loader,
        device,
        epochs=config.source_epochs,
        lr=config.lr,
        label="source_mlp",
    )
    source_final = source_hist[-1]

    localized = localized_source_eval(source, test_loader, device, config.local_windows)
    pruned = pruned_copy_eval(source, train_loader, test_loader, device, config.prune_fracs)
    masked_rows, masked_histories = masked_resynthesis_eval(
        source,
        train_loader,
        test_loader,
        device,
        windows=config.masked_windows,
        epochs=config.distill_epochs,
        lr=config.student_lr,
        seed=config.seed,
    )
    filters, patch_rows = decompile_patches(source, patch_size=config.patch_size, filters=config.conv_filters)
    metrics = decomp_metrics(source, filters, patch_rows)

    student_histories: dict[str, list[dict[str, Any]]] = {**masked_histories}
    conv_rows: list[dict[str, Any]] = []
    if config.run_conv:
        scratch = TinyConv(config.conv_filters, config.patch_size)
        scratch_hist = train_supervised(
            scratch,
            train_loader,
            test_loader,
            device,
            epochs=config.student_epochs,
            lr=config.student_lr,
            label="tiny_conv_scratch",
        )

        distilled = TinyConv(config.conv_filters, config.patch_size)
        distilled_hist = train_distilled(
            distilled,
            source,
            train_loader,
            test_loader,
            device,
            epochs=config.distill_epochs,
            lr=config.student_lr,
            label="tiny_conv_distilled_random_init",
        )

        circuit = TinyConv(config.conv_filters, config.patch_size)
        init_conv_from_patches(circuit, filters)
        circuit_hist = train_distilled(
            circuit,
            source,
            train_loader,
            test_loader,
            device,
            epochs=config.distill_epochs,
            lr=config.student_lr,
            label="tiny_conv_distilled_circuit_init",
        )

        fixed = FixedStrokeBank(filters)
        fixed_hist = train_distilled(
            fixed,
            source,
            train_loader,
            test_loader,
            device,
            epochs=config.distill_epochs,
            lr=config.student_lr,
            label="fixed_stroke_bank_distilled",
        )
        student_histories.update(
            {
                "tiny_conv_scratch": scratch_hist,
                "tiny_conv_distilled_random_init": distilled_hist,
                "tiny_conv_distilled_circuit_init": circuit_hist,
                "fixed_stroke_bank_distilled": fixed_hist,
            }
        )
        conv_rows = [
            {"method": "tiny_conv_scratch", "accuracy": scratch_hist[-1]["accuracy"], "params": param_count(scratch), "note": "same conv architecture, no teacher"},
            {"method": "tiny_conv_distilled_random_init", "accuracy": distilled_hist[-1]["accuracy"], "params": param_count(distilled), "note": "black-box distillation"},
            {"method": "tiny_conv_distilled_circuit_init", "accuracy": circuit_hist[-1]["accuracy"], "params": param_count(circuit), "note": "conv filters initialized from decompiled MLP patches"},
            {"method": "fixed_stroke_bank_distilled", "accuracy": fixed_hist[-1]["accuracy"], "params": param_count(fixed), "note": "decompiled filters frozen, train only head"},
        ]

    final_table: list[dict[str, Any]] = [
        {"method": "source_mlp", "accuracy": source_final["accuracy"], "params": param_count(source), "note": "bad substrate teacher"},
    ]
    final_table.extend(conv_rows)
    for row in pruned:
        final_table.append(
            {
                "method": row["method"],
                "accuracy": row["accuracy"],
                "params": row["nonzero_weight_params"],
                "note": f"{row['score']} no-retrain source pruning",
            }
        )
    final_table.extend(masked_rows)
    final_table.sort(key=lambda row: float(row["accuracy"]), reverse=True)

    best_masked = max(masked_rows, key=lambda row: float(row["accuracy"])) if masked_rows else None
    read = [
        f"- Best local sparse resynthesis: `{best_masked['method']}` at `{best_masked['accuracy']:.4f}` accuracy with `{best_masked['weight_fraction_left']:.4f}` of weight params left." if best_masked else "- Local sparse resynthesis was not run.",
        "- The source-local-window table tests whether the MLP first layer is actually local enough to recompile into conv filters.",
        "- The masked MLP rows test a weaker target than conv: position-specific local circuits with the same hidden width, compared against random local masks.",
        "- Magnitude/Wanda pruning are included as dumb compression baselines; they do not change architecture.",
    ]
    if config.run_conv:
        circuit_acc = next(row["accuracy"] for row in conv_rows if row["method"] == "tiny_conv_distilled_circuit_init")
        random_acc = next(row["accuracy"] for row in conv_rows if row["method"] == "tiny_conv_distilled_random_init")
        read.insert(
            0,
            f"- Decompiled circuit initialization {'helped' if circuit_acc > random_acc else 'did not beat'} plain conv distillation in this run: `{circuit_acc:.4f}` vs `{random_acc:.4f}`.",
        )
        read.insert(
            4,
            "- Fixed stroke-bank performance tests whether the recovered patches are sufficient features without learning the conv bank.",
        )

    summary = {
        "config": {**asdict(config), "output_dir": str(config.output_dir), "data_dir": str(config.data_dir)},
        "elapsed_seconds": time.monotonic() - started,
        "source_history": source_hist,
        "source_final": source_final,
        "localized_source": localized,
        "pruned_source": pruned,
        "masked_resynthesis": masked_rows,
        "decomp_metrics": metrics,
        "student_histories": student_histories,
        "params": {
            "source_mlp": param_count(source),
            "tiny_conv": param_count(TinyConv(config.conv_filters, config.patch_size)),
            "fixed_stroke_bank": param_count(FixedStrokeBank(filters)),
        },
        "final_table": final_table,
        "read": read,
    }
    (config.output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True, default=str) + "\n")
    write_report(config.output_dir / "summary.md", summary)
    print("SUMMARY", json.dumps({"output_dir": str(config.output_dir), "elapsed_seconds": summary["elapsed_seconds"], "best": final_table[0]}, sort_keys=True), flush=True)
    return summary


def parse_tuple_int(value: str) -> tuple[int, ...]:
    return tuple(int(v.strip()) for v in value.split(",") if v.strip())


def parse_tuple_float(value: str) -> tuple[float, ...]:
    return tuple(float(v.strip()) for v in value.split(",") if v.strip())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Config.output_dir)
    parser.add_argument("--data-dir", type=Path, default=Config.data_dir)
    parser.add_argument("--seed", type=int, default=Config.seed)
    parser.add_argument("--batch-size", type=int, default=Config.batch_size)
    parser.add_argument("--source-hidden", type=int, default=Config.source_hidden)
    parser.add_argument("--source-epochs", type=int, default=Config.source_epochs)
    parser.add_argument("--student-epochs", type=int, default=Config.student_epochs)
    parser.add_argument("--distill-epochs", type=int, default=Config.distill_epochs)
    parser.add_argument("--lr", type=float, default=Config.lr)
    parser.add_argument("--student-lr", type=float, default=Config.student_lr)
    parser.add_argument("--train-subset", type=int, default=Config.train_subset)
    parser.add_argument("--test-subset", type=int, default=Config.test_subset)
    parser.add_argument("--conv-filters", type=int, default=Config.conv_filters)
    parser.add_argument("--patch-size", type=int, default=Config.patch_size)
    parser.add_argument("--local-windows", default="3,5,7,9,11,15")
    parser.add_argument("--masked-windows", default="7,11,15")
    parser.add_argument("--prune-fracs", default="0.5,0.8,0.9,0.95")
    parser.add_argument("--skip-conv", action="store_true")
    parser.add_argument("--device", default=Config.device)
    args = parser.parse_args()

    config = Config(
        output_dir=args.output_dir,
        data_dir=args.data_dir,
        seed=args.seed,
        batch_size=args.batch_size,
        source_hidden=args.source_hidden,
        source_epochs=args.source_epochs,
        student_epochs=args.student_epochs,
        distill_epochs=args.distill_epochs,
        lr=args.lr,
        student_lr=args.student_lr,
        train_subset=args.train_subset,
        test_subset=args.test_subset,
        conv_filters=args.conv_filters,
        patch_size=args.patch_size,
        local_windows=parse_tuple_int(args.local_windows),
        masked_windows=parse_tuple_int(args.masked_windows),
        prune_fracs=parse_tuple_float(args.prune_fracs),
        run_conv=not args.skip_conv,
        device=args.device,
    )
    run(config)


if __name__ == "__main__":
    main()
