from __future__ import annotations

import argparse
import json
import random
import re
import time
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm.auto import tqdm

from eval_da2k import (
    MODEL_CONFIGS,
    add_pair,
    empty_counts,
    finalize_counts,
    load_model,
    point_value,
    resolve_device,
    scene_from_path,
    selected_annotations,
)


TRANSFORMER_CIRCUIT_KINDS = (
    "attn_q_head",
    "attn_k_head",
    "attn_v_head",
    "attn_q_group",
    "attn_k_group",
    "attn_v_group",
    "attn_proj_group",
    "mlp_group",
)
DEPTH_HEAD_CIRCUIT_KINDS = (
    "head_channel_group",
    "head_input_channel_group",
)
ALL_CIRCUIT_KINDS = TRANSFORMER_CIRCUIT_KINDS + DEPTH_HEAD_CIRCUIT_KINDS

TRANSFORMER_LINEAR_SUFFIXES = ("attn.qkv", "attn.proj", "mlp.fc1", "mlp.fc2")
PEFT_METHODS = ("lora", "lora-bitfit", "dora", "loha", "sapt", "ia3-out", "ia3-in", "bitfit")
LORA_PLACEMENTS = (
    "masked",
    "head",
    "head-masked",
    "same-block",
    "previous-block",
    "next-block",
    "previous-window",
    "next-window",
    "before-and-masked",
    "around-window",
    "prefix",
    "all-prior",
)


@dataclass(frozen=True)
class CircuitLoRARepairConfig:
    dataset_root: Path
    checkpoint: Path
    circuit_summary: Path
    output_dir: Path
    encoder: str = "vits"
    input_size: int = 518
    device: str = "auto"
    scene_type: str = ""
    train_images: int = 24
    eval_skip_images: int = 24
    eval_images: int = 64
    allow_train_eval_overlap: bool = False
    budget_nodes: int = 8
    budget_values: int = 0
    candidate_kinds: tuple[str, ...] = TRANSFORMER_CIRCUIT_KINDS
    selection: str = "weak"
    peft_method: str = "lora"
    lora_placement: str = "masked"
    lora_window: int = 1
    lora_module_set: str = "all"
    lora_rank: int = 8
    lora_alpha: float = 16.0
    epochs: int = 3
    lr: float = 2e-3
    weight_decay: float = 0.0
    optimizer: str = "adamw"
    muon_momentum: float = 0.95
    muon_damping: float = 1e-2
    depth_loss_mode: str = "smoothl1"
    kl_direction: str = "forward"
    kl_temperature: float = 1.0
    pairwise_weight: float = 0.0
    pairwise_teacher_weight: float = 1.0
    pairwise_label_weight: float = 0.0
    pairwise_tau: float = 0.25
    seed: int = 123
    log_every: int = 12
    save_checkpoint: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "dataset_root", Path(self.dataset_root))
        object.__setattr__(self, "checkpoint", Path(self.checkpoint))
        object.__setattr__(self, "circuit_summary", Path(self.circuit_summary))
        object.__setattr__(self, "output_dir", Path(self.output_dir))
        if self.encoder not in MODEL_CONFIGS:
            raise ValueError(f"unknown encoder: {self.encoder}")
        if self.train_images <= 0:
            raise ValueError("train_images must be positive")
        if self.eval_skip_images < 0 or self.eval_images < 0:
            raise ValueError("eval_skip_images/eval_images must be non-negative")
        if not self.allow_train_eval_overlap and self.eval_skip_images < self.train_images:
            raise ValueError("eval_skip_images must be >= train_images unless --allow-train-eval-overlap is set")
        if self.budget_nodes <= 0:
            raise ValueError("budget_nodes must be positive")
        if self.budget_values < 0:
            raise ValueError("budget_values must be non-negative")
        if self.selection not in {"weak", "safe", "random", "stability", "stability_param", "stability-param"}:
            raise ValueError("selection must be weak, safe, random, stability, or stability_param")
        if self.peft_method not in PEFT_METHODS:
            raise ValueError(f"peft_method must be one of {PEFT_METHODS}")
        placements = tuple(part for part in self.lora_placement.split("+") if part)
        if not placements or any(placement not in LORA_PLACEMENTS for placement in placements):
            raise ValueError("unknown lora_placement")
        if self.lora_window < 0:
            raise ValueError("lora_window must be non-negative")
        if self.lora_module_set not in {"all", "attn", "mlp"}:
            raise ValueError("lora_module_set must be all, attn, or mlp")
        if self.lora_rank <= 0:
            raise ValueError("lora_rank must be positive")
        if self.epochs <= 0:
            raise ValueError("epochs must be positive")
        if self.lr <= 0:
            raise ValueError("lr must be positive")
        if self.optimizer not in {"adamw", "muon", "comp-muon"}:
            raise ValueError("optimizer must be adamw, muon, or comp-muon")
        if not 0 <= self.muon_momentum < 1:
            raise ValueError("muon_momentum must be in [0, 1)")
        if self.muon_damping <= 0:
            raise ValueError("muon_damping must be positive")
        if self.depth_loss_mode not in {"smoothl1", "l2", "kl"}:
            raise ValueError("depth_loss_mode must be smoothl1, l2, or kl")
        if self.kl_direction not in {"forward", "reverse"}:
            raise ValueError("kl_direction must be forward or reverse")
        if self.kl_temperature <= 0:
            raise ValueError("kl_temperature must be positive")
        if self.pairwise_weight < 0 or self.pairwise_teacher_weight < 0 or self.pairwise_label_weight < 0:
            raise ValueError("pairwise weights must be non-negative")
        if self.pairwise_tau <= 0:
            raise ValueError("pairwise_tau must be positive")


@dataclass(frozen=True)
class TrainSample:
    relative_path: str
    tensor: torch.Tensor
    height: int
    width: int
    point1_rows: torch.Tensor
    point1_cols: torch.Tensor
    point2_rows: torch.Tensor
    point2_cols: torch.Tensor


class LoRALinear(nn.Module):
    def __init__(
        self,
        base: nn.Linear,
        *,
        rank: int,
        alpha: float,
        pruned_weight_mask: torch.Tensor | None = None,
    ):
        super().__init__()
        self.base = base
        self.rank = int(rank)
        self.alpha = float(alpha)
        self.scaling = self.alpha / float(self.rank)
        self.lora_a = nn.Linear(base.in_features, self.rank, bias=False)
        self.lora_b = nn.Linear(self.rank, base.out_features, bias=False)
        nn.init.kaiming_uniform_(self.lora_a.weight, a=5**0.5)
        nn.init.zeros_(self.lora_b.weight)
        self.base.weight.requires_grad_(False)
        if self.base.bias is not None:
            self.base.bias.requires_grad_(False)
        if pruned_weight_mask is None:
            self.register_buffer("lora_weight_mask", None)
        else:
            if tuple(pruned_weight_mask.shape) != tuple(base.weight.shape):
                raise ValueError("pruned_weight_mask must match base weight shape")
            allowed_mask = (~pruned_weight_mask.bool()).to(dtype=base.weight.dtype)
            self.register_buffer("lora_weight_mask", allowed_mask)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.lora_weight_mask is None:
            return self.base(x) + self.lora_b(self.lora_a(x)) * self.scaling
        delta = self.lora_b.weight @ self.lora_a.weight
        delta = delta * self.lora_weight_mask.to(device=delta.device, dtype=delta.dtype)
        return self.base(x) + F.linear(x, delta) * self.scaling

    @torch.no_grad()
    def merge(self) -> nn.Linear:
        delta = self.lora_b.weight @ self.lora_a.weight
        if self.lora_weight_mask is not None:
            delta = delta * self.lora_weight_mask.to(device=delta.device, dtype=delta.dtype)
        self.base.weight.add_(delta.to(device=self.base.weight.device, dtype=self.base.weight.dtype) * self.scaling)
        return self.base


class LoRABitFitLinear(LoRALinear):
    def __init__(
        self,
        base: nn.Linear,
        *,
        rank: int,
        alpha: float,
        pruned_weight_mask: torch.Tensor | None = None,
    ):
        super().__init__(base, rank=rank, alpha=alpha, pruned_weight_mask=pruned_weight_mask)
        self.bias_delta = nn.Parameter(torch.zeros(base.out_features, dtype=base.weight.dtype))
        if pruned_weight_mask is None:
            self.register_buffer("bias_allowed_mask", None)
        else:
            allowed = (~pruned_weight_mask.bool()).any(dim=1).to(dtype=base.weight.dtype)
            self.register_buffer("bias_allowed_mask", allowed)

    def effective_bias_delta(self) -> torch.Tensor:
        if self.bias_allowed_mask is None:
            return self.bias_delta
        return self.bias_delta * self.bias_allowed_mask.to(device=self.bias_delta.device, dtype=self.bias_delta.dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return super().forward(x) + self.effective_bias_delta().to(device=x.device, dtype=x.dtype)

    @torch.no_grad()
    def merge(self) -> nn.Linear:
        merged = super().merge()
        delta = self.effective_bias_delta().to(device=merged.weight.device, dtype=merged.weight.dtype)
        if merged.bias is None:
            merged.bias = nn.Parameter(delta.clone())
        else:
            merged.bias.add_(delta)
        return merged


class LoHALinear(nn.Module):
    def __init__(
        self,
        base: nn.Linear,
        *,
        rank: int,
        alpha: float,
        pruned_weight_mask: torch.Tensor | None = None,
    ):
        super().__init__()
        self.base = base
        self.rank = int(rank)
        self.alpha = float(alpha)
        self.scaling = self.alpha / float(self.rank)
        self.hada_a1 = nn.Linear(base.in_features, self.rank, bias=False)
        self.hada_b1 = nn.Linear(self.rank, base.out_features, bias=False)
        self.hada_a2 = nn.Linear(base.in_features, self.rank, bias=False)
        self.hada_b2 = nn.Linear(self.rank, base.out_features, bias=False)
        nn.init.kaiming_uniform_(self.hada_a1.weight, a=5**0.5)
        nn.init.zeros_(self.hada_b1.weight)
        nn.init.kaiming_uniform_(self.hada_a2.weight, a=5**0.5)
        nn.init.kaiming_uniform_(self.hada_b2.weight, a=5**0.5)
        self.base.weight.requires_grad_(False)
        if self.base.bias is not None:
            self.base.bias.requires_grad_(False)
        if pruned_weight_mask is None:
            self.register_buffer("loha_weight_mask", None)
        else:
            if tuple(pruned_weight_mask.shape) != tuple(base.weight.shape):
                raise ValueError("pruned_weight_mask must match base weight shape")
            allowed_mask = (~pruned_weight_mask.bool()).to(dtype=base.weight.dtype)
            self.register_buffer("loha_weight_mask", allowed_mask)

    def delta_weight(self) -> torch.Tensor:
        delta = (self.hada_b1.weight @ self.hada_a1.weight) * (self.hada_b2.weight @ self.hada_a2.weight)
        if self.loha_weight_mask is not None:
            delta = delta * self.loha_weight_mask.to(device=delta.device, dtype=delta.dtype)
        return delta * self.scaling

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base(x) + F.linear(x, self.delta_weight())

    @torch.no_grad()
    def merge(self) -> nn.Linear:
        self.base.weight.add_(self.delta_weight().to(device=self.base.weight.device, dtype=self.base.weight.dtype))
        return self.base


class SAPTLinear(nn.Module):
    def __init__(
        self,
        base: nn.Linear,
        *,
        rank: int,
        alpha: float,
        pruned_weight_mask: torch.Tensor | None = None,
    ):
        super().__init__()
        self.base = base
        self.rank = int(rank)
        self.alpha = float(alpha)
        self.scaling = self.alpha / float(self.rank)
        self.lora_a = nn.Parameter(torch.empty(self.rank, base.in_features, dtype=base.weight.dtype))
        self.lora_b = nn.Parameter(torch.zeros(base.out_features, self.rank, dtype=base.weight.dtype))
        nn.init.kaiming_uniform_(self.lora_a, a=5**0.5)

        weight = base.weight.detach().float()
        svd_rank = min(self.rank, min(weight.shape))
        u, _s, vh = torch.linalg.svd(weight, full_matrices=False)
        self.register_buffer("sapt_u", u[:, :svd_rank].contiguous().to(dtype=base.weight.dtype))
        self.register_buffer("sapt_vh", vh[:svd_rank, :].contiguous().to(dtype=base.weight.dtype))
        self.spectral_diag = nn.Parameter(torch.zeros(svd_rank, dtype=base.weight.dtype))
        self.square_core = nn.Parameter(torch.zeros(svd_rank, svd_rank, dtype=base.weight.dtype))

        self.hada_a1 = nn.Parameter(torch.empty(self.rank, base.in_features, dtype=base.weight.dtype))
        self.hada_b1 = nn.Parameter(torch.zeros(base.out_features, self.rank, dtype=base.weight.dtype))
        self.hada_a2 = nn.Parameter(torch.empty(self.rank, base.in_features, dtype=base.weight.dtype))
        self.hada_b2 = nn.Parameter(torch.empty(base.out_features, self.rank, dtype=base.weight.dtype))
        nn.init.kaiming_uniform_(self.hada_a1, a=5**0.5)
        nn.init.kaiming_uniform_(self.hada_a2, a=5**0.5)
        nn.init.kaiming_uniform_(self.hada_b2, a=5**0.5)

        self.sapt_gates = nn.Parameter(torch.ones(4, dtype=base.weight.dtype))
        self.base.weight.requires_grad_(False)
        if self.base.bias is not None:
            self.base.bias.requires_grad_(False)
        if pruned_weight_mask is None:
            self.register_buffer("sapt_weight_mask", None)
        else:
            if tuple(pruned_weight_mask.shape) != tuple(base.weight.shape):
                raise ValueError("pruned_weight_mask must match base weight shape")
            allowed_mask = (~pruned_weight_mask.bool()).to(dtype=base.weight.dtype)
            self.register_buffer("sapt_weight_mask", allowed_mask)

    def delta_weight(self) -> torch.Tensor:
        gates = self.sapt_gates.to(device=self.base.weight.device, dtype=self.base.weight.dtype)
        lora_delta = self.lora_b @ self.lora_a
        spectral_delta = (self.sapt_u * self.spectral_diag.unsqueeze(0)) @ self.sapt_vh
        hada_delta = (self.hada_b1 @ self.hada_a1) * (self.hada_b2 @ self.hada_a2)
        square_delta = self.sapt_u @ self.square_core @ self.sapt_vh
        delta = (
            gates[0] * lora_delta
            + gates[1] * spectral_delta
            + gates[2] * hada_delta
            + gates[3] * square_delta
        )
        if self.sapt_weight_mask is not None:
            delta = delta * self.sapt_weight_mask.to(device=delta.device, dtype=delta.dtype)
        return delta * self.scaling

    def effective_weight(self) -> torch.Tensor:
        return self.base.weight + self.delta_weight().to(device=self.base.weight.device, dtype=self.base.weight.dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.effective_weight().to(device=x.device, dtype=x.dtype), self.base.bias)

    @torch.no_grad()
    def merge(self) -> nn.Linear:
        self.base.weight.copy_(self.effective_weight().to(device=self.base.weight.device, dtype=self.base.weight.dtype))
        return self.base


class DoRALinear(nn.Module):
    def __init__(
        self,
        base: nn.Linear,
        *,
        rank: int,
        alpha: float,
        pruned_weight_mask: torch.Tensor | None = None,
    ):
        super().__init__()
        self.base = base
        self.rank = int(rank)
        self.alpha = float(alpha)
        self.scaling = self.alpha / float(self.rank)
        self.lora_a = nn.Linear(base.in_features, self.rank, bias=False)
        self.lora_b = nn.Linear(self.rank, base.out_features, bias=False)
        nn.init.kaiming_uniform_(self.lora_a.weight, a=5**0.5)
        nn.init.zeros_(self.lora_b.weight)
        magnitude = base.weight.detach().float().norm(dim=1).to(dtype=base.weight.dtype)
        self.dora_magnitude = nn.Parameter(magnitude)
        self.base.weight.requires_grad_(False)
        if self.base.bias is not None:
            self.base.bias.requires_grad_(False)
        if pruned_weight_mask is None:
            self.register_buffer("dora_weight_mask", None)
        else:
            if tuple(pruned_weight_mask.shape) != tuple(base.weight.shape):
                raise ValueError("pruned_weight_mask must match base weight shape")
            allowed_mask = (~pruned_weight_mask.bool()).to(dtype=base.weight.dtype)
            self.register_buffer("dora_weight_mask", allowed_mask)

    def effective_weight(self) -> torch.Tensor:
        delta = self.lora_b.weight @ self.lora_a.weight
        if self.dora_weight_mask is not None:
            delta = delta * self.dora_weight_mask.to(device=delta.device, dtype=delta.dtype)
        direction = self.base.weight + delta.to(device=self.base.weight.device, dtype=self.base.weight.dtype) * self.scaling
        norm = direction.float().norm(dim=1, keepdim=True).clamp_min(1e-6).to(dtype=direction.dtype)
        magnitude = self.dora_magnitude.to(device=direction.device, dtype=direction.dtype).unsqueeze(1)
        return direction / norm * magnitude

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.effective_weight(), self.base.bias)

    @torch.no_grad()
    def merge(self) -> nn.Linear:
        self.base.weight.copy_(self.effective_weight().to(device=self.base.weight.device, dtype=self.base.weight.dtype))
        return self.base


class IA3Linear(nn.Module):
    def __init__(self, base: nn.Linear, *, mode: str):
        super().__init__()
        if mode not in {"in", "out"}:
            raise ValueError("IA3 mode must be in or out")
        self.base = base
        self.mode = mode
        dim = base.in_features if mode == "in" else base.out_features
        self.ia3_scale = nn.Parameter(torch.ones(dim, dtype=base.weight.dtype))
        self.base.weight.requires_grad_(False)
        if self.base.bias is not None:
            self.base.bias.requires_grad_(False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scale = self.ia3_scale.to(device=x.device, dtype=x.dtype)
        if self.mode == "in":
            return self.base(x * scale)
        return self.base(x) * scale

    @torch.no_grad()
    def merge(self) -> nn.Linear:
        scale = self.ia3_scale.to(device=self.base.weight.device, dtype=self.base.weight.dtype)
        if self.mode == "in":
            self.base.weight.mul_(scale.unsqueeze(0))
        else:
            self.base.weight.mul_(scale.unsqueeze(1))
            if self.base.bias is not None:
                self.base.bias.mul_(scale)
        return self.base


class BitFitLinear(nn.Module):
    def __init__(self, base: nn.Linear, *, pruned_weight_mask: torch.Tensor | None = None):
        super().__init__()
        self.base = base
        self.bias_delta = nn.Parameter(torch.zeros(base.out_features, dtype=base.weight.dtype))
        self.base.weight.requires_grad_(False)
        if self.base.bias is not None:
            self.base.bias.requires_grad_(False)
        if pruned_weight_mask is None:
            self.register_buffer("bias_allowed_mask", None)
        else:
            if tuple(pruned_weight_mask.shape) != tuple(base.weight.shape):
                raise ValueError("pruned_weight_mask must match base weight shape")
            allowed = (~pruned_weight_mask.bool()).any(dim=1).to(dtype=base.weight.dtype)
            self.register_buffer("bias_allowed_mask", allowed)

    def effective_bias_delta(self) -> torch.Tensor:
        if self.bias_allowed_mask is None:
            return self.bias_delta
        return self.bias_delta * self.bias_allowed_mask.to(device=self.bias_delta.device, dtype=self.bias_delta.dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base(x) + self.effective_bias_delta().to(device=x.device, dtype=x.dtype)

    @torch.no_grad()
    def merge(self) -> nn.Linear:
        delta = self.effective_bias_delta().to(device=self.base.weight.device, dtype=self.base.weight.dtype)
        if self.base.bias is None:
            self.base.bias = nn.Parameter(delta.clone())
        else:
            self.base.bias.add_(delta)
        return self.base


class LoRAConv(nn.Module):
    def __init__(
        self,
        base: nn.Conv2d | nn.ConvTranspose2d,
        *,
        rank: int,
        alpha: float,
        pruned_weight_mask: torch.Tensor | None = None,
    ):
        super().__init__()
        if base.groups != 1:
            raise ValueError("LoRAConv currently supports groups=1 only")
        self.base = base
        self.rank = int(rank)
        self.alpha = float(alpha)
        self.scaling = self.alpha / float(self.rank)
        if isinstance(base, nn.ConvTranspose2d):
            logical_out = int(base.out_channels)
            logical_in = int(base.in_channels * base.kernel_size[0] * base.kernel_size[1])
        else:
            logical_out = int(base.out_channels)
            logical_in = int(base.in_channels * base.kernel_size[0] * base.kernel_size[1])
        self.logical_out = logical_out
        self.logical_in = logical_in
        self.lora_a = nn.Parameter(torch.empty(self.rank, logical_in, dtype=base.weight.dtype))
        self.lora_b = nn.Parameter(torch.zeros(logical_out, self.rank, dtype=base.weight.dtype))
        nn.init.kaiming_uniform_(self.lora_a, a=5**0.5)
        self.base.weight.requires_grad_(False)
        if self.base.bias is not None:
            self.base.bias.requires_grad_(False)
        if pruned_weight_mask is None:
            self.register_buffer("lora_weight_mask", None)
        else:
            if tuple(pruned_weight_mask.shape) != tuple(base.weight.shape):
                raise ValueError("pruned_weight_mask must match base weight shape")
            allowed_mask = (~pruned_weight_mask.bool()).to(dtype=base.weight.dtype)
            self.register_buffer("lora_weight_mask", allowed_mask)

    def delta_weight(self) -> torch.Tensor:
        delta = (self.lora_b @ self.lora_a).view(
            self.logical_out,
            int(self.base.in_channels),
            int(self.base.kernel_size[0]),
            int(self.base.kernel_size[1]),
        )
        if isinstance(self.base, nn.ConvTranspose2d):
            delta = delta.permute(1, 0, 2, 3).contiguous()
        if self.lora_weight_mask is not None:
            delta = delta * self.lora_weight_mask.to(device=delta.device, dtype=delta.dtype)
        return delta * self.scaling

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        delta = self.delta_weight().to(device=x.device, dtype=x.dtype)
        if isinstance(self.base, nn.ConvTranspose2d):
            return self.base(x) + F.conv_transpose2d(
                x,
                delta,
                bias=None,
                stride=self.base.stride,
                padding=self.base.padding,
                output_padding=self.base.output_padding,
                groups=self.base.groups,
                dilation=self.base.dilation,
            )
        return self.base(x) + F.conv2d(
            x,
            delta,
            bias=None,
            stride=self.base.stride,
            padding=self.base.padding,
            dilation=self.base.dilation,
            groups=self.base.groups,
        )

    @torch.no_grad()
    def merge(self) -> nn.Conv2d | nn.ConvTranspose2d:
        self.base.weight.add_(self.delta_weight().to(device=self.base.weight.device, dtype=self.base.weight.dtype))
        return self.base


class LoRABitFitConv(LoRAConv):
    def __init__(
        self,
        base: nn.Conv2d | nn.ConvTranspose2d,
        *,
        rank: int,
        alpha: float,
        pruned_weight_mask: torch.Tensor | None = None,
    ):
        super().__init__(base, rank=rank, alpha=alpha, pruned_weight_mask=pruned_weight_mask)
        self.bias_delta = nn.Parameter(torch.zeros(base.out_channels, dtype=base.weight.dtype))
        if pruned_weight_mask is None:
            self.register_buffer("bias_allowed_mask", None)
        else:
            if isinstance(base, nn.ConvTranspose2d):
                allowed = (~pruned_weight_mask.bool()).any(dim=(0, 2, 3)).to(dtype=base.weight.dtype)
            else:
                allowed = (~pruned_weight_mask.bool()).any(dim=(1, 2, 3)).to(dtype=base.weight.dtype)
            self.register_buffer("bias_allowed_mask", allowed)

    def effective_bias_delta(self) -> torch.Tensor:
        if self.bias_allowed_mask is None:
            return self.bias_delta
        return self.bias_delta * self.bias_allowed_mask.to(device=self.bias_delta.device, dtype=self.bias_delta.dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return super().forward(x) + self.effective_bias_delta().to(device=x.device, dtype=x.dtype).view(1, -1, 1, 1)

    @torch.no_grad()
    def merge(self) -> nn.Conv2d | nn.ConvTranspose2d:
        merged = super().merge()
        delta = self.effective_bias_delta().to(device=merged.weight.device, dtype=merged.weight.dtype)
        if merged.bias is None:
            merged.bias = nn.Parameter(delta.clone())
        else:
            merged.bias.add_(delta)
        return merged


class BitFitConv(nn.Module):
    def __init__(self, base: nn.Conv2d | nn.ConvTranspose2d, *, pruned_weight_mask: torch.Tensor | None = None):
        super().__init__()
        self.base = base
        self.bias_delta = nn.Parameter(torch.zeros(base.out_channels, dtype=base.weight.dtype))
        self.base.weight.requires_grad_(False)
        if self.base.bias is not None:
            self.base.bias.requires_grad_(False)
        if pruned_weight_mask is None:
            self.register_buffer("bias_allowed_mask", None)
        else:
            if tuple(pruned_weight_mask.shape) != tuple(base.weight.shape):
                raise ValueError("pruned_weight_mask must match base weight shape")
            if isinstance(base, nn.ConvTranspose2d):
                allowed = (~pruned_weight_mask.bool()).any(dim=(0, 2, 3)).to(dtype=base.weight.dtype)
            else:
                allowed = (~pruned_weight_mask.bool()).any(dim=(1, 2, 3)).to(dtype=base.weight.dtype)
            self.register_buffer("bias_allowed_mask", allowed)

    def effective_bias_delta(self) -> torch.Tensor:
        if self.bias_allowed_mask is None:
            return self.bias_delta
        return self.bias_delta * self.bias_allowed_mask.to(device=self.bias_delta.device, dtype=self.bias_delta.dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base(x) + self.effective_bias_delta().to(device=x.device, dtype=x.dtype).view(1, -1, 1, 1)

    @torch.no_grad()
    def merge(self) -> nn.Conv2d | nn.ConvTranspose2d:
        delta = self.effective_bias_delta().to(device=self.base.weight.device, dtype=self.base.weight.dtype)
        if self.base.bias is None:
            self.base.bias = nn.Parameter(delta.clone())
        else:
            self.base.bias.add_(delta)
        return self.base


PEFT_LINEAR_MODULES = (LoRALinear, LoRABitFitLinear, LoHALinear, SAPTLinear, DoRALinear, IA3Linear, BitFitLinear)
PEFT_CONV_MODULES = (LoRAConv, LoRABitFitConv, BitFitConv)
PEFT_MODULES = PEFT_LINEAR_MODULES + PEFT_CONV_MODULES


def parse_csv(value: str) -> tuple[str, ...]:
    return tuple(part.strip() for part in value.split(",") if part.strip())


def mean_abs_margin_delta(row: dict[str, Any]) -> float:
    pair_delta = row.get("pair_delta", {})
    if isinstance(pair_delta, dict) and "mean_abs_margin_delta" in pair_delta:
        return float(pair_delta["mean_abs_margin_delta"])
    return abs(float(row.get("mean_margin_drop", 0.0)))


def selected_candidate_rows(config: CircuitLoRARepairConfig) -> list[dict[str, Any]]:
    summary = json.loads(config.circuit_summary.read_text())
    rows = summary.get("rows_by_accuracy_drop")
    if not isinstance(rows, list):
        raise ValueError("circuit summary is missing rows_by_accuracy_drop")
    kinds = set(config.candidate_kinds)
    candidates = [
        row
        for row in rows
        if isinstance(row, dict)
        and row.get("kind") in kinds
        and isinstance(row.get("module_name"), str)
    ]
    if not candidates:
        raise RuntimeError("no circuit candidates selected")

    if config.selection == "random":
        rng = random.Random(config.seed)
        candidates = list(candidates)
        rng.shuffle(candidates)
    elif config.selection == "safe":
        candidates = sorted(
            candidates,
            key=lambda row: (
                max(float(row.get("correct_drop", 0.0)), 0.0),
                mean_abs_margin_delta(row),
                int(row.get("parameter_estimate", 0)),
                str(row.get("name")),
            ),
        )
    elif config.selection == "stability":
        candidates = sorted(
            candidates,
            key=lambda row: (
                mean_abs_margin_delta(row),
                int(row.get("parameter_estimate", 0)),
                str(row.get("name")),
            ),
        )
    elif config.selection in {"stability_param", "stability-param"}:
        candidates = sorted(
            candidates,
            key=lambda row: (
                mean_abs_margin_delta(row) / max(float(row.get("parameter_estimate", 1.0)), 1.0),
                mean_abs_margin_delta(row),
                int(row.get("parameter_estimate", 0)),
                str(row.get("name")),
            ),
        )
    else:
        candidates = sorted(
            candidates,
            key=lambda row: (
                float(row.get("correct_drop", 0.0)),
                mean_abs_margin_delta(row),
                int(row.get("parameter_estimate", 0)),
                str(row.get("name")),
            ),
        )

    selected: list[dict[str, Any]] = []
    running = 0
    for row in candidates:
        selected.append(row)
        running += int(row.get("parameter_estimate", 0))
        if config.budget_values > 0:
            if running >= config.budget_values:
                break
        elif len(selected) >= config.budget_nodes:
            break
    return selected


def block_index_from_module_name(module_name: str) -> int | None:
    match = re.match(r"^pretrained\.blocks\.(\d+)\.", module_name)
    if match is None:
        return None
    return int(match.group(1))


def transformer_block_indices(model: nn.Module) -> list[int]:
    indices: set[int] = set()
    for name, _module in model.named_modules():
        match = re.match(r"^pretrained\.blocks\.(\d+)(?:\.|$)", name)
        if match is not None:
            indices.add(int(match.group(1)))
    return sorted(indices)


def filtered_transformer_suffixes(module_set: str) -> tuple[str, ...]:
    if module_set == "attn":
        return ("attn.qkv", "attn.proj")
    if module_set == "mlp":
        return ("mlp.fc1", "mlp.fc2")
    return TRANSFORMER_LINEAR_SUFFIXES


def transformer_linear_names_for_blocks(
    model: nn.Module,
    blocks: set[int],
    *,
    module_set: str,
) -> list[str]:
    names: list[str] = []
    suffixes = filtered_transformer_suffixes(module_set)
    for block_index in sorted(blocks):
        for suffix in suffixes:
            name = f"pretrained.blocks.{block_index}.{suffix}"
            try:
                module = model.get_submodule(name)
            except AttributeError:
                continue
            if isinstance(module, nn.Linear):
                names.append(name)
    return names


def depth_head_conv_names(model: nn.Module) -> list[str]:
    names: list[str] = []
    for name, module in model.named_modules():
        if name.startswith("depth_head.") and isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
            names.append(name)
    return sorted(names)


def lora_target_names_for_placement(
    model: nn.Module,
    *,
    masked_names: list[str],
    selected_rows: list[dict[str, Any]],
    placement: str,
    window: int,
    module_set: str,
) -> list[str]:
    if "+" in placement:
        targets: list[str] = []
        for subplacement in (part for part in placement.split("+") if part):
            targets.extend(
                lora_target_names_for_placement(
                    model,
                    masked_names=masked_names,
                    selected_rows=selected_rows,
                    placement=subplacement,
                    window=window,
                    module_set=module_set,
                )
            )
        targets = sorted(set(targets))
        if not targets:
            raise RuntimeError(f"LoRA placement {placement!r} selected no linear modules")
        return targets

    if placement == "masked":
        return sorted(set(masked_names))
    if placement == "head":
        targets = depth_head_conv_names(model)
        if not targets:
            raise RuntimeError("LoRA placement 'head' found no depth_head conv modules")
        return targets
    if placement == "head-masked":
        targets = [
            name
            for name in sorted(set(masked_names))
            if name.startswith("depth_head.")
            and isinstance(model.get_submodule(name), (nn.Conv2d, nn.ConvTranspose2d))
        ]
        if not targets:
            raise RuntimeError("LoRA placement 'head-masked' found no masked depth_head conv modules")
        return targets

    available_blocks = transformer_block_indices(model)
    if not available_blocks:
        raise RuntimeError("could not find transformer block indices")
    min_available = min(available_blocks)
    max_available = max(available_blocks)
    selected_blocks = {
        block
        for row in selected_rows
        if (block := block_index_from_module_name(str(row.get("module_name", "")))) is not None
    }
    if not selected_blocks:
        raise RuntimeError("selected circuits did not map to transformer blocks")

    target_blocks: set[int] = set()
    if placement == "same-block":
        target_blocks = set(selected_blocks)
    elif placement == "previous-block":
        target_blocks = {block - 1 for block in selected_blocks}
    elif placement == "next-block":
        target_blocks = {block + 1 for block in selected_blocks}
    elif placement == "previous-window":
        for block in selected_blocks:
            target_blocks.update(range(block - window, block))
    elif placement == "next-window":
        for block in selected_blocks:
            target_blocks.update(range(block + 1, block + window + 1))
    elif placement == "before-and-masked":
        for block in selected_blocks:
            target_blocks.update(range(block - window, block))
    elif placement == "around-window":
        for block in selected_blocks:
            target_blocks.update(range(block - window, block + window + 1))
    elif placement == "prefix":
        target_blocks = set(range(min_available, min(selected_blocks)))
    elif placement == "all-prior":
        for block in selected_blocks:
            target_blocks.update(range(min_available, block))
    else:
        raise ValueError(f"unknown lora placement: {placement}")

    target_blocks = {block for block in target_blocks if min_available <= block <= max_available}
    targets = transformer_linear_names_for_blocks(model, target_blocks, module_set=module_set)
    if placement == "before-and-masked":
        targets.extend(masked_names)
    targets = sorted(set(targets))
    if not targets:
        raise RuntimeError(f"LoRA placement {placement!r} selected no linear modules")
    return targets


def linear_mask(masked: dict[str, torch.Tensor], name: str, linear: nn.Linear) -> torch.Tensor:
    mask = masked.get(name)
    if mask is None:
        mask = torch.zeros_like(linear.weight.detach(), dtype=torch.bool, device="cpu")
        masked[name] = mask
    return mask


def module_weight_mask(masked: dict[str, torch.Tensor], name: str, module: nn.Module) -> torch.Tensor:
    mask = masked.get(name)
    if mask is None:
        if not hasattr(module, "weight"):
            raise TypeError(f"{name} has no weight")
        mask = torch.zeros_like(module.weight.detach(), dtype=torch.bool, device="cpu")
        masked[name] = mask
    return mask


def zero_linear_rows_(linear: nn.Linear, mask: torch.Tensor, start: int, end: int) -> int:
    with torch.no_grad():
        linear.weight[start:end, :].zero_()
        if linear.bias is not None:
            linear.bias[start:end].zero_()
    mask[start:end, :] = True
    return int((end - start) * linear.weight.shape[1] + (end - start if linear.bias is not None else 0))


def zero_linear_cols_(linear: nn.Linear, mask: torch.Tensor, start: int, end: int) -> int:
    with torch.no_grad():
        linear.weight[:, start:end].zero_()
    mask[:, start:end] = True
    return int(linear.weight.shape[0] * (end - start))


def zero_conv_output_(module: nn.Conv2d | nn.ConvTranspose2d, mask: torch.Tensor, start: int, end: int) -> int:
    with torch.no_grad():
        if isinstance(module, nn.ConvTranspose2d):
            module.weight[:, start:end, :, :].zero_()
            mask[:, start:end, :, :] = True
        else:
            module.weight[start:end, :, :, :].zero_()
            mask[start:end, :, :, :] = True
        if module.bias is not None:
            module.bias[start:end].zero_()
    kernel = int(module.weight.shape[-1] * module.weight.shape[-2])
    if isinstance(module, nn.ConvTranspose2d):
        weight_count = int(module.weight.shape[0] * (end - start) * kernel)
    else:
        weight_count = int((end - start) * module.weight.shape[1] * kernel)
    return weight_count + int(end - start if module.bias is not None else 0)


def zero_conv_input_(module: nn.Conv2d | nn.ConvTranspose2d, mask: torch.Tensor, start: int, end: int) -> int:
    with torch.no_grad():
        if isinstance(module, nn.ConvTranspose2d):
            module.weight[start:end, :, :, :].zero_()
            mask[start:end, :, :, :] = True
        else:
            module.weight[:, start:end, :, :].zero_()
            mask[:, start:end, :, :] = True
    kernel = int(module.weight.shape[-1] * module.weight.shape[-2])
    if isinstance(module, nn.ConvTranspose2d):
        return int((end - start) * module.weight.shape[1] * kernel)
    return int(module.weight.shape[0] * (end - start) * kernel)


def apply_circuit_weight_masks_(
    model: nn.Module,
    rows: list[dict[str, Any]],
) -> tuple[dict[str, torch.Tensor], list[dict[str, Any]]]:
    masks: dict[str, torch.Tensor] = {}
    ops: list[dict[str, Any]] = []
    for row in rows:
        kind = str(row["kind"])
        module_name = str(row["module_name"])
        module = model.get_submodule(module_name)
        start = int(row["start"])
        end = int(row["end"])
        masked_values = 0
        target_names: list[str] = []

        if kind in {"attn_q_head", "attn_k_head", "attn_v_head", "attn_q_group", "attn_k_group", "attn_v_group"}:
            qkv = module.qkv
            if not isinstance(qkv, nn.Linear):
                raise TypeError(f"{module_name}.qkv is not Linear")
            channels = int(qkv.in_features)
            offset = {
                "attn_q_head": 0,
                "attn_q_group": 0,
                "attn_k_head": channels,
                "attn_k_group": channels,
                "attn_v_head": 2 * channels,
                "attn_v_group": 2 * channels,
            }[kind]
            target = module_name + ".qkv"
            masked_values = zero_linear_rows_(qkv, linear_mask(masks, target, qkv), offset + start, offset + end)
            target_names = [target]

        elif kind == "attn_proj_group":
            proj = module.proj
            if not isinstance(proj, nn.Linear):
                raise TypeError(f"{module_name}.proj is not Linear")
            target = module_name + ".proj"
            masked_values = zero_linear_rows_(proj, linear_mask(masks, target, proj), start, end)
            target_names = [target]

        elif kind == "mlp_group":
            fc1 = module.fc1
            fc2 = module.fc2
            if not isinstance(fc1, nn.Linear) or not isinstance(fc2, nn.Linear):
                raise TypeError(f"{module_name}.fc1/fc2 are not Linear")
            fc1_name = module_name + ".fc1"
            fc2_name = module_name + ".fc2"
            masked_values = zero_linear_rows_(fc1, linear_mask(masks, fc1_name, fc1), start, end)
            masked_values += zero_linear_cols_(fc2, linear_mask(masks, fc2_name, fc2), start, end)
            target_names = [fc1_name, fc2_name]

        elif kind == "head_channel_group":
            if not isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
                raise TypeError(f"{module_name} is not Conv2d/ConvTranspose2d")
            masked_values = zero_conv_output_(module, module_weight_mask(masks, module_name, module), start, end)
            target_names = [module_name]

        elif kind == "head_input_channel_group":
            if not isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
                raise TypeError(f"{module_name} is not Conv2d/ConvTranspose2d")
            masked_values = zero_conv_input_(module, module_weight_mask(masks, module_name, module), start, end)
            target_names = [module_name]

        else:
            raise ValueError(f"unsupported circuit kind for LoRA repair: {kind}")

        ops.append(
            {
                "name": row["name"],
                "kind": kind,
                "module_name": module_name,
                "start": start,
                "end": end,
                "target_names": target_names,
                "parameter_estimate": int(row.get("parameter_estimate", 0)),
                "masked_tensor_values": masked_values,
                "correct_drop": row.get("correct_drop"),
                "mean_abs_margin_delta": mean_abs_margin_delta(row),
            }
        )
    return masks, ops


def reapply_masks_(model: nn.Module, masks: dict[str, torch.Tensor]) -> None:
    with torch.no_grad():
        for name, mask in masks.items():
            module = model.get_submodule(name)
            if not isinstance(module, (nn.Linear, nn.Conv2d, nn.ConvTranspose2d)):
                raise TypeError(f"{name} is not Linear/Conv after merge")
            module.weight.masked_fill_(mask.to(module.weight.device), 0)


@torch.no_grad()
def max_masked_weight_abs(model: nn.Module, masks: dict[str, torch.Tensor]) -> float:
    max_abs = 0.0
    for name, mask in masks.items():
        module = model.get_submodule(name)
        if not isinstance(module, (nn.Linear, nn.Conv2d, nn.ConvTranspose2d)):
            raise TypeError(f"{name} is not Linear/Conv")
        pruned = module.weight.detach()[mask.to(module.weight.device)]
        if pruned.numel():
            max_abs = max(max_abs, float(pruned.float().abs().max().item()))
    return max_abs


def parent_and_attr(model: nn.Module, module_name: str) -> tuple[nn.Module, str]:
    parts = module_name.split(".")
    parent = model.get_submodule(".".join(parts[:-1])) if len(parts) > 1 else model
    return parent, parts[-1]


def add_peft_modules_(
    model: nn.Module,
    module_names: list[str],
    *,
    method: str,
    rank: int,
    alpha: float,
    masks: dict[str, torch.Tensor] | None = None,
) -> dict[str, int]:
    stats: dict[str, int] = {}
    if method not in PEFT_METHODS:
        raise ValueError(f"unknown PEFT method: {method}")
    for name in sorted(set(module_names)):
        parent, attr = parent_and_attr(model, name)
        module = getattr(parent, attr)
        pruned_weight_mask = None if masks is None else masks.get(name)
        if isinstance(module, nn.Linear):
            if method == "lora":
                wrapper = LoRALinear(module, rank=rank, alpha=alpha, pruned_weight_mask=pruned_weight_mask)
            elif method == "lora-bitfit":
                wrapper = LoRABitFitLinear(module, rank=rank, alpha=alpha, pruned_weight_mask=pruned_weight_mask)
            elif method == "dora":
                wrapper = DoRALinear(module, rank=rank, alpha=alpha, pruned_weight_mask=pruned_weight_mask)
            elif method == "loha":
                wrapper = LoHALinear(module, rank=rank, alpha=alpha, pruned_weight_mask=pruned_weight_mask)
            elif method == "sapt":
                wrapper = SAPTLinear(module, rank=rank, alpha=alpha, pruned_weight_mask=pruned_weight_mask)
            elif method == "ia3-out":
                wrapper = IA3Linear(module, mode="out")
            elif method == "ia3-in":
                wrapper = IA3Linear(module, mode="in")
            elif method == "bitfit":
                wrapper = BitFitLinear(module, pruned_weight_mask=pruned_weight_mask)
            else:
                raise AssertionError(method)
        elif isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
            if method == "lora":
                wrapper = LoRAConv(module, rank=rank, alpha=alpha, pruned_weight_mask=pruned_weight_mask)
            elif method == "lora-bitfit":
                wrapper = LoRABitFitConv(module, rank=rank, alpha=alpha, pruned_weight_mask=pruned_weight_mask)
            elif method == "bitfit":
                wrapper = BitFitConv(module, pruned_weight_mask=pruned_weight_mask)
            else:
                raise TypeError(f"{method} is only implemented for Linear modules, not conv module {name}")
        else:
            raise TypeError(f"{name} is not Linear/Conv")
        wrapper.to(device=module.weight.device, dtype=module.weight.dtype)
        setattr(parent, attr, wrapper)
        stats[name] = sum(
            int(param.numel())
            for param_name, param in wrapper.named_parameters()
            if not param_name.startswith("base.")
        )
    return stats


def peft_pruned_weight_mask(module: nn.Module) -> torch.Tensor | None:
    for attr in ("lora_weight_mask", "loha_weight_mask", "sapt_weight_mask", "dora_weight_mask"):
        allowed_mask = getattr(module, attr, None)
        if isinstance(allowed_mask, torch.Tensor):
            return ~allowed_mask.bool()
    return None


def merge_peft_modules_with_safety_(model: nn.Module) -> tuple[dict[str, float], dict[str, dict[str, Any]]]:
    merge_stats: dict[str, float] = {}
    safety_stats: dict[str, dict[str, Any]] = {}
    for name, module in list(model.named_modules()):
        if not isinstance(module, PEFT_MODULES):
            continue
        parent, attr = parent_and_attr(model, name)
        before = module.base.weight.detach().float().clone()
        pruned_mask = peft_pruned_weight_mask(module)
        if pruned_mask is not None:
            pruned_mask = pruned_mask.to(device=before.device)
        merged = module.merge()
        setattr(parent, attr, merged)
        after = merged.weight.detach().float()
        delta = after - before
        merge_stats[name] = float(delta.pow(2).mean().sqrt().item())
        if pruned_mask is not None:
            pruned_count = int(pruned_mask.sum().item())
            if pruned_count:
                pruned_delta = delta[pruned_mask]
                pruned_after = after[pruned_mask]
                safety_stats[name] = {
                    "pruned_values": pruned_count,
                    "pruned_delta_abs_max": float(pruned_delta.abs().max().item()),
                    "pruned_delta_rms": float(pruned_delta.pow(2).mean().sqrt().item()),
                    "pruned_after_abs_max": float(pruned_after.abs().max().item()),
                    "safe": bool(pruned_delta.abs().max().item() <= 1e-12 and pruned_after.abs().max().item() <= 1e-12),
                }
            else:
                safety_stats[name] = {
                    "pruned_values": 0,
                    "pruned_delta_abs_max": 0.0,
                    "pruned_delta_rms": 0.0,
                    "pruned_after_abs_max": 0.0,
                    "safe": True,
                }
    return merge_stats, safety_stats


def merge_peft_modules_(model: nn.Module) -> dict[str, float]:
    merge_stats, _ = merge_peft_modules_with_safety_(model)
    return merge_stats


def freeze_except_peft_(model: nn.Module) -> None:
    for param in model.parameters():
        param.requires_grad_(False)
    for module in model.modules():
        if isinstance(module, PEFT_MODULES):
            for param_name, param in module.named_parameters():
                if not param_name.startswith("base."):
                    param.requires_grad_(True)


def trainable_params(model: nn.Module) -> int:
    return sum(int(p.numel()) for p in model.parameters() if p.requires_grad)


def matrix_sign(update: torch.Tensor) -> torch.Tensor:
    if update.ndim != 2:
        raise ValueError("matrix_sign expects a 2D tensor")
    if not torch.isfinite(update).all():
        update = torch.nan_to_num(update)
    u, _s, vh = torch.linalg.svd(update.float(), full_matrices=False)
    return (u @ vh).to(dtype=update.dtype)


def inverse_sqrt_spd(matrix: torch.Tensor, damping: float) -> torch.Tensor:
    eye = torch.eye(matrix.shape[0], device=matrix.device, dtype=torch.float32)
    eigvals, eigvecs = torch.linalg.eigh(matrix.float() + float(damping) * eye)
    inv_sqrt = eigvals.clamp_min(float(damping)).rsqrt()
    return (eigvecs * inv_sqrt.unsqueeze(0)) @ eigvecs.T


def peft_factor_pairs(model: nn.Module) -> list[tuple[str, nn.Parameter, nn.Parameter]]:
    pairs: list[tuple[str, nn.Parameter, nn.Parameter]] = []
    for name, module in model.named_modules():
        if isinstance(module, (LoRALinear, LoRABitFitLinear, DoRALinear)):
            pairs.append((f"{name}.lora", module.lora_b.weight, module.lora_a.weight))
        elif isinstance(module, LoHALinear):
            pairs.append((f"{name}.hada1", module.hada_b1.weight, module.hada_a1.weight))
            pairs.append((f"{name}.hada2", module.hada_b2.weight, module.hada_a2.weight))
        elif isinstance(module, SAPTLinear):
            pairs.append((f"{name}.sapt_lora", module.lora_b, module.lora_a))
            pairs.append((f"{name}.sapt_hada1", module.hada_b1, module.hada_a1))
            pairs.append((f"{name}.sapt_hada2", module.hada_b2, module.hada_a2))
        elif isinstance(module, (LoRAConv, LoRABitFitConv)):
            pairs.append((f"{name}.lora", module.lora_b, module.lora_a))
    return [(name, b, a) for name, b, a in pairs if b.requires_grad and a.requires_grad]


class PEFTMuonOptimizer:
    def __init__(
        self,
        model: nn.Module,
        *,
        mode: str,
        lr: float,
        weight_decay: float,
        momentum: float,
        damping: float,
    ):
        if mode not in {"muon", "comp-muon"}:
            raise ValueError("mode must be muon or comp-muon")
        self.mode = mode
        self.lr = float(lr)
        self.weight_decay = float(weight_decay)
        self.momentum = float(momentum)
        self.damping = float(damping)
        self.params = [param for param in model.parameters() if param.requires_grad]
        self.factor_pairs = peft_factor_pairs(model)
        paired_ids = {id(param) for _name, b, a in self.factor_pairs for param in (b, a)}
        self.muon_params = [
            param
            for param in self.params
            if param.ndim == 2 and (self.mode == "muon" or id(param) not in paired_ids)
        ]
        muon_ids = {id(param) for param in self.muon_params}
        adam_params = [
            param
            for param in self.params
            if id(param) not in paired_ids and id(param) not in muon_ids
        ]
        self.adam = torch.optim.AdamW(adam_params, lr=lr, weight_decay=weight_decay) if adam_params else None
        self.momentum_buffers: dict[int, torch.Tensor] = {}

    def zero_grad(self, *, set_to_none: bool = True) -> None:
        for param in self.params:
            param.grad = None if set_to_none else torch.zeros_like(param)

    def momentum_buffer(self, param: nn.Parameter) -> torch.Tensor:
        key = id(param)
        if key not in self.momentum_buffers:
            self.momentum_buffers[key] = torch.zeros_like(param, dtype=torch.float32)
        return self.momentum_buffers[key]

    def update_momentum(self, param: nn.Parameter) -> torch.Tensor | None:
        if param.grad is None:
            return None
        grad = param.grad.detach().float()
        buf = self.momentum_buffer(param)
        buf.mul_(self.momentum).add_(grad)
        return buf

    @torch.no_grad()
    def step(self) -> None:
        if self.adam is not None:
            self.adam.step()

        updates: list[tuple[nn.Parameter, torch.Tensor]] = []
        if self.mode == "comp-muon":
            for _name, b_param, a_param in self.factor_pairs:
                b_momentum = self.update_momentum(b_param)
                a_momentum = self.update_momentum(a_param)
                if b_momentum is None or a_momentum is None:
                    continue
                a_weight = a_param.detach().float()
                b_weight = b_param.detach().float()
                a_inv_sqrt = inverse_sqrt_spd(a_weight @ a_weight.T, self.damping)
                b_inv_sqrt = inverse_sqrt_spd(b_weight.T @ b_weight, self.damping)
                b_update = matrix_sign(b_momentum @ a_inv_sqrt) @ a_inv_sqrt
                a_update = b_inv_sqrt @ matrix_sign(b_inv_sqrt @ a_momentum)
                updates.append((b_param, 0.5 * b_update))
                updates.append((a_param, 0.5 * a_update))

        for param in self.muon_params:
            momentum = self.update_momentum(param)
            if momentum is not None:
                updates.append((param, matrix_sign(momentum)))

        for param, update in updates:
            if self.weight_decay:
                param.mul_(1.0 - self.lr * self.weight_decay)
            param.add_(update.to(device=param.device, dtype=param.dtype), alpha=-self.lr)


def load_train_samples(
    model: nn.Module,
    dataset_root: Path,
    items: list[tuple[str, list[dict[str, Any]]]],
    *,
    input_size: int,
    device: torch.device,
) -> list[TrainSample]:
    samples: list[TrainSample] = []
    for relative_path, pairs in items:
        image = cv2.imread(str(dataset_root / relative_path))
        if image is None:
            continue
        tensor, (height, width) = model.image2tensor(image, input_size)
        point1_rows = []
        point1_cols = []
        point2_rows = []
        point2_cols = []
        for pair in pairs:
            if pair.get("closer_point") != "point1":
                raise ValueError(f"unsupported closer_point in {relative_path}: {pair}")
            point1_rows.append(max(0, min(int(pair["point1"][0]), height - 1)))
            point1_cols.append(max(0, min(int(pair["point1"][1]), width - 1)))
            point2_rows.append(max(0, min(int(pair["point2"][0]), height - 1)))
            point2_cols.append(max(0, min(int(pair["point2"][1]), width - 1)))
        samples.append(
            TrainSample(
                relative_path=relative_path,
                tensor=tensor.to(device=device, non_blocking=True),
                height=height,
                width=width,
                point1_rows=torch.tensor(point1_rows, dtype=torch.long),
                point1_cols=torch.tensor(point1_cols, dtype=torch.long),
                point2_rows=torch.tensor(point2_rows, dtype=torch.long),
                point2_cols=torch.tensor(point2_cols, dtype=torch.long),
            )
        )
    if not samples:
        raise RuntimeError("no images could be loaded")
    return samples


@torch.no_grad()
def cache_teacher_targets(
    teacher: nn.Module,
    samples: list[TrainSample],
    device: torch.device,
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    teacher.eval()
    outputs: list[torch.Tensor] = []
    pair_margins: list[torch.Tensor] = []
    for sample in tqdm(samples, desc="cache teacher", unit="image"):
        output = teacher(sample.tensor.to(device=device, non_blocking=True)).detach().float()
        outputs.append(output.cpu())
        pair_margins.append(
            pair_margins_from_depth(
                output,
                sample,
                device=device,
                normalize=True,
            )
            .detach()
            .cpu()
        )
    return outputs, pair_margins


def normalized_depth(depth: torch.Tensor) -> torch.Tensor:
    flat = depth.flatten(1)
    mean = flat.mean(dim=1).view(-1, 1, 1)
    std = flat.std(dim=1).clamp_min(1e-6).view(-1, 1, 1)
    return (depth - mean) / std


def normalize_depth_2d(depth: torch.Tensor) -> torch.Tensor:
    return (depth - depth.mean()) / depth.std().clamp_min(1e-6)


def pair_margins_from_depth(
    depth: torch.Tensor,
    sample: TrainSample,
    *,
    device: torch.device,
    normalize: bool,
) -> torch.Tensor:
    if sample.point1_rows.numel() == 0:
        return depth.sum().new_zeros(0)
    if depth.ndim == 3:
        depth_2d = F.interpolate(depth[:, None], (sample.height, sample.width), mode="bilinear", align_corners=True)[0, 0]
    elif depth.ndim == 2:
        depth_2d = depth
    else:
        raise ValueError(f"unsupported depth shape: {tuple(depth.shape)}")
    if normalize:
        depth_2d = normalize_depth_2d(depth_2d.float())
    rows1 = sample.point1_rows.to(device=device, non_blocking=True)
    cols1 = sample.point1_cols.to(device=device, non_blocking=True)
    rows2 = sample.point2_rows.to(device=device, non_blocking=True)
    cols2 = sample.point2_cols.to(device=device, non_blocking=True)
    return depth_2d[rows1, cols1] - depth_2d[rows2, cols2]


def smooth_depth_distill_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    pred_norm = normalized_depth(pred.float())
    target_norm = normalized_depth(target.float().to(pred.device))
    value = F.smooth_l1_loss(pred_norm, target_norm)
    pred_dx = pred_norm[:, :, 1:] - pred_norm[:, :, :-1]
    target_dx = target_norm[:, :, 1:] - target_norm[:, :, :-1]
    pred_dy = pred_norm[:, 1:, :] - pred_norm[:, :-1, :]
    target_dy = target_norm[:, 1:, :] - target_norm[:, :-1, :]
    grad = F.smooth_l1_loss(pred_dx, target_dx) + F.smooth_l1_loss(pred_dy, target_dy)
    return value + 0.25 * grad


def l2_depth_distill_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    pred_norm = normalized_depth(pred.float())
    target_norm = normalized_depth(target.float().to(pred.device))
    value = F.mse_loss(pred_norm, target_norm)
    pred_dx = pred_norm[:, :, 1:] - pred_norm[:, :, :-1]
    target_dx = target_norm[:, :, 1:] - target_norm[:, :, :-1]
    pred_dy = pred_norm[:, 1:, :] - pred_norm[:, :-1, :]
    target_dy = target_norm[:, 1:, :] - target_norm[:, :-1, :]
    grad = F.mse_loss(pred_dx, target_dx) + F.mse_loss(pred_dy, target_dy)
    return value + 0.25 * grad


def kl_depth_distill_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    *,
    direction: str,
    temperature: float,
) -> torch.Tensor:
    pred_logits = normalized_depth(pred.float()).flatten(1) / temperature
    target_logits = normalized_depth(target.float().to(pred.device)).flatten(1) / temperature
    student_log_prob = F.log_softmax(pred_logits, dim=1)
    teacher_log_prob = F.log_softmax(target_logits, dim=1)
    student_prob = student_log_prob.exp()
    teacher_prob = teacher_log_prob.exp()
    if direction == "forward":
        # KL(teacher || student)
        loss = (teacher_prob * (teacher_log_prob - student_log_prob)).sum(dim=1).mean()
    elif direction == "reverse":
        # KL(student || teacher)
        loss = (student_prob * (student_log_prob - teacher_log_prob)).sum(dim=1).mean()
    else:
        raise ValueError("direction must be forward or reverse")
    return loss * (temperature * temperature)


def depth_distill_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    *,
    mode: str,
    kl_direction: str,
    kl_temperature: float,
) -> torch.Tensor:
    if mode == "smoothl1":
        return smooth_depth_distill_loss(pred, target)
    if mode == "l2":
        return l2_depth_distill_loss(pred, target)
    if mode == "kl":
        return kl_depth_distill_loss(pred, target, direction=kl_direction, temperature=kl_temperature)
    raise ValueError("mode must be smoothl1, l2, or kl")


def pairwise_distill_loss(
    pred: torch.Tensor,
    sample: TrainSample,
    teacher_margins: torch.Tensor,
    *,
    device: torch.device,
    teacher_weight: float,
    label_weight: float,
    tau: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    pred_margins = pair_margins_from_depth(pred, sample, device=device, normalize=True)
    if pred_margins.numel() == 0:
        zero = pred.sum() * 0.0
        return zero, {"pair_teacher_loss": 0.0, "pair_label_loss": 0.0, "pair_accuracy": 0.0}

    total = pred_margins.sum() * 0.0
    teacher_loss_value = pred_margins.sum() * 0.0
    label_loss_value = pred_margins.sum() * 0.0
    if teacher_weight > 0:
        target_margins = teacher_margins.to(device=device, non_blocking=True)
        teacher_loss_value = F.smooth_l1_loss(pred_margins, target_margins)
        total = total + teacher_weight * teacher_loss_value
    if label_weight > 0:
        label_loss_value = F.softplus(-pred_margins / tau).mean()
        total = total + label_weight * label_loss_value
    pair_accuracy = float((pred_margins.detach() > 0).float().mean().cpu().item())
    return total, {
        "pair_teacher_loss": float(teacher_loss_value.detach().cpu().item()),
        "pair_label_loss": float(label_loss_value.detach().cpu().item()),
        "pair_accuracy": pair_accuracy,
    }


def train_peft_repair(
    student: nn.Module,
    samples: list[TrainSample],
    teacher_outputs: list[torch.Tensor],
    teacher_pair_margins: list[torch.Tensor],
    *,
    device: torch.device,
    epochs: int,
    lr: float,
    weight_decay: float,
    optimizer: str,
    muon_momentum: float,
    muon_damping: float,
    pairwise_weight: float,
    pairwise_teacher_weight: float,
    pairwise_label_weight: float,
    pairwise_tau: float,
    log_every: int,
    depth_loss_mode: str = "smoothl1",
    kl_direction: str = "forward",
    kl_temperature: float = 1.0,
) -> list[dict[str, Any]]:
    params = [p for p in student.parameters() if p.requires_grad]
    if not params:
        raise RuntimeError("no trainable PEFT parameters")
    if optimizer == "adamw":
        opt = torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    else:
        opt = PEFTMuonOptimizer(
            student,
            mode=optimizer,
            lr=lr,
            weight_decay=weight_decay,
            momentum=muon_momentum,
            damping=muon_damping,
        )
    history: list[dict[str, Any]] = []
    step = 0
    for epoch in range(1, epochs + 1):
        order = list(range(len(samples)))
        random.shuffle(order)
        losses: list[float] = []
        depth_losses: list[float] = []
        pair_losses: list[float] = []
        pair_accs: list[float] = []
        student.train()
        for local_index, item_index in enumerate(order, start=1):
            step += 1
            sample = samples[item_index]
            x = sample.tensor.to(device=device, non_blocking=True)
            target = teacher_outputs[item_index].to(device=device, non_blocking=True)
            pred = student(x)
            depth_loss = depth_distill_loss(
                pred,
                target,
                mode=depth_loss_mode,
                kl_direction=kl_direction,
                kl_temperature=kl_temperature,
            )
            pair_loss = pred.sum() * 0.0
            pair_stats = {"pair_teacher_loss": 0.0, "pair_label_loss": 0.0, "pair_accuracy": 0.0}
            if pairwise_weight > 0 and (pairwise_teacher_weight > 0 or pairwise_label_weight > 0):
                pair_loss, pair_stats = pairwise_distill_loss(
                    pred,
                    sample,
                    teacher_pair_margins[item_index],
                    device=device,
                    teacher_weight=pairwise_teacher_weight,
                    label_weight=pairwise_label_weight,
                    tau=pairwise_tau,
                )
            loss = depth_loss + pairwise_weight * pair_loss
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            losses.append(float(loss.detach().cpu().item()))
            depth_losses.append(float(depth_loss.detach().cpu().item()))
            pair_losses.append(float(pair_loss.detach().cpu().item()))
            pair_accs.append(pair_stats["pair_accuracy"])
            if log_every > 0 and step % log_every == 0:
                print(
                    json.dumps(
                        {
                            "step": step,
                            "epoch": epoch,
                            "loss": losses[-1],
                            "depth_loss": depth_losses[-1],
                            "pair_loss": pair_losses[-1],
                            **pair_stats,
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
        row = {
            "epoch": epoch,
            "mean_loss": float(np.mean(losses)),
            "mean_depth_loss": float(np.mean(depth_losses)),
            "mean_pair_loss": float(np.mean(pair_losses)),
            "mean_pair_accuracy": float(np.mean(pair_accs)),
            "last_loss": losses[-1],
        }
        history.append(row)
        print(json.dumps(row, sort_keys=True), flush=True)
    student.eval()
    return history


@torch.no_grad()
def evaluate_items(
    model: nn.Module,
    items: list[tuple[str, list[dict[str, Any]]]],
    *,
    dataset_root: Path,
    input_size: int,
    device: torch.device,
    label: str,
) -> dict[str, Any]:
    counts = empty_counts()
    by_scene = defaultdict(empty_counts)
    missing_images: list[str] = []
    model.eval()
    for relative_path, pairs in tqdm(items, desc=label, unit="image"):
        image = cv2.imread(str(dataset_root / relative_path))
        if image is None:
            missing_images.append(str(dataset_root / relative_path))
            continue
        tensor, (height, width) = model.image2tensor(image, input_size)
        depth = model(tensor.to(device=device, non_blocking=True))
        depth = F.interpolate(depth[:, None], (height, width), mode="bilinear", align_corners=True)[0, 0].detach().float().cpu()
        scene = scene_from_path(relative_path)
        for pair in pairs:
            d1 = point_value(depth, pair["point1"])
            d2 = point_value(depth, pair["point2"])
            add_pair(counts, d1, d2)
            add_pair(by_scene[scene], d1, d2)
    return {
        "label": label,
        "overall": finalize_counts(counts),
        "by_scene": {scene: finalize_counts(scene_counts) for scene, scene_counts in sorted(by_scene.items())},
        "missing_images": missing_images,
        "evaluated_images": len(items) - len(missing_images),
    }


def write_report(path: Path, summary: dict[str, Any]) -> None:
    lines = ["# Circuit PEFT Repair DA2K", ""]
    lines.append("## Results")
    lines.append("")
    lines.append("| model | accuracy | correct | pairs |")
    lines.append("| --- | ---: | ---: | ---: |")
    for row in summary["results"]:
        overall = row["overall"]
        lines.append(
            f"| `{row['label']}` | {overall['larger_is_closer_accuracy']:.4f} | {overall['larger_correct']} | {overall['pairs']} |"
        )
    lines.append("")
    lines.append("## Selected Circuits")
    lines.append("")
    lines.append("| name | kind | module | range | correct_drop | abs_margin_delta | params |")
    lines.append("| --- | --- | --- | ---: | ---: | ---: | ---: |")
    for op in summary["mask_operations"]:
        lines.append(
            f"| `{op['name']}` | `{op['kind']}` | `{op['module_name']}` | {op['start']}:{op['end']} | "
            f"{op['correct_drop']} | {op['mean_abs_margin_delta']:.4f} | {op['parameter_estimate']} |"
        )
    lines.append("")
    lines.append("## Repair")
    lines.append("")
    lines.append(f"- PEFT method: `{summary['config']['peft_method']}`")
    lines.append(f"- Optimizer: `{summary['config']['optimizer']}`")
    if summary["config"]["optimizer"] in {"muon", "comp-muon"}:
        lines.append(
            f"- Muon momentum/damping: `{summary['config']['muon_momentum']}` / "
            f"`{summary['config']['muon_damping']}`"
        )
    lines.append(f"- Depth loss: `{summary['config']['depth_loss_mode']}`")
    if summary["config"]["depth_loss_mode"] == "kl":
        lines.append(
            f"- KL depth distillation: `{summary['config']['kl_direction']}` "
            f"(temperature `{summary['config']['kl_temperature']}`)"
        )
    lines.append(f"- Train/eval overlap images: `{summary['train_eval_overlap_count']}`")
    lines.append(f"- Masked tensor values: `{summary['masked_tensor_values']}`")
    lines.append(f"- PEFT trainable params: `{summary['peft_trainable_params']}`")
    lines.append(f"- PEFT modules: `{len(summary['peft_modules'])}`")
    lines.append(f"- Merge RMS deltas: `{summary['merge_rms_delta_by_module']}`")
    lines.append(f"- Merge mask safety all safe: `{summary['merge_mask_safety_all_safe']}`")
    lines.append(
        f"- Folded masked weight max before/after remask: "
        f"`{summary['folded_masked_weight_abs_max_before_remask']}` / "
        f"`{summary['folded_masked_weight_abs_max_after_remask']}`"
    )
    lines.append(f"- Folded unmasked/remasked overall match: `{summary['folded_unmasked_remasked_overall_match']}`")
    path.write_text("\n".join(lines) + "\n")


def run(config: CircuitLoRARepairConfig) -> dict[str, Any]:
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")

    started = time.monotonic()
    device = resolve_device(config.device)
    config.output_dir.mkdir(parents=True, exist_ok=True)
    (config.output_dir / "config.json").write_text(json.dumps(asdict(config), indent=2, sort_keys=True, default=str) + "\n")

    all_items = selected_annotations(config.dataset_root, scene_type=config.scene_type, max_images=0)
    train_items = all_items[: config.train_images]
    eval_end = None if config.eval_images == 0 else config.eval_skip_images + config.eval_images
    eval_items = all_items[config.eval_skip_images : eval_end]
    if not train_items or not eval_items:
        raise RuntimeError("empty train/eval item selection")
    train_paths = {item[0] for item in train_items}
    eval_paths = {item[0] for item in eval_items}
    overlap_paths = sorted(train_paths & eval_paths)
    if overlap_paths and not config.allow_train_eval_overlap:
        raise RuntimeError("train/eval image overlap detected")

    selected_rows = selected_candidate_rows(config)
    teacher = load_model(config.encoder, config.checkpoint, device)
    for param in teacher.parameters():
        param.requires_grad_(False)
    teacher.eval()

    student = load_model(config.encoder, config.checkpoint, device)
    masks, mask_operations = apply_circuit_weight_masks_(student, selected_rows)
    peft_targets = lora_target_names_for_placement(
        student,
        masked_names=sorted(masks),
        selected_rows=selected_rows,
        placement=config.lora_placement,
        window=config.lora_window,
        module_set=config.lora_module_set,
    )
    add_stats = add_peft_modules_(
        student,
        peft_targets,
        method=config.peft_method,
        rank=config.lora_rank,
        alpha=config.lora_alpha,
        masks=masks,
    )
    freeze_except_peft_(student)
    peft_param_count = trainable_params(student)

    train_samples = load_train_samples(
        teacher,
        config.dataset_root,
        train_items,
        input_size=config.input_size,
        device=device,
    )
    teacher_outputs, teacher_pair_margins = cache_teacher_targets(teacher, train_samples, device)

    results: list[dict[str, Any]] = []
    results.append(
        evaluate_items(
            teacher,
            eval_items,
            dataset_root=config.dataset_root,
            input_size=config.input_size,
            device=device,
            label="dense_teacher",
        )
    )

    pruned_eval_model = load_model(config.encoder, config.checkpoint, device)
    apply_circuit_weight_masks_(pruned_eval_model, selected_rows)
    results.append(
        evaluate_items(
            pruned_eval_model,
            eval_items,
            dataset_root=config.dataset_root,
            input_size=config.input_size,
            device=device,
            label="pruned_student",
        )
    )
    del pruned_eval_model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    history = train_peft_repair(
        student,
        train_samples,
        teacher_outputs,
        teacher_pair_margins,
        device=device,
        epochs=config.epochs,
        lr=config.lr,
        weight_decay=config.weight_decay,
        optimizer=config.optimizer,
        muon_momentum=config.muon_momentum,
        muon_damping=config.muon_damping,
        pairwise_weight=config.pairwise_weight,
        pairwise_teacher_weight=config.pairwise_teacher_weight,
        pairwise_label_weight=config.pairwise_label_weight,
        pairwise_tau=config.pairwise_tau,
        log_every=config.log_every,
        depth_loss_mode=config.depth_loss_mode,
        kl_direction=config.kl_direction,
        kl_temperature=config.kl_temperature,
    )
    results.append(
        evaluate_items(
            student,
            eval_items,
            dataset_root=config.dataset_root,
            input_size=config.input_size,
            device=device,
            label="peft_repaired_unmerged",
        )
    )

    merge_stats, merge_mask_safety = merge_peft_modules_with_safety_(student)
    results.append(
        evaluate_items(
            student,
            eval_items,
            dataset_root=config.dataset_root,
            input_size=config.input_size,
            device=device,
            label="folded_peft_unmasked",
        )
    )

    folded_masked_weight_abs_max_before_remask = max_masked_weight_abs(student, masks)
    reapply_masks_(student, masks)
    folded_masked_weight_abs_max_after_remask = max_masked_weight_abs(student, masks)
    results.append(
        evaluate_items(
            student,
            eval_items,
            dataset_root=config.dataset_root,
            input_size=config.input_size,
            device=device,
            label="folded_peft_remasked",
        )
    )

    checkpoint_path = None
    if config.save_checkpoint:
        checkpoint_path = config.output_dir / "folded_peft_remasked.pth"
        torch.save(student.state_dict(), checkpoint_path)

    summary = {
        "config": asdict(config),
        "device": str(device),
        "train_items": [item[0] for item in train_items],
        "eval_items": [item[0] for item in eval_items],
        "train_eval_overlap_count": len(overlap_paths),
        "train_eval_overlap_items": overlap_paths,
        "eval_image_count": len(eval_items),
        "selected_circuits": selected_rows,
        "mask_operations": mask_operations,
        "masked_tensor_values": int(sum(int(op["masked_tensor_values"]) for op in mask_operations)),
        "mask_linear_modules": sorted(masks),
        "peft_modules": add_stats,
        "peft_trainable_params": peft_param_count,
        # Backward-compatible aliases for older result parsers.
        "lora_modules": add_stats,
        "lora_trainable_params": peft_param_count,
        "history": history,
        "results": results,
        "merge_rms_delta_by_module": merge_stats,
        "merge_mask_safety_by_module": merge_mask_safety,
        "merge_mask_safety_all_safe": all(row["safe"] for row in merge_mask_safety.values()),
        "folded_masked_weight_abs_max_before_remask": folded_masked_weight_abs_max_before_remask,
        "folded_masked_weight_abs_max_after_remask": folded_masked_weight_abs_max_after_remask,
        "folded_unmasked_remasked_overall_match": results[-2]["overall"] == results[-1]["overall"],
        "checkpoint_path": str(checkpoint_path) if checkpoint_path else None,
        "elapsed_seconds": time.monotonic() - started,
    }
    (config.output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True, default=str) + "\n")
    write_report(config.output_dir / "summary.md", summary)
    print(
        json.dumps(
            {
                "output_dir": str(config.output_dir),
                "elapsed_seconds": summary["elapsed_seconds"],
                "results": [
                    {
                        "label": row["label"],
                        "accuracy": row["overall"]["larger_is_closer_accuracy"],
                        "correct": row["overall"]["larger_correct"],
                        "pairs": row["overall"]["pairs"],
                    }
                    for row in results
                ],
                "masked_tensor_values": summary["masked_tensor_values"],
                "merge_mask_safety_all_safe": summary["merge_mask_safety_all_safe"],
                "folded_masked_weight_abs_max_before_remask": summary["folded_masked_weight_abs_max_before_remask"],
                "folded_masked_weight_abs_max_after_remask": summary["folded_masked_weight_abs_max_after_remask"],
                "folded_unmasked_remasked_overall_match": summary["folded_unmasked_remasked_overall_match"],
                "train_eval_overlap_count": summary["train_eval_overlap_count"],
                "peft_trainable_params": peft_param_count,
                "lora_trainable_params": peft_param_count,
            },
            indent=2,
            sort_keys=True,
        ),
        flush=True,
    )
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prune DAV2 circuits, repair with foldable PEFT distillation, fold, and evaluate.")
    parser.add_argument("--dataset-root", type=Path, default=Path("/home/ubuntu/vision_token_tests/datasets/DA-2K/DA-2K"))
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/depth_anything_v2_vits.pth"))
    parser.add_argument("--circuit-summary", type=Path, default=Path("/home/ubuntu/remote-work/depth-anything-2/eval_outputs/subcircuit_fine_all_zero_32_g32_h16/summary.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("eval_outputs/circuit_lora_repair"))
    parser.add_argument("--encoder", choices=sorted(MODEL_CONFIGS), default="vits")
    parser.add_argument("--input-size", type=int, default=518)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--scene-type", default="")
    parser.add_argument("--train-images", type=int, default=24)
    parser.add_argument("--eval-skip-images", type=int, default=24)
    parser.add_argument("--eval-images", type=int, default=64)
    parser.add_argument("--allow-train-eval-overlap", action="store_true")
    parser.add_argument("--budget-nodes", type=int, default=8)
    parser.add_argument("--budget-values", type=int, default=0)
    parser.add_argument("--candidate-kinds", default=",".join(TRANSFORMER_CIRCUIT_KINDS))
    parser.add_argument("--selection", choices=("weak", "safe", "random", "stability", "stability_param", "stability-param"), default="weak")
    parser.add_argument("--peft-method", choices=PEFT_METHODS, default="lora")
    parser.add_argument("--lora-placement", default="masked")
    parser.add_argument("--lora-window", type=int, default=1)
    parser.add_argument("--lora-module-set", choices=("all", "attn", "mlp"), default="all")
    parser.add_argument("--lora-rank", type=int, default=8)
    parser.add_argument("--lora-alpha", type=float, default=16.0)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--optimizer", choices=("adamw", "muon", "comp-muon"), default="adamw")
    parser.add_argument("--muon-momentum", type=float, default=0.95)
    parser.add_argument("--muon-damping", type=float, default=1e-2)
    parser.add_argument("--depth-loss-mode", choices=("smoothl1", "l2", "kl"), default="smoothl1")
    parser.add_argument("--kl-direction", choices=("forward", "reverse"), default="forward")
    parser.add_argument("--kl-temperature", type=float, default=1.0)
    parser.add_argument("--pairwise-weight", type=float, default=0.0)
    parser.add_argument("--pairwise-teacher-weight", type=float, default=1.0)
    parser.add_argument("--pairwise-label-weight", type=float, default=0.0)
    parser.add_argument("--pairwise-tau", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--log-every", type=int, default=12)
    parser.add_argument("--save-checkpoint", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = CircuitLoRARepairConfig(
        dataset_root=args.dataset_root,
        checkpoint=args.checkpoint,
        circuit_summary=args.circuit_summary,
        output_dir=args.output_dir,
        encoder=args.encoder,
        input_size=args.input_size,
        device=args.device,
        scene_type=args.scene_type,
        train_images=args.train_images,
        eval_skip_images=args.eval_skip_images,
        eval_images=args.eval_images,
        allow_train_eval_overlap=bool(args.allow_train_eval_overlap),
        budget_nodes=args.budget_nodes,
        budget_values=args.budget_values,
        candidate_kinds=parse_csv(args.candidate_kinds),
        selection=args.selection,
        peft_method=args.peft_method,
        lora_placement=args.lora_placement,
        lora_window=args.lora_window,
        lora_module_set=args.lora_module_set,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        optimizer=args.optimizer,
        muon_momentum=args.muon_momentum,
        muon_damping=args.muon_damping,
        depth_loss_mode=args.depth_loss_mode,
        kl_direction=args.kl_direction,
        kl_temperature=args.kl_temperature,
        pairwise_weight=args.pairwise_weight,
        pairwise_teacher_weight=args.pairwise_teacher_weight,
        pairwise_label_weight=args.pairwise_label_weight,
        pairwise_tau=args.pairwise_tau,
        seed=args.seed,
        log_every=args.log_every,
        save_checkpoint=bool(args.save_checkpoint),
    )
    run(config)


if __name__ == "__main__":
    main()
