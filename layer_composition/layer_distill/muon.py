from __future__ import annotations

import math
from collections.abc import Iterable

import torch


def split_muon_params(module: torch.nn.Module) -> tuple[list[torch.nn.Parameter], list[torch.nn.Parameter]]:
    muon_params: list[torch.nn.Parameter] = []
    adamw_params: list[torch.nn.Parameter] = []
    for param in module.parameters():
        if not param.requires_grad:
            continue
        if param.ndim >= 2:
            muon_params.append(param)
        else:
            adamw_params.append(param)
    return muon_params, adamw_params


def zeropower_via_newtonschulz5(grad: torch.Tensor, steps: int = 5, eps: float = 1e-7) -> torch.Tensor:
    original_shape = grad.shape
    matrix = grad.reshape(grad.shape[0], -1) if grad.ndim > 2 else grad
    update = matrix.float()
    transposed = update.shape[0] > update.shape[1]
    if transposed:
        update = update.T

    update = update / (update.norm() + eps)
    a, b, c = 3.4445, -4.7750, 2.0315
    for _ in range(steps):
        gram = update @ update.T
        update = a * update + (b * gram + c * (gram @ gram)) @ update

    if transposed:
        update = update.T
    return update.reshape(original_shape).to(dtype=grad.dtype)


class Muon(torch.optim.Optimizer):
    def __init__(
        self,
        params: Iterable[torch.nn.Parameter],
        *,
        lr: float = 0.02,
        momentum: float = 0.95,
        nesterov: bool = True,
        ns_steps: int = 5,
        weight_decay: float = 0.0,
    ):
        if lr <= 0:
            raise ValueError("lr must be positive")
        if not 0 <= momentum < 1:
            raise ValueError("momentum must be in [0, 1)")
        if ns_steps <= 0:
            raise ValueError("ns_steps must be positive")
        defaults = {
            "lr": lr,
            "momentum": momentum,
            "nesterov": nesterov,
            "ns_steps": ns_steps,
            "weight_decay": weight_decay,
        }
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            momentum = group["momentum"]
            nesterov = group["nesterov"]
            ns_steps = group["ns_steps"]
            weight_decay = group["weight_decay"]
            for param in group["params"]:
                if param.grad is None:
                    continue
                grad = param.grad
                if grad.ndim < 2:
                    raise ValueError("Muon expects matrix-shaped parameters")
                if weight_decay:
                    param.mul_(1 - lr * weight_decay)

                state = self.state[param]
                if momentum:
                    buffer = state.get("momentum_buffer")
                    if buffer is None:
                        buffer = torch.zeros_like(grad)
                        state["momentum_buffer"] = buffer
                    buffer.mul_(momentum).add_(grad)
                    grad = grad.add(buffer, alpha=momentum) if nesterov else buffer

                update = zeropower_via_newtonschulz5(grad, steps=ns_steps)
                if update.ndim >= 2:
                    fan_out = update.shape[0]
                    fan_in = math.prod(update.shape[1:])
                    update = update * math.sqrt(max(1.0, fan_out / fan_in))
                param.add_(update, alpha=-lr)
        return loss
