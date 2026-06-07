import contextlib

import torch


class _LayerCapture(Exception):
    pass


def middle_layer_index(num_layers: int, layer_fraction: float) -> int:
    if num_layers <= 0:
        raise ValueError("num_layers must be positive.")
    fraction = min(max(float(layer_fraction), 0.0), 1.0)
    return max(1, min(num_layers, round(num_layers * fraction)))


def transformer_layers(model) -> torch.nn.ModuleList:
    candidates = [
        ("layers",),
        ("model", "layers"),
        ("model", "model", "layers"),
        ("transformer", "h"),
        ("gpt_neox", "layers"),
    ]
    for path in candidates:
        current = model
        for attr in path:
            current = getattr(current, attr, None)
            if current is None:
                break
        if isinstance(current, torch.nn.ModuleList):
            return current
        if isinstance(current, (list, tuple)) and all(
            isinstance(layer, torch.nn.Module) for layer in current
        ):
            return current
    raise AttributeError("Could not locate transformer layers on the scorer model.")


def _first_tensor(output) -> torch.Tensor:
    if isinstance(output, torch.Tensor):
        return output
    if isinstance(output, (tuple, list)) and output and isinstance(output[0], torch.Tensor):
        return output[0]
    raise TypeError("Layer output does not contain a hidden-state tensor.")


def _target_module(layer: torch.nn.Module, activation_target: str) -> torch.nn.Module:
    if activation_target == "layer":
        return layer
    candidates = {
        "attn": ("self_attn", "attention", "attn"),
        "ffn": ("mlp", "feed_forward", "ffn"),
    }.get(activation_target)
    if candidates is None:
        raise ValueError("activation_target must be layer, attn, or ffn.")
    for name in candidates:
        module = getattr(layer, name, None)
        if isinstance(module, torch.nn.Module):
            return module
    raise AttributeError(
        f"Could not locate {activation_target} module on layer {type(layer).__name__}."
    )


def capture_layer_hidden_states(
    model,
    *,
    input_ids: torch.Tensor,
    layer_index: int,
    activation_target: str = "layer",
    attention_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    layers = transformer_layers(model)
    if not 1 <= layer_index <= len(layers):
        raise ValueError(f"layer_index must be in [1, {len(layers)}].")

    captured = None

    def hook(_module, _inputs, output):
        nonlocal captured
        captured = _first_tensor(output)
        raise _LayerCapture()

    module = _target_module(layers[layer_index - 1], activation_target)
    handle = module.register_forward_hook(hook)
    try:
        kwargs = {
            "input_ids": input_ids,
            "use_cache": False,
        }
        if attention_mask is not None:
            kwargs["attention_mask"] = attention_mask
        try:
            model(**kwargs)
        except _LayerCapture:
            pass
    finally:
        handle.remove()

    if captured is None:
        raise RuntimeError("Target layer hook did not capture hidden states.")
    return captured


@contextlib.contextmanager
def inference_scorer_mode(model):
    was_training = model.training
    model.eval()
    with torch.inference_mode():
        yield
    if was_training:
        model.train()
