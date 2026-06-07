import unittest

import torch

from speculative_prefill.vllm_patch.middle_layer_scorer import (
    capture_layer_hidden_states,
    middle_layer_index,
    transformer_layers,
)


class CountingLayer(torch.nn.Module):
    def __init__(self, idx: int, counters: list[int]):
        super().__init__()
        self.idx = idx
        self.counters = counters

    def forward(self, hidden_states):
        self.counters[self.idx] += 1
        return hidden_states + (self.idx + 1)


class InnerModel(torch.nn.Module):
    def __init__(self, counters: list[int]):
        super().__init__()
        self.layers = torch.nn.ModuleList(
            CountingLayer(idx, counters) for idx in range(len(counters))
        )

    def forward(self, input_ids, **kwargs):
        hidden_states = input_ids.float().unsqueeze(-1)
        for layer in self.layers:
            hidden_states = layer(hidden_states)
        return hidden_states


class FakeLayeredModel(torch.nn.Module):
    def __init__(self, layer_count: int):
        super().__init__()
        self.counters = [0] * layer_count
        self.model = InnerModel(self.counters)
        self.kwargs = None

    def forward(self, **kwargs):
        self.kwargs = kwargs
        return self.model(**kwargs)


class DirectLayerModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = torch.nn.ModuleList([torch.nn.Identity()])


class CountingModule(torch.nn.Module):
    def __init__(self, key: str, counters: dict[str, int], delta: float):
        super().__init__()
        self.key = key
        self.counters = counters
        self.delta = delta

    def forward(self, hidden_states):
        self.counters[self.key] += 1
        return hidden_states + self.delta


class FakeQwenBlock(torch.nn.Module):
    def __init__(self, idx: int, counters: dict[str, int]):
        super().__init__()
        self.self_attn = CountingModule(f"attn_{idx}", counters, 10.0)
        self.mlp = CountingModule(f"ffn_{idx}", counters, 100.0)

    def forward(self, hidden_states):
        hidden_states = self.self_attn(hidden_states)
        hidden_states = self.mlp(hidden_states)
        return hidden_states


class FakeQwenModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.counters = {
            "attn_0": 0,
            "ffn_0": 0,
            "attn_1": 0,
            "ffn_1": 0,
        }
        self.layers = torch.nn.ModuleList([
            FakeQwenBlock(0, self.counters),
            FakeQwenBlock(1, self.counters),
        ])

    def forward(self, input_ids, **kwargs):
        hidden_states = input_ids.float().unsqueeze(-1)
        for layer in self.layers:
            hidden_states = layer(hidden_states)
        return hidden_states


class MiddleLayerScorerTest(unittest.TestCase):
    def test_middle_layer_index_is_one_based_and_clamped(self):
        self.assertEqual(middle_layer_index(28, 0.5), 14)
        self.assertEqual(middle_layer_index(28, 0.0), 1)
        self.assertEqual(middle_layer_index(28, 2.0), 28)

    def test_capture_layer_hidden_states_exits_after_target_layer(self):
        model = FakeLayeredModel(layer_count=6)
        input_ids = torch.tensor([[1, 2, 3]])

        states = capture_layer_hidden_states(
            model,
            input_ids=input_ids,
            layer_index=3,
        )

        self.assertEqual(model.counters, [1, 1, 1, 0, 0, 0])
        self.assertEqual(states.shape, (1, 3, 1))
        self.assertTrue(torch.allclose(states[:, :, 0], input_ids.float() + 1 + 2 + 3))
        self.assertNotIn("output_hidden_states", model.kwargs)
        self.assertFalse(model.kwargs["use_cache"])

    def test_transformer_layers_supports_direct_layers_attr(self):
        layers = transformer_layers(DirectLayerModel())

        self.assertEqual(len(layers), 1)

    def test_capture_first_attention_output_exits_before_ffn(self):
        model = FakeQwenModel()
        input_ids = torch.tensor([[1, 2]])

        states = capture_layer_hidden_states(
            model,
            input_ids=input_ids,
            layer_index=1,
            activation_target="attn",
        )

        self.assertEqual(model.counters, {
            "attn_0": 1,
            "ffn_0": 0,
            "attn_1": 0,
            "ffn_1": 0,
        })
        self.assertTrue(torch.allclose(states[:, :, 0], input_ids.float() + 10.0))

    def test_capture_first_ffn_output_exits_after_ffn(self):
        model = FakeQwenModel()
        input_ids = torch.tensor([[1, 2]])

        states = capture_layer_hidden_states(
            model,
            input_ids=input_ids,
            layer_index=1,
            activation_target="ffn",
        )

        self.assertEqual(model.counters, {
            "attn_0": 1,
            "ffn_0": 1,
            "attn_1": 0,
            "ffn_1": 0,
        })
        self.assertTrue(torch.allclose(states[:, :, 0], input_ids.float() + 110.0))


if __name__ == "__main__":
    unittest.main()
