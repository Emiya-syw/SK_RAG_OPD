# Copyright 2026 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

"""Regression coverage for equal-length mRoPE rows received from storage."""

import importlib.util
import pickle
import unittest
from pathlib import Path

import torch
from tensordict import TensorDict

_MODULE = Path(__file__).resolve().parents[2] / "verl/utils/tensordict_utils.py"
_SPEC = importlib.util.spec_from_file_location("mrope_tu", _MODULE)
tu = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(tu)


def make_batch(lengths, axes=4, canonical=False):
    rows = [torch.arange(axes * n).reshape(axes, n) + i * 10000 for i, n in enumerate(lengths)]
    positions = (
        tu.nested_tensor_from_tensor_list(rows, ragged_idx=2)
        if canonical
        else torch.nested.as_nested_tensor(rows, layout=torch.jagged)
    )
    tokens = torch.nested.as_nested_tensor([torch.arange(n) for n in lengths], layout=torch.jagged)
    return TensorDict({"input_ids": tokens, "position_ids": positions}, batch_size=[len(rows)]), rows


class TestMropeNestedLayout(unittest.TestCase):
    def assert_layout(self, batch, rows):
        positions = batch["position_ids"]
        self.assertEqual(positions._ragged_idx, 2)
        torch.testing.assert_close(positions.offsets(), batch["input_ids"].offsets())
        torch.testing.assert_close(positions.values(), torch.cat(rows, dim=-1))
        for actual, expected in zip(positions.unbind(), rows, strict=True):
            torch.testing.assert_close(actual, expected)

    def test_equal_unequal_single_and_serialized_rows(self):
        for lengths in ([7, 7], [5, 9], [7], [4, 4], [7, 7, 7, 7]):
            for axes in (3, 4):
                for serialize in (False, True):
                    with self.subTest(lengths=lengths, axes=axes, serialize=serialize):
                        batch, rows = make_batch(lengths, axes)
                        if serialize:
                            batch = pickle.loads(pickle.dumps(batch))
                        tu.maybe_fix_3d_position_ids(batch)
                        self.assert_layout(batch, rows)
                        fixed = batch["position_ids"]
                        tu.maybe_fix_3d_position_ids(batch)
                        self.assertIs(batch["position_ids"], fixed)

    def test_corrupted_metadata_and_rebatching(self):
        for canonical in (False, True):
            batch, rows = make_batch([7, 7], canonical=canonical)
            # Cover both the old workaround and serialization metadata loss.
            batch["position_ids"]._ragged_idx = 1 if canonical else 2
            tu.maybe_fix_3d_position_ids(batch)
            self.assert_layout(batch, rows)
            selected = tu.index_select_tensor_dict(batch, [1, 0])
            self.assert_layout(selected, rows[::-1])
            chunks = tu.chunk_tensordict(selected, 2)
            for chunk, row in zip(chunks, rows[::-1], strict=True):
                self.assert_layout(chunk, [row])
            joined = tu.concat_nested_tensors([chunk["position_ids"] for chunk in chunks])
            torch.testing.assert_close(joined.values(), torch.cat(rows[::-1], dim=-1))

    def test_qwen_rope_reproduces_failure_then_matches_reference(self):
        from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLTextConfig
        from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLTextRotaryEmbedding

        batch, rows = make_batch([7, 7])
        self.assertEqual(tuple(batch["position_ids"].values().shape), (8, 7))
        config = Qwen3VLTextConfig(rope_scaling={"rope_type": "default", "mrope_section": [24, 20, 20]})
        rope = Qwen3VLTextRotaryEmbedding(config)
        hidden = torch.zeros(1, 14, config.hidden_size)
        with self.assertRaisesRegex(RuntimeError, "size of tensor a .*3.*size of tensor b .*8"):
            rope(hidden, batch["position_ids"].values().unsqueeze(1))
        tu.maybe_fix_3d_position_ids(batch)
        packed = batch["position_ids"].values().unsqueeze(1)
        actual = rope(hidden, packed[1:])
        expected = rope(hidden, torch.cat(rows, dim=-1)[1:].unsqueeze(1))
        for a, b in zip(actual, expected, strict=True):
            torch.testing.assert_close(a, b)

    def test_invalid_layout_fails_before_model(self):
        batch, _ = make_batch([5, 9])
        batch["position_ids"] = torch.nested.as_nested_tensor([torch.zeros(4, 6)] * 2, layout=torch.jagged)
        with self.assertRaisesRegex(ValueError, "Cannot align position_ids"):
            tu.maybe_fix_3d_position_ids(batch)

    def test_text_positions_unchanged(self):
        batch = TensorDict({"position_ids": torch.arange(8).reshape(2, 4)}, batch_size=[2])
        original = batch["position_ids"]
        tu.maybe_fix_3d_position_ids(batch)
        self.assertIs(batch["position_ids"], original)
        batch = TensorDict({}, batch_size=[2])
        tu.maybe_fix_3d_position_ids(batch)


if __name__ == "__main__":
    unittest.main()
