# Copyright 2026 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Licensed under the Apache License, Version 2.0 (the "License");
# http://www.apache.org/licenses/LICENSE-2.0
"""Regression tests for core `hope` defects found by reading hope.py.

Covers three bugs, each reproduced on the pre-fix code before fixing:

1. `hope_encode` raised `ZeroDivisionError` on an all-zero weights dict
   (density = nonzeros / initial_nonzeros with a zero denominator).
2. `hope_decode_step` restored NaN producer kernels for neurons whose BN
   moving variance is negative, because the decoder did not clamp the
   variance at 0 the way the encoder's BN folding does.
3. `read_quadruplet` raised `ValueError` on zero-size kernel dimensions
   (a zero-width producer layer, or a producer with zero input channels)
   because a `reshape(-1, ...)` dimension cannot be inferred for a
   0-element tensor.

Needs only NumPy and SciPy:

    python -m pytest test_hope_core_regressions.py
"""

import os
import tempfile
from typing import Dict

import hope
import numpy as np

WeightsDict = Dict[str, np.ndarray]

_WIDTH = 16  # Block input/output channels.
_MID = 8  # Bottleneck channels.


def _make_synthetic_resnet_weights(seed: int = 0) -> WeightsDict:
  """Tiny Keras-style ResNet weights (same construction as test_hope.py)."""
  rs = np.random.RandomState(seed)
  w: WeightsDict = {}

  def conv(name: str, k: int, c_in: int, c_out: int) -> None:
    w[f"{name}/kernel"] = (
        rs.standard_normal((k, k, c_in, c_out)) * np.sqrt(2.0 / (k * k * c_in))
    ).astype(np.float32)
    w[f"{name}/bias"] = (0.01 * rs.standard_normal(c_out)).astype(np.float32)

  def bn(name: str, c: int) -> None:
    w[f"{name}/gamma"] = rs.uniform(0.5, 1.5, c).astype(np.float32)
    w[f"{name}/beta"] = (0.5 * rs.standard_normal(c)).astype(np.float32)
    w[f"{name}/moving_mean"] = (0.5 * rs.standard_normal(c)).astype(np.float32)
    w[f"{name}/moving_variance"] = rs.uniform(0.5, 2.0, c).astype(np.float32)

  conv("conv1_conv", 7, 3, _WIDTH)
  bn("conv1_bn", _WIDTH)
  for b in (1, 2):
    p = f"conv2_block{b}"
    conv(f"{p}_1_conv", 1, _WIDTH, _MID)
    bn(f"{p}_1_bn", _MID)
    conv(f"{p}_2_conv", 3, _MID, _MID)
    bn(f"{p}_2_bn", _MID)
    if b == 1:
      conv(f"{p}_0_conv", 1, _WIDTH, _WIDTH)
      bn(f"{p}_0_bn", _WIDTH)
    conv(f"{p}_3_conv", 1, _MID, _WIDTH)
    bn(f"{p}_3_bn", _WIDTH)
  w["predictions/kernel"] = (0.1 * rs.standard_normal((_WIDTH, 10))).astype(
      np.float32
  )
  w["predictions/bias"] = np.zeros(10, np.float32)

  k1, k2 = w["conv2_block1_1_conv/kernel"], w["conv2_block1_2_conv/kernel"]
  k1[..., 1] = 1.01 * k1[..., 0]
  k2[:, :, 1, :] = 0.99 * k2[:, :, 0, :]
  for param in ("gamma", "beta", "moving_mean", "moving_variance"):
    w[f"conv2_block1_1_bn/{param}"][1] = w[f"conv2_block1_1_bn/{param}"][0]
  return w


def test_encode_all_zero_weights_completes_with_zero_density():
  """All-zero weights: density is 0.0 at every step, not ZeroDivisionError."""
  weights = {
      k: np.zeros_like(v) for k, v in _make_synthetic_resnet_weights().items()
  }
  quadruplets = hope.find_bottleneck_quadruplets(list(weights))

  final, curriculum, history = hope.hope_encode(
      weights, quadruplets, eval_stride=1
  )
  assert curriculum, "expected the encoder to consume the zero neurons"
  assert history, "expected a density history with eval_stride=1"
  assert all(float(density) == 0.0 for density, _ in history)

  restored = hope.hope_decode(final, curriculum)
  for k, v in weights.items():
    assert np.array_equal(restored[k], v), k


def test_encode_all_zero_weights_with_output_file():
  """The final CSV row of an all-zero model must not divide by zero either."""
  weights = {
      k: np.zeros_like(v) for k, v in _make_synthetic_resnet_weights().items()
  }
  quadruplets = hope.find_bottleneck_quadruplets(list(weights))
  with tempfile.TemporaryDirectory() as tmpdir:
    path = os.path.join(tmpdir, "density.csv")
    hope.hope_encode(weights, quadruplets, eval_stride=10, output_file=path)
    with open(path) as f:
      lines = f.read().strip().splitlines()
  assert lines[0] == "density,accuracy"
  assert lines[-1].startswith("0.000000,"), lines[-1]


def test_decode_restores_neurons_with_negative_moving_variance():
  """Negative BN variance: decoder clamps at 0 (like the encoder), no NaN."""
  weights = _make_synthetic_resnet_weights()
  weights["conv2_block1_1_bn/moving_variance"][:] = -0.5
  quadruplets = hope.find_bottleneck_quadruplets(list(weights))

  final, curriculum, _ = hope.hope_encode(weights, quadruplets, eval_stride=1000)
  restored = hope.hope_decode(final, curriculum)

  for k, v in restored.items():
    assert np.isfinite(v).all(), f"{k} restored with non-finite values"
  # Producer kernels are recovered by inverting the BN folding, so they are
  # compared with the same tolerance test_hope.py uses for them.
  np.testing.assert_allclose(
      restored["conv2_block1_1_conv/kernel"],
      weights["conv2_block1_1_conv/kernel"],
      rtol=1e-6,
      atol=1e-7,
  )
  # The stored (negative) variance itself is restored exactly.
  assert np.array_equal(
      restored["conv2_block1_1_bn/moving_variance"],
      weights["conv2_block1_1_bn/moving_variance"],
  )


def _zero_width_weights() -> WeightsDict:
  """One quadruplet whose producer layer has 0 output channels."""
  w: WeightsDict = {}
  w["b_1_conv/kernel"] = np.zeros((1, 1, 4, 0), dtype=np.float32)
  for param in ("gamma", "beta", "moving_mean", "moving_variance"):
    w[f"b_1_bn/{param}"] = np.zeros(0, dtype=np.float32)
  w["b_2_conv/kernel"] = np.zeros((1, 1, 0, 4), dtype=np.float32)
  w["b_2_bn/gamma"] = np.zeros(4, dtype=np.float32)
  w["b_2_bn/beta"] = np.zeros(4, dtype=np.float32)
  w["b_2_bn/moving_mean"] = np.zeros(4, dtype=np.float32)
  w["b_2_bn/moving_variance"] = np.ones(4, dtype=np.float32)
  return w


def test_read_quadruplet_zero_width_layer():
  """A zero-width producer layer reads as a (0, fan_in + c) joint matrix."""
  quadruplet = ("b_1_conv", "b_1_bn", "b_2_conv", "b_2_bn")
  joint_weights, fan_in, _ = hope.read_quadruplet(
      _zero_width_weights(), quadruplet
  )
  assert fan_in == 4
  assert joint_weights.shape == (0, 8)

  # Encoding such a model is a no-op, not a crash.
  _, curriculum, _ = hope.hope_encode(
      _zero_width_weights(), [quadruplet], eval_stride=1
  )
  assert curriculum == []


def test_read_quadruplet_zero_fan_in_layer():
  """A producer with 0 input channels reads with fan_in == 0."""
  w: WeightsDict = {}
  w["b_1_conv/kernel"] = np.zeros((1, 1, 0, 4), dtype=np.float32)
  w["b_1_bn/gamma"] = np.ones(4, dtype=np.float32)
  w["b_1_bn/beta"] = np.zeros(4, dtype=np.float32)
  w["b_1_bn/moving_mean"] = np.zeros(4, dtype=np.float32)
  w["b_1_bn/moving_variance"] = np.ones(4, dtype=np.float32)
  w["b_2_conv/kernel"] = np.zeros((1, 1, 4, 3), dtype=np.float32)
  w["b_2_bn/gamma"] = np.ones(3, dtype=np.float32)
  w["b_2_bn/beta"] = np.zeros(3, dtype=np.float32)
  w["b_2_bn/moving_mean"] = np.zeros(3, dtype=np.float32)
  w["b_2_bn/moving_variance"] = np.ones(3, dtype=np.float32)

  joint_weights, fan_in, _ = hope.read_quadruplet(
      w, ("b_1_conv", "b_1_bn", "b_2_conv", "b_2_bn")
  )
  assert fan_in == 0
  assert joint_weights.shape == (4, 3)


if __name__ == "__main__":
  for name, fn in sorted(
      (n, f) for n, f in list(globals().items()) if n.startswith("test_")
  ):
    fn()
    print(f"PASS {name}")
