# Copyright 2026 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Licensed under the Apache License, Version 2.0 (the "License");
# http://www.apache.org/licenses/LICENSE-2.0
"""Regression tests for hope_baselines / plot_compression_curves / compress_resnet50 fixes.

Each test names the defect it guards. All tests need only NumPy, SciPy and
(for the plot tests) matplotlib / Pillow, mirroring test_hope.py's minimal
environment. Run with: python -m pytest test_baselines_regressions.py
"""

import os
import sys
import tempfile

import hope
import hope_baselines
import numpy as np
import plot_compression_curves


def _mixed_weights():
  """The mixed 4D Conv -> 2D Dense -> 2D Dense network from test_hope.py."""
  rng = np.random.default_rng(7)
  weights = {
      "conv1/kernel": rng.normal(0.0, 0.2, size=(3, 3, 3, 6)).astype(
          np.float32
      ),
      "bn1/gamma": rng.uniform(0.5, 1.5, size=(6,)).astype(np.float32),
      "bn1/beta": rng.normal(0.0, 0.3, size=(6,)).astype(np.float32),
      "bn1/moving_mean": rng.normal(0.0, 0.3, size=(6,)).astype(np.float32),
      "bn1/moving_variance": rng.uniform(0.3, 1.8, size=(6,)).astype(
          np.float32
      ),
      "bottleneck/kernel": rng.normal(0.0, 0.2, size=(6, 6)).astype(np.float32),
      "bn_bot/gamma": rng.uniform(0.5, 1.5, size=(6,)).astype(np.float32),
      "bn_bot/beta": rng.normal(0.0, 0.3, size=(6,)).astype(np.float32),
      "bn_bot/moving_mean": rng.normal(0.0, 0.3, size=(6,)).astype(np.float32),
      "bn_bot/moving_variance": rng.uniform(0.3, 1.8, size=(6,)).astype(
          np.float32
      ),
      "output/kernel": rng.normal(0.0, 0.2, size=(6, 4)).astype(np.float32),
      "output/bias": rng.normal(0.0, 0.1, size=(4,)).astype(np.float32),
  }
  # Near-duplicate neurons so WS_COSINE_MERGE performs MERGE actions,
  # including on the `bn_b=None` quadruplet.
  weights["conv1/kernel"][..., 1] = weights["conv1/kernel"][..., 0] * 1.001
  weights["bottleneck/kernel"][1, :] = (
      weights["bottleneck/kernel"][0, :] * 0.999
  )
  weights["bottleneck/kernel"][:, 1] = (
      weights["bottleneck/kernel"][:, 0] * 1.001
  )
  weights["output/kernel"][1, :] = weights["output/kernel"][0, :] * 0.999
  for bn in ("bn1", "bn_bot"):
    for p in ("gamma", "beta", "moving_mean", "moving_variance"):
      weights[f"{bn}/{p}"][1] = weights[f"{bn}/{p}"][0] * 1.001
  quadruplets = [
      ("conv1", "bn1", "bottleneck", "bn_bot"),
      ("bottleneck", "bn_bot", "output", None),
  ]
  return weights, quadruplets


def test_baselines_support_mixed_conv_dense_quadruplets():
  """All 7 baselines used to crash with IndexError on a 2D (Dense) consumer.

  _execute_baseline_prune/_merge indexed the consumer kernel as
  k_b[:, :, i, :] unconditionally, which is invalid for the 2D kernels that
  hope.py itself supports (mixed quadruplets with `bn_b=None`). Every method
  must now encode the mixed network to completion and decode losslessly.
  """
  weights, quadruplets = _mixed_weights()
  for method in hope_baselines.BASELINE_METHODS:
    final, curriculum, _ = hope_baselines.baseline_encode(
        weights,
        quadruplets,
        benchmark_method=method,
        eval_stride=0,
        output_file="",
    )
    assert len(curriculum) == 12, f"{method}: {len(curriculum)} steps"
    if method == "WS_COSINE_MERGE":
      assert any(
          p.action == "MERGE" and p.quadruplet[3] is None for p in curriculum
      ), "WS_COSINE_MERGE should MERGE on the bn_b=None quadruplet"
    decoded = hope.hope_decode(final, curriculum)
    for k, ref in weights.items():
      np.testing.assert_allclose(
          decoded[k], ref, rtol=1e-6, atol=1e-7, err_msg=f"{method}: {k}"
      )


def test_biascorr_prune_without_consumer_bn_does_not_compensate():
  """L1_JOINT_BIASCORR must not touch a consumer BN that does not exist.

  With `bn_b=None` there is no consumer BatchNorm running mean to
  compensate; hope.py's own prune guards this with `has_consumer_bn`. The
  baseline used to dereference `weights['None/moving_mean']`
  unconditionally.
  """
  weights, quadruplets = _mixed_weights()
  _, curriculum, _ = hope_baselines.baseline_encode(
      weights,
      quadruplets,
      benchmark_method="L1_JOINT_BIASCORR",
      eval_stride=0,
      output_file="",
  )
  for packet in curriculum:
    if packet.quadruplet[3] is None:
      assert packet.consumer_mean_before is None
    else:
      assert packet.consumer_mean_before is not None


def test_baseline_encode_all_zero_weights_with_eval():
  """Density must not divide by a zero initial nonzero count.

  An all-zero weights dict has total_params_initial == 0; recording a
  (density, accuracy) point used to raise ZeroDivisionError whenever
  eval_stride > 0.
  """
  rng = np.random.default_rng(0)
  weights, quadruplets = _mixed_weights()
  zeroed = {k: np.zeros_like(v) for k, v in weights.items()}
  _, curriculum, results = hope_baselines.baseline_encode(
      zeroed,
      quadruplets,
      benchmark_method="L1_IN",
      eval_fn=lambda _: 0.5,
      eval_stride=2,
      output_file="",
  )
  assert len(curriculum) == 12
  assert results, "expected (density, accuracy) points to be recorded"
  for density, acc in results:
    assert density == 0.0 and acc == 0.5


def _run_all() -> int:
  """Runs all ``test_*`` functions in this module and returns an exit code."""
  import traceback  # pylint: disable=g-import-not-at-top

  tests = [
      (name, fn)
      for name, fn in sorted(globals().items())
      if name.startswith("test_") and callable(fn)
  ]
  failed = 0
  for name, fn in tests:
    try:
      fn()
      print(f"PASS  {name}")
    except Exception:  # pylint: disable=broad-exception-caught
      failed += 1
      print(f"FAIL  {name}")
      traceback.print_exc()
  print(f"\n{len(tests) - failed}/{len(tests)} tests passed.")
  return 1 if failed else 0


if __name__ == "__main__":
  sys.exit(_run_all())
