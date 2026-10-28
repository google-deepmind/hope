# Copyright 2026 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Licensed under the Apache License, Version 2.0 (the "License");
# http://www.apache.org/licenses/LICENSE-2.0
"""Regression tests for the encoder's zero-BN-scale (gamma = 0) guard.

A neuron whose BN scale gamma / sqrt(var + eps) is zero has an identically
zero BN-folded filter row, so the PRUNE/MERGE packets that store only the
folded row cannot restore its raw filter: `hope_encode` used to emit such a
curriculum silently and full decoding restored zeros where the filter had
been (the decode-side limitation is disclosed in `hope_decode_step`'s
docstring). The encoder now refuses, at registration time, any registered
neuron with a zero BN scale and a nonzero raw filter; a zero-scale neuron
whose filter is already zero loses nothing and is still encoded.

Run with: python -m pytest test_hope_gamma_guard.py
          python test_hope_gamma_guard.py
"""

import sys
import time
import traceback

import hope
import numpy as np

from test_hope import make_synthetic_resnet_weights

# Producer kernels are restored as (k * s) / s in float32, as in test_hope.py.
RTOL, ATOL = 1e-6, 1e-7


def _is_producer_kernel(key: str) -> bool:
  return key.endswith("_1_conv/kernel") or key.endswith("_2_conv/kernel")


def _assert_restored(restored, reference) -> None:
  assert set(restored) == set(reference)
  for k, ref in reference.items():
    got = restored[k]
    if _is_producer_kernel(k):
      np.testing.assert_allclose(got, ref, rtol=RTOL, atol=ATOL, err_msg=k)
      assert np.count_nonzero(got) == np.count_nonzero(ref), k
    else:
      assert np.array_equal(got, ref), k


def _encode_expecting_value_error(weights) -> ValueError:
  try:
    hope.hope_encode(weights, hope.find_bottleneck_quadruplets(weights))
  except ValueError as e:
    return e
  raise AssertionError("hope_encode did not raise ValueError")


def test_zero_gamma_with_nonzero_filter_raises():
  weights = make_synthetic_resnet_weights()
  weights["conv2_block1_1_bn/gamma"][3] = np.float32(0.0)
  assert np.count_nonzero(weights["conv2_block1_1_conv/kernel"][..., 3]) == 16
  e = _encode_expecting_value_error(weights)
  msg = str(e)
  assert "conv2_block1_1_conv" in msg, msg
  assert "conv2_block1_1_bn" in msg, msg
  assert "neuron 3" in msg, msg
  assert "gamma" in msg, msg


def test_zero_gamma_in_later_quadruplet_raises():
  weights = make_synthetic_resnet_weights()
  weights["conv2_block2_2_bn/gamma"][5] = np.float32(0.0)
  e = _encode_expecting_value_error(weights)
  msg = str(e)
  assert "conv2_block2_2_conv" in msg, msg
  assert "neuron 5" in msg, msg


def test_zero_gamma_with_zero_filter_encodes_losslessly():
  # Nothing to lose: the folded row and the raw filter are both zero, so
  # the guard accepts the neuron and the round trip stays lossless.
  weights = make_synthetic_resnet_weights()
  weights["conv2_block1_1_bn/gamma"][3] = np.float32(0.0)
  weights["conv2_block1_1_conv/kernel"][..., 3] = np.float32(0.0)
  compressed, curriculum, _ = hope.hope_encode(
      weights, hope.find_bottleneck_quadruplets(weights)
  )
  _assert_restored(hope.hope_decode(compressed, curriculum), weights)


def test_tiny_nonzero_gamma_is_not_rejected():
  # The guard is an exact-zero check, not a threshold: restoration of a
  # neuron with a tiny but nonzero scale is inexact (see the
  # `hope_decode_step` docstring) but it is not the total, silent loss
  # this guard exists to prevent, so encoding still proceeds.
  weights = make_synthetic_resnet_weights()
  weights["conv2_block1_1_bn/gamma"][3] = np.float32(1e-30)
  compressed, curriculum, _ = hope.hope_encode(
      weights, hope.find_bottleneck_quadruplets(weights)
  )
  assert curriculum


def _run_all() -> int:
  """Runs all ``test_*`` functions in this module and returns an exit code."""
  tests = [
      (name, fn)
      for name, fn in globals().items()
      if name.startswith("test_") and callable(fn)
  ]
  failed = 0
  for name, fn in tests:
    t0 = time.time()
    try:
      fn()
      print(f"PASS  {name} ({time.time() - t0:.2f}s)")
    except Exception:  # pylint: disable=broad-exception-caught
      failed += 1
      print(f"FAIL  {name}")
      traceback.print_exc()
  print(f"\n{len(tests) - failed}/{len(tests)} tests passed.")
  return 1 if failed else 0


if __name__ == "__main__":
  sys.exit(_run_all())
