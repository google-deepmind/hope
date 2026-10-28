# Copyright 2026 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Licensed under the Apache License, Version 2.0 (the "License");
# http://www.apache.org/licenses/LICENSE-2.0
"""Regression tests for merge-cache psi statistics on near-degenerate pairs.

For a pair whose input Gram determinant falls below the 1e-12 fallback
threshold in ``span_coefficients``, the psi statistics used to be
inconsistent with the direction-based quantities they are documented to
equal: ``psi_inner_i + psi_inner_j`` differed from ``alignment`` by a
factor of ``1 / sqrt(R_F)`` and ``psi_sq_norm`` from 1 by ``1 / R_F``.
The span fallback returns the (0.5, 0.5) coefficients of ``u_c`` itself,
discarding the ``sqrt(R_F)`` scale of psi*'s input weights
``sqrt(R_F) u_c``; the fix rescales the fallback coefficients by
``sqrt(R_F)`` in both ``merge_pair_geometry`` and ``build_merge_cache``.

The repro pair is neurons 0 and 1 of the first quadruplet of the
synthetic ResNet from ``test_hope`` (near-duplicates, Gram determinant
~1.6e-15). Before the fix (float64): ``psi_inner_sum = 1.1744937`` vs
``alignment = 1.1016244``, and ``psi_sq_norm = 1.1366697``.

    python test_merge_cache_regressions.py
    python -m pytest test_merge_cache_regressions.py
"""

import math
import sys
import time
import traceback

import hope
import numpy as np

from test_hope import make_synthetic_resnet_weights


def _near_duplicate_pair():
  """The float64 near-duplicate pair (0, 1) of the first quadruplet."""
  weights = make_synthetic_resnet_weights()
  quadruplet = hope.find_bottleneck_quadruplets(list(weights))[0]
  joint, fan_in, bn = hope.read_quadruplet(weights, quadruplet)
  joint64 = joint.astype(np.float64)
  bn64 = {k: v.astype(np.float64) for k, v in bn.items()}
  return joint64, fan_in, bn64


def _random_layer(seed=3, num_neurons=5, fan_in=6, out=4):
  """A well-conditioned float64 joint-weight layer and producer BN."""
  rng = np.random.default_rng(seed)
  joint = rng.standard_normal((num_neurons, fan_in + out))
  bn = {
      "gamma": rng.uniform(0.5, 1.5, num_neurons),
      "beta": 0.5 * rng.standard_normal(num_neurons),
      "mean": 0.3 * rng.standard_normal(num_neurons),
      "var": rng.uniform(0.5, 2.0, num_neurons),
  }
  return joint, fan_in, bn


def test_near_duplicate_pair_is_in_the_fallback_regime():
  """Precondition: the repro pair really has Gram det below 1e-12."""
  joint, fan_in, _ = _near_duplicate_pair()
  w_i, w_j = joint[0, :fan_in], joint[1, :fan_in]
  gram_det = float(np.dot(w_i, w_i)) * float(np.dot(w_j, w_j)) - float(
      np.dot(w_i, w_j)
  ) ** 2
  assert 0.0 <= gram_det < 1e-12


def test_near_duplicate_pair_scalar_psi_consistency():
  """Scalar geometry: psi inner products decompose the alignment."""
  joint, fan_in, bn = _near_duplicate_pair()
  geometry = hope.merge_pair_geometry(0, 1, joint, fan_in, bn)
  assert geometry["is_valid"]
  # <psi*, f_i + f_j>_o is documented as the two summands of b.
  assert abs(
      geometry["psi_inner_i"] + geometry["psi_inner_j"]
      - geometry["alignment"]
  ) < 1e-6
  # psi* is a unit operator in the o-norm: ||psi*||_o^2 = 1.
  assert abs(geometry["psi_sq_norm"] - 1.0) < 1e-6
  assert abs(geometry["psi_capacity"] - 1.0) < 1e-6
  # Root-cause pin: the fallback coefficients carry the sqrt(R_F)
  # scale of psi*'s input weights, i.e. 0.5 * sqrt(R_F) here.
  expected_c = 0.5 * math.sqrt(geometry["frobenius_ratio"])
  assert abs(geometry["psi_c1"] - expected_c) < 1e-9
  assert abs(geometry["psi_c2"] - expected_c) < 1e-9


def test_near_duplicate_pair_cache_psi_consistency():
  """Merge cache: psi fields agree with `alignment` and the scalar path."""
  joint, fan_in, bn = _near_duplicate_pair()
  cache = hope.build_merge_cache(joint, fan_in, bn)
  geometry = hope.merge_pair_geometry(0, 1, joint, fan_in, bn)
  assert cache["admissible"][0, 1]
  assert abs(cache["psi_inner_sum"][0, 1] - cache["alignment"][0, 1]) < 1e-6
  assert abs(cache["psi_sq_norm"][0, 1] - 1.0) < 1e-6
  assert abs(cache["psi_capacity"][0, 1] - 1.0) < 1e-6
  # Cache and scalar geometry now agree on the psi fields of this pair.
  assert abs(
      cache["psi_inner_sum"][0, 1]
      - (geometry["psi_inner_i"] + geometry["psi_inner_j"])
  ) < 1e-6
  assert abs(cache["psi_sq_norm"][0, 1] - geometry["psi_sq_norm"]) < 1e-6
  assert abs(cache["psi_capacity"][0, 1] - geometry["psi_capacity"]) < 1e-6
  assert abs(cache["alignment"][0, 1] - geometry["alignment"]) < 1e-6


def test_normal_pair_psi_consistency_unchanged():
  """Well-conditioned pairs: identities hold and cache == scalar (as before)."""
  joint, fan_in, bn = _random_layer()
  cache = hope.build_merge_cache(joint, fan_in, bn)
  num = joint.shape[0]
  for i in range(num):
    for j in range(i + 1, num):
      geometry = hope.merge_pair_geometry(i, j, joint, fan_in, bn)
      assert geometry["is_valid"]
      assert abs(
          geometry["psi_inner_i"] + geometry["psi_inner_j"]
          - geometry["alignment"]
      ) < 1e-8
      assert abs(geometry["psi_sq_norm"] - 1.0) < 1e-8
      assert abs(cache["psi_inner_sum"][i, j]
                 - geometry["psi_inner_i"] - geometry["psi_inner_j"]) < 1e-6
      assert abs(cache["psi_sq_norm"][i, j]
                 - geometry["psi_sq_norm"]) < 1e-6
      assert abs(cache["alignment"][i, j] - geometry["alignment"]) < 1e-6


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
