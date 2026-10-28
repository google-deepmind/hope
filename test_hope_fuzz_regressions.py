# Copyright 2026 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Licensed under the Apache License, Version 2.0 (the "License");
# http://www.apache.org/licenses/LICENSE-2.0
"""Regression tests from a property-based fuzz sweep over ``hope_encode``.

The sweep encodes hundreds of small synthetic networks (varied widths,
weight scales, BatchNorm ranges and sparsity) and checks the encoder's
invariants on every run: encoding completes, the recorded density history
is monotone non-increasing, full decoding is lossless, staged decoding
equals one-shot decoding, and a pickled curriculum decodes identically.

It found one defect, reproduced minimally below: a MERGE whose children
had input slices eliminated by earlier prunes in the upstream layer wrote
a synthesized parent carrying those eliminated components again, so the
nonzero count (and the recorded density) *increased* at that step. The
parent is now masked to the support of the two live joint rows in
``hope._execute_merge``.

Run with: python -m pytest test_hope_fuzz_regressions.py
"""

import pickle

import hope
import numpy as np

RTOL, ATOL = 1e-6, 1e-7


def _make_weights(
    seed,
    width=12,
    mid=12,
    kscale=0.007154368226057808,
    gamma_lo=0.1,
    beta_scale=2.0,
    var_lo=0.001,
    var_hi=2.0,
    sparse=0.0,
):
  """Tiny 2-block ResNet-style weights (the fuzz fixture that failed)."""
  rs = np.random.RandomState(seed)
  w = {}

  def conv(name, k, c_in, c_out):
    w[f"{name}/kernel"] = (
        rs.standard_normal((k, k, c_in, c_out))
        * kscale
        * np.sqrt(2.0 / (k * k * c_in))
    ).astype(np.float32)
    w[f"{name}/bias"] = (0.01 * rs.standard_normal(c_out)).astype(np.float32)

  def bn(name, c):
    w[f"{name}/gamma"] = rs.uniform(gamma_lo, 3.0, c).astype(np.float32)
    w[f"{name}/beta"] = (beta_scale * rs.standard_normal(c)).astype(np.float32)
    w[f"{name}/moving_mean"] = (0.5 * rs.standard_normal(c)).astype(np.float32)
    w[f"{name}/moving_variance"] = rs.uniform(var_lo, var_hi, c).astype(
        np.float32
    )

  conv("conv1_conv", 7, 3, width)
  bn("conv1_bn", width)
  for b in (1, 2):
    p = f"conv2_block{b}"
    conv(f"{p}_1_conv", 1, width, mid)
    bn(f"{p}_1_bn", mid)
    conv(f"{p}_2_conv", 3, mid, mid)
    bn(f"{p}_2_bn", mid)
    if b == 1:
      conv(f"{p}_0_conv", 1, width, width)
    conv(f"{p}_3_conv", 1, mid, width)
    if b == 1:
      bn(f"{p}_0_bn", width)
    bn(f"{p}_3_bn", width)
  w["predictions/kernel"] = (0.1 * rs.standard_normal((width, 10))).astype(
      np.float32
  )
  w["predictions/bias"] = np.zeros(10, np.float32)
  if sparse:
    for key in list(w):
      if key.endswith("kernel"):
        mask = rs.random(w[key].shape) < sparse
        w[key] = w[key].copy()
        w[key][mask] = 0
  return w


def _is_producer_kernel(key):
  return key.endswith("_1_conv/kernel") or key.endswith("_2_conv/kernel")


def _assert_restored(restored, reference):
  assert set(restored) == set(reference)
  for key, ref in reference.items():
    got = restored[key]
    assert got.dtype == ref.dtype and got.shape == ref.shape, key
    assert np.count_nonzero(got) == np.count_nonzero(ref), (
        f"{key}: sparsity pattern differs"
    )
    if _is_producer_kernel(key):
      np.testing.assert_allclose(got, ref, rtol=RTOL, atol=ATOL, err_msg=key)
    else:
      assert np.array_equal(got, ref), f"{key}: not restored bit-exactly"


def _check_invariants(weights):
  """Asserts the fuzz invariants for one fixture; returns the curriculum."""
  quadruplets = hope.find_bottleneck_quadruplets(list(weights))
  counts = []

  def record(current):
    counts.append(sum(np.count_nonzero(v) for v in current.values()))
    return 0.5

  final, curriculum, history = hope.hope_encode(
      weights, quadruplets, eval_fn=record, eval_stride=1
  )
  # The nonzero count (hence the recorded density) never increases: every
  # encoder action must release, never resurrect, parameters.
  assert all(b <= a for a, b in zip(counts, counts[1:])), (
      "nonzero count increased during encoding: "
      f"{[(i, a, b) for i, (a, b) in enumerate(zip(counts, counts[1:])) if b > a]}"
  )
  densities = [float(d) for d, _ in history]
  assert all(0.0 <= d <= 1.0 for d in densities)
  assert all(b <= a for a, b in zip(densities, densities[1:]))
  # Full decode is lossless.
  decoded = hope.hope_decode(final, curriculum)
  _assert_restored(decoded, weights)
  # Staged decode (k packets, then the rest) equals one-shot decode.
  n = len(curriculum)
  for k in {0, 1, n // 2, n}:
    staged = hope.hope_decode(final, curriculum, num_steps=k)
    rest = hope.hope_decode(staged, curriculum[k:])
    assert all(np.array_equal(rest[key], decoded[key]) for key in decoded)
  # A pickled curriculum decodes identically.
  reloaded = pickle.loads(pickle.dumps(curriculum))
  decoded2 = hope.hope_decode(final, reloaded)
  assert all(np.array_equal(decoded2[key], decoded[key]) for key in decoded)
  return curriculum


def test_merge_does_not_resurrect_pruned_input_slices():
  """Minimal repro: upstream prunes, then a downstream MERGE.

  With seed 497771 the first eight actions prune eight of the twelve
  neurons of ``conv2_block1_1_conv``; a later MERGE in
  ``conv2_block1_2_conv`` then synthesized a parent whose kernel was
  nonzero on all eight eliminated input slices (+20 nonzeros overall,
  density 0.5525 -> 0.5560). The masked parent keeps those slices zero.
  """
  weights = _make_weights(seed=497771)
  curriculum = _check_invariants(weights)
  assert any(p.action == "MERGE" for p in curriculum)


def test_property_sweep_density_and_lossless():
  """Seeded sweep over scales, shapes and sparsity (all must satisfy the
  invariants in ``_check_invariants``)."""
  configs = [
      dict(seed=11, width=8, mid=4, kscale=0.05),
      dict(seed=23, width=12, mid=6, kscale=0.001),
      dict(seed=37, width=8, mid=8, kscale=0.13, sparse=0.4),
      dict(seed=41, width=16, mid=6, kscale=0.17, sparse=0.4),
      dict(seed=53, width=8, mid=4, kscale=0.001, sparse=0.7),
      dict(seed=67, width=12, mid=8, kscale=11.4, sparse=0.3),
      dict(seed=71, width=8, mid=6, kscale=0.043, sparse=0.3),
      dict(seed=83, width=12, mid=12, kscale=0.74, gamma_lo=0.5, beta_scale=0.5),
  ]
  for cfg in configs:
    _check_invariants(_make_weights(**cfg))
