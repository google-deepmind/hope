# Copyright 2026 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0
"""Unit tests for the HOPE kernels, geometry, baselines and plot utilities.

Companion to ``test_hope.py`` (which covers the public encode/decode API
end to end). This file pins the individual building blocks against
independent references:

  * the closed-form Gaussian / biased-ReLU kernels (``relu_mean_scalar``,
    ``relu_variance``, ``bivariate_normal_cdf``, ``relu_covariance``)
    against Monte Carlo integration with a fixed seed and against
    ``scipy.stats``;
  * the pair geometry (``merge_pair_geometry``, ``build_merge_cache``,
    ``update_merge_cache_row``) against its defining invariants and
    against the scalar path;
  * ``read_quadruplet`` BN folding (including Dense producers/consumers
    and the negative-variance clamp), ``consumer_tap_fractions`` and
    ``neuron_capacities`` against hand-computed values;
  * ``InstructionPacket`` serialization and ``hope_decode`` edge cases;
  * the baseline prune-cost formulas, the baseline heap manager semantics
    and baseline-specific behaviour (bias correction, partial encoding);
  * the ``plot_compression_curves`` data loader against hand-written
    files in every supported format.

Needs only NumPy and SciPy (matplotlib/Pillow plot-rendering checks are
skipped when those libraries are absent):

    python -m pytest test_hope_unit.py
    python test_hope_unit.py
"""

import functools
import math
import os
import pickle
import sys
import tempfile
import traceback
from typing import Dict

import hope
import hope_baselines
import numpy as np
import plot_compression_curves
from scipy import stats

from test_hope import make_synthetic_resnet_weights

WeightsDict = Dict[str, np.ndarray]


def _assert_raises(exc_type, fn, *args, **kwargs):
  """Asserts that ``fn(*args, **kwargs)`` raises ``exc_type``."""
  try:
    fn(*args, **kwargs)
  except exc_type:
    return
  raise AssertionError(f"expected {exc_type.__name__} from {fn}")


@functools.cache
def _synthetic():
  """Synthetic weights and their quadruplets, built once per process."""
  weights = make_synthetic_resnet_weights()
  quadruplets = hope.find_bottleneck_quadruplets(list(weights))
  return weights, quadruplets


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


# ==============================================================================
# 1. Gaussian helpers and ReLU kernels
# ==============================================================================


def test_normal_cdf_pdf_known_values():
  """Phi and phi match known constants and the scalar/vector paths agree."""
  assert hope._normal_cdf_scalar(0.0) == 0.5
  assert abs(hope._normal_cdf_scalar(1.0) - 0.8413447460685429) < 1e-15
  assert abs(hope._normal_pdf_scalar(0.0) - 1.0 / math.sqrt(2 * math.pi)) < 1e-16
  z = np.array([-2.0, -0.5, 0.0, 0.5, 2.0])
  # Symmetry Phi(-z) = 1 - Phi(z), and phi is even.
  np.testing.assert_allclose(
      hope._normal_cdf(z), 1.0 - hope._normal_cdf(-z), atol=1e-15
  )
  np.testing.assert_allclose(
      hope._normal_pdf(z), hope._normal_pdf(-z), atol=1e-17
  )
  for value in z:
    assert abs(hope._normal_cdf_scalar(float(value)) - hope._normal_cdf(
        np.array([value]))[0]) < 1e-15
    assert abs(hope._normal_pdf_scalar(float(value)) - hope._normal_pdf(
        np.array([value]))[0]) < 1e-16
  # The vectorized CDF is scipy's ndtr.
  np.testing.assert_allclose(hope._normal_cdf(z), stats.norm.cdf(z), atol=1e-15)


def test_relu_mean_scalar_known_values():
  """E[ReLU(y)] for hand-computable (gamma, beta) combinations."""
  # beta = 0: m = gamma * phi(0) = gamma / sqrt(2 pi).
  assert abs(hope.relu_mean_scalar(1.0, 0.0) - 0.3989422804014327) < 1e-12
  assert abs(hope.relu_mean_scalar(2.0, 0.0) - 2 * 0.3989422804014327) < 1e-12
  # Degenerate gamma = 0: y is the constant beta, so m = max(beta, 0).
  assert hope.relu_mean_scalar(0.0, 2.0) == 2.0
  assert hope.relu_mean_scalar(0.0, -2.0) == 0.0
  # A very positive mean passes through almost unchanged; a very negative
  # mean is almost surely rectified away.
  assert abs(hope.relu_mean_scalar(0.1, 5.0) - 5.0) < 1e-9
  assert 0.0 <= hope.relu_mean_scalar(0.1, -5.0) < 1e-9
  # The sign of gamma is irrelevant (the argument is |gamma|).
  assert hope.relu_mean_scalar(-1.0, 0.3) == hope.relu_mean_scalar(1.0, 0.3)
  # m is always non-negative and never exceeds E|y| <= |beta| + gamma.
  rng = np.random.default_rng(0)
  for gamma, beta in rng.uniform(-3, 3, size=(50, 2)):
    m = hope.relu_mean_scalar(float(gamma), float(beta))
    assert 0.0 <= m <= abs(beta) + abs(gamma) + 1e-9


def test_relu_kernels_match_monte_carlo():
  """Closed-form mean/variance/covariance agree with Monte Carlo sampling."""
  rng = np.random.default_rng(20261028)
  cases = [
      (1.0, 0.3, 0.7, -0.5, 0.4),
      (0.5, -1.2, 1.5, 0.8, -0.6),
      (2.0, 0.0, 1.0, 0.5, 0.0),
  ]
  n = 300_000
  for gamma_i, beta_i, gamma_j, beta_j, rho in cases:
    cov = rho * gamma_i * gamma_j
    samples = rng.multivariate_normal(
        [beta_i, beta_j],
        [[gamma_i**2, cov], [cov, gamma_j**2]],
        size=n,
    )
    rectified = np.clip(samples, 0.0, None)
    # Monte Carlo standard errors at n = 300k scale with the moments
    # themselves (~1e-3 at gamma = 1, ~1e-2 for the gamma = 2 variance),
    # so the tolerances are scale-aware; the seed is fixed, making the
    # comparisons deterministic.
    assert abs(rectified[:, 0].mean() - hope.relu_mean_scalar(
        gamma_i, beta_i)) < 6e-3 * max(1.0, gamma_i)
    assert abs(rectified[:, 0].var() - hope.relu_variance_scalar(
        gamma_i, beta_i)) < 6e-3 * max(1.0, gamma_i**2)
    mc_cov = float(np.cov(rectified.T)[0, 1])
    assert abs(mc_cov - hope.relu_covariance_scalar(
        gamma_i, beta_i, gamma_j, beta_j, rho)) < 6e-3 * max(
            1.0, gamma_i * gamma_j), (mc_cov, rho)


def test_relu_variance_known_values_and_vector_consistency():
  """Var[ReLU(y)] closed form at beta = 0 and scalar/vector agreement."""
  # beta = 0, gamma = 1: E[ReLU^2] = 1/2 and m^2 = 1/(2 pi).
  expected = 0.5 - 1.0 / (2.0 * math.pi)
  assert abs(hope.relu_variance_scalar(1.0, 0.0) - expected) < 1e-12
  # Degenerate gamma = 0 gives a constant, hence zero variance.
  assert hope.relu_variance_scalar(0.0, 2.0) == 0.0
  gammas = np.array([0.5, 1.0, 2.0, 0.05])
  betas = np.array([-1.0, 0.0, 0.7, 3.0])
  vector = hope.relu_variance(gammas, betas)
  scalar = np.array(
      [hope.relu_variance_scalar(float(g), float(b)) for g, b in
       zip(gammas, betas)]
  )
  np.testing.assert_allclose(vector, scalar, rtol=1e-12, atol=1e-14)
  assert np.all(vector >= 0.0)
  # Variance never exceeds the raw second moment E[y^2] = gamma^2 + beta^2.
  assert np.all(vector <= gammas**2 + betas**2 + 1e-12)


def test_bivariate_normal_cdf_matches_scipy():
  """Phi_2 agrees with scipy.stats.multivariate_normal.cdf to ~1e-10."""
  cases = [
      (0.0, 0.0, 0.5),
      (1.0, -0.5, 0.3),
      (-1.5, 2.0, -0.7),
      (0.7, 0.7, 0.99),
      (3.0, 3.0, 0.0),
      (-3.0, 1.0, 0.9),
      (0.0, 1.2, 0.0),
      (0.0, 2.0, 0.5),   # on the x = 0 axis
      (-0.25, 0.0, -0.4),  # on the y = 0 axis
  ]
  for x, y, rho in cases:
    got = float(
        hope.bivariate_normal_cdf(np.array([x]), np.array([y]),
                                  np.array([rho]))[0]
    )
    ref = stats.multivariate_normal(
        [0.0, 0.0], [[1.0, rho], [rho, 1.0]]).cdf([x, y])
    assert abs(got - ref) < 1e-10, (x, y, rho, got, ref)


def test_bivariate_normal_cdf_identities_and_edges():
  """Owen's-T identities: arcsine law, factorisation, limits, symmetry."""
  # Phi_2(0, 0; rho) = 1/4 + arcsin(rho) / (2 pi).
  for rho in (-0.9, -0.3, 0.0, 0.5, 0.95):
    got = float(
        hope.bivariate_normal_cdf(np.array([0.0]), np.array([0.0]),
                                  np.array([rho]))[0]
    )
    assert abs(got - (0.25 + math.asin(rho) / (2 * math.pi))) < 1e-12
  # rho = 0 factorises into the product of marginals.
  got = float(
      hope.bivariate_normal_cdf(np.array([0.8]), np.array([-0.4]),
                                np.array([0.0]))[0]
  )
  assert abs(got - stats.norm.cdf(0.8) * stats.norm.cdf(-0.4)) < 1e-12
  # rho -> +1 gives Phi(min(x, y)); rho -> -1 gives max(0, Phi(x)+Phi(y)-1).
  assert abs(float(hope.bivariate_normal_cdf(
      np.array([1.0]), np.array([2.0]), np.array([1.0]))[0])
      - stats.norm.cdf(1.0)) < 1e-9
  assert abs(float(hope.bivariate_normal_cdf(
      np.array([1.0]), np.array([1.0]), np.array([-1.0]))[0])
      - (2 * stats.norm.cdf(1.0) - 1.0)) < 1e-9
  # Marginal limits: y -> +inf recovers Phi(x); x -> -inf gives 0.
  assert abs(float(hope.bivariate_normal_cdf(
      np.array([0.5]), np.array([8.0]), np.array([0.4]))[0])
      - stats.norm.cdf(0.5)) < 1e-9
  assert float(hope.bivariate_normal_cdf(
      np.array([-8.0]), np.array([0.5]), np.array([0.4]))[0]) < 1e-12
  # Symmetry in (x, y) and range [0, 1] on a random grid (incl. |rho| > 1
  # inputs, which are clipped).
  rng = np.random.default_rng(1)
  xs, ys = rng.uniform(-4, 4, 40), rng.uniform(-4, 4, 40)
  rhos = rng.uniform(-1.5, 1.5, 40)
  forward = hope.bivariate_normal_cdf(xs, ys, rhos)
  np.testing.assert_allclose(
      forward, hope.bivariate_normal_cdf(ys, xs, rhos), atol=1e-12
  )
  assert np.all((0.0 <= forward) & (forward <= 1.0))


def test_relu_covariance_properties():
  """Symmetry, independence, monotonicity in rho and Cauchy-Schwarz."""
  # Independence: rho = 0 gives exactly zero covariance.
  assert hope.relu_covariance_scalar(1.0, 0.3, 0.7, -0.5, 0.0) == 0.0
  # Symmetry under swapping the two neurons.
  rng = np.random.default_rng(2)
  for _ in range(25):
    gi, bi, gj, bj = rng.uniform(-2, 2, 4)
    rho = float(rng.uniform(-0.95, 0.95))
    forward = hope.relu_covariance_scalar(abs(gi), bi, abs(gj), bj, rho)
    swapped = hope.relu_covariance_scalar(abs(gj), bj, abs(gi), bi, rho)
    assert abs(forward - swapped) < 1e-12
    # Cauchy-Schwarz: |C(i, j)| <= sqrt(v_i v_j).
    bound = math.sqrt(
        hope.relu_variance_scalar(abs(gi), bi)
        * hope.relu_variance_scalar(abs(gj), bj)
    )
    assert abs(forward) <= bound + 1e-9
  # For fixed marginals the covariance is non-decreasing in rho and
  # changes sign with it.
  by_rho = [
      hope.relu_covariance_scalar(1.0, 0.3, 0.8, -0.2, rho)
      for rho in (-0.8, -0.3, 0.0, 0.3, 0.8)
  ]
  assert all(b >= a for a, b in zip(by_rho, by_rho[1:]))
  assert by_rho[0] < 0.0 < by_rho[-1]
  # In the perfectly correlated limit the covariance approaches the
  # variance (rho is clipped at 0.9999, so agreement is to ~1e-3).
  var = hope.relu_variance_scalar(1.0, 0.3)
  assert abs(hope.relu_covariance_scalar(1.0, 0.3, 1.0, 0.3, 0.9999)
             - var) < 2e-3


def test_warped_correlation_properties():
  """Warped correlation: orthogonal/parallel/degenerate inputs, range."""
  e1 = np.array([1.0, 0.0, 0.0])
  e2 = np.array([0.0, 1.0, 0.0])
  assert hope.warped_correlation(e1, e2, 1.0, 1.0) == 0.0
  # Parallel vectors saturate at the +-(1 - 1e-7) clip of rho_eff.
  assert abs(hope.warped_correlation(e1, 2.0 * e1, 1.0, 1.0) - 1.0) < 2e-6
  assert abs(hope.warped_correlation(e1, -e1, 1.0, 1.0) + 1.0) < 2e-6
  # Degenerate guards: a zero weight vector or a zero gamma gives 0.
  assert hope.warped_correlation(np.zeros(3), e2, 1.0, 1.0) == 0.0
  assert hope.warped_correlation(e1, e2, 0.0, 1.0) == 0.0
  # Range and symmetry on random inputs; the sign follows the cosine.
  rng = np.random.default_rng(4)
  for _ in range(50):
    wi, wj = rng.standard_normal(20), rng.standard_normal(20)
    rho = hope.warped_correlation(wi, wj, 1.3, 0.7)
    assert -1.0 <= rho <= 1.0
    assert abs(rho - hope.warped_correlation(wj, wi, 0.7, 1.3)) < 1e-12
    assert rho == 0.0 or np.sign(rho) == np.sign(float(wi @ wj))


def test_span_coefficients_projection():
  """Least-squares span coefficients recover combinations and project."""
  rng = np.random.default_rng(5)
  wi, wj = rng.standard_normal(50), rng.standard_normal(50)
  c1, c2 = hope.span_coefficients(wi, wj, 2.5 * wi - 1.25 * wj)
  assert abs(c1 - 2.5) < 1e-10 and abs(c2 + 1.25) < 1e-10
  # For a general target the residual is orthogonal to both basis vectors.
  target = rng.standard_normal(50)
  c1, c2 = hope.span_coefficients(wi, wj, target)
  residual = target - (c1 * wi + c2 * wj)
  assert abs(float(residual @ wi)) < 1e-8
  assert abs(float(residual @ wj)) < 1e-8
  # Exactly parallel inputs hit the documented (0.5, 0.5) fallback.
  assert hope.span_coefficients(
      np.array([1.0, 1.0]), np.array([2.0, 2.0]),
      np.array([1.0, 0.0])) == (0.5, 0.5)


def test_combined_preactivation_and_correlations():
  """Combined (gamma, beta) of c1 y_i + c2 y_j and its correlations."""
  gamma, beta = hope._combined_preactivation(1.0, 0.0, 1.3, 0.4, 0.9, -0.2,
                                             0.5)
  assert (gamma, beta) == (1.3, 0.4)
  gamma, beta = hope._combined_preactivation(0.0, 1.0, 1.3, 0.4, 0.9, -0.2,
                                             0.5)
  assert (gamma, beta) == (0.9, -0.2)
  # Averaging two identical perfectly correlated variables changes
  # nothing; averaging independent ones shrinks gamma by 1/sqrt(2).
  gamma, beta = hope._combined_preactivation(0.5, 0.5, 1.1, 0.3, 1.1, 0.3,
                                             1.0)
  assert abs(gamma - 1.1) < 1e-12 and abs(beta - 0.3) < 1e-12
  gamma, _ = hope._combined_preactivation(0.5, 0.5, 1.1, 0.3, 1.1, 0.3, 0.0)
  assert abs(gamma - 1.1 / math.sqrt(2.0)) < 1e-12
  # Correlations of the combination with its components: identity mix.
  rho_ui, rho_uj = hope._preactivation_correlations(1.0, 0.0, 1.3, 1.3, 0.9,
                                                    0.5)
  assert abs(rho_ui - 1.0) < 1e-9 and abs(rho_uj - 0.5) < 1e-9
  rng = np.random.default_rng(6)
  for _ in range(25):
    c1, c2 = rng.uniform(-2, 2, 2)
    gi, gj = rng.uniform(0.2, 2.0, 2)
    rho = float(rng.uniform(-1, 1))
    gamma_u, _ = hope._combined_preactivation(c1, c2, gi, 0.0, gj, 0.0, rho)
    if gamma_u == 0.0:
      continue
    rho_ui, rho_uj = hope._preactivation_correlations(c1, c2, gamma_u, gi,
                                                      gj, rho)
    assert -1.0 <= rho_ui <= 1.0 and -1.0 <= rho_uj <= 1.0


# ==============================================================================
# 2. Quadruplet reading, tap fractions and capacities
# ==============================================================================


def test_find_bottleneck_quadruplets_incomplete_and_ordering():
  """Incomplete blocks yield only the quadruplets they can support."""
  block = ["a_1_conv", "a_1_bn", "a_2_conv", "a_2_bn", "a_3_conv", "a_3_bn"]
  assert hope.find_bottleneck_quadruplets(block) == [
      ("a_1_conv", "a_1_bn", "a_2_conv", "a_2_bn"),
      ("a_2_conv", "a_2_bn", "a_3_conv", "a_3_bn"),
  ]
  # Without BN_3 only Q1 survives; a lone second stage yields only Q2.
  assert hope.find_bottleneck_quadruplets(block[:-1]) == [
      ("a_1_conv", "a_1_bn", "a_2_conv", "a_2_bn")
  ]
  assert hope.find_bottleneck_quadruplets(block[2:]) == [
      ("a_2_conv", "a_2_bn", "a_3_conv", "a_3_bn")
  ]
  # Without BN_2 neither quadruplet is complete.
  assert hope.find_bottleneck_quadruplets(
      ["a_1_conv", "a_1_bn", "a_2_conv", "a_3_conv", "a_3_bn"]) == []
  assert hope.find_bottleneck_quadruplets([]) == []
  # Order follows the first occurrence of the producer layer, not the
  # numeric block order.
  shuffled = [
      "zz_2_conv", "zz_2_bn", "zz_3_conv", "zz_3_bn",
      "zz_1_conv", "zz_1_bn",
  ]
  assert hope.find_bottleneck_quadruplets(shuffled) == [
      ("zz_2_conv", "zz_2_bn", "zz_3_conv", "zz_3_bn"),
      ("zz_1_conv", "zz_1_bn", "zz_2_conv", "zz_2_bn"),
  ]
  # Duplicate names and repeated weight paths do not duplicate blocks.
  with_dupes = [f"{name}/kernel" for name in block] + block
  assert hope.find_bottleneck_quadruplets(with_dupes) == \
      hope.find_bottleneck_quadruplets(block)


def test_read_quadruplet_bn_folding():
  """Joint rows equal the hand-folded BN-scaled kernels, in C order."""
  weights, quadruplets = _synthetic()
  quad = quadruplets[0]
  joint, fan_in, bn = hope.read_quadruplet(weights, quad)
  kernel_a = weights[f"{quad[0]}/kernel"]
  kernel_b = weights[f"{quad[2]}/kernel"]
  num_neurons = kernel_a.shape[-1]
  assert joint.shape == (num_neurons, fan_in + kernel_b.size // num_neurons)
  scale = bn["gamma"] / np.sqrt(np.maximum(bn["var"], 0.0) + hope.BN_EPSILON)
  for i in (0, 3, num_neurons - 1):
    expected_in = (kernel_a[..., i] * scale[i]).reshape(-1)
    np.testing.assert_allclose(joint[i, :fan_in], expected_in, rtol=1e-6)
    expected_out = kernel_b[:, :, i, :].reshape(-1)
    np.testing.assert_array_equal(joint[i, fan_in:], expected_out)
  # The returned BN dict holds copies, not views into the weights.
  bn["gamma"][:] = -999.0
  assert np.all(weights[f"{quad[1]}/gamma"] != -999.0)


def test_read_quadruplet_dense_and_missing_bn():
  """Dense (2D) producers/consumers and the consumer-BN early failure."""
  weights = {
      "d1/kernel": np.array([[1.0, 2.0], [3.0, 4.0]]),
      "d1_bn/gamma": np.array([2.0, 1.0]),
      "d1_bn/beta": np.array([0.5, -0.5]),
      "d1_bn/moving_mean": np.array([0.1, 0.2]),
      "d1_bn/moving_variance": np.array([4.0, 9.0]),
      "head/kernel": np.array([[1.0, 0.0, 1.0], [0.0, 1.0, 1.0]]),
  }
  joint, fan_in, _ = hope.read_quadruplet(weights, ("d1", "d1_bn", "head",
                                                    None))
  assert fan_in == 2 and joint.shape == (2, 5)
  scale = np.array(
      [2.0 / math.sqrt(4.0 + hope.BN_EPSILON),
       1.0 / math.sqrt(9.0 + hope.BN_EPSILON)]
  )
  expected_in = np.stack(
      [weights["d1/kernel"][:, i] * scale[i] for i in range(2)]
  )
  np.testing.assert_allclose(joint[:, :fan_in], expected_in, rtol=1e-12)
  np.testing.assert_array_equal(joint[:, fan_in:], weights["head/kernel"])

  weights_synth, quadruplets = _synthetic()
  quad = quadruplets[0]
  # A missing consumer BN tensor fails early with a KeyError naming it...
  broken = {
      k: v for k, v in weights_synth.items() if not k.startswith(quad[3])
  }
  _assert_raises(KeyError, hope.read_quadruplet, broken, quad)
  # ...while bn_b=None skips that check entirely.
  joint_none, _, _ = hope.read_quadruplet(
      weights_synth, (quad[0], quad[1], quad[2], None)
  )
  joint_full, _, _ = hope.read_quadruplet(weights_synth, quad)
  np.testing.assert_array_equal(joint_none, joint_full)


def test_read_quadruplet_negative_variance_is_clamped():
  """A negative stored moving variance is clamped to 0 before folding."""
  weights, quadruplets = _synthetic()
  quad = quadruplets[0]
  weights = {k: v.copy() for k, v in weights.items()}
  weights[f"{quad[1]}/moving_variance"][0] = -5.0
  joint, fan_in, bn = hope.read_quadruplet(weights, quad)
  assert np.all(np.isfinite(joint))
  kernel_a = weights[f"{quad[0]}/kernel"]
  clamped_scale = float(bn["gamma"][0]) / math.sqrt(hope.BN_EPSILON)
  expected_in = (kernel_a[..., 0] * clamped_scale).reshape(-1)
  np.testing.assert_allclose(joint[0, :fan_in], expected_in, rtol=1e-5)


def test_consumer_tap_fractions():
  """Tap fractions: ones for 1x1/Dense, outer(r, r) for 3x3, overrides."""
  ones = np.ones((1, 1, 1))
  np.testing.assert_array_equal(
      hope.consumer_tap_fractions(np.zeros((8, 4)), "dense"), ones)
  np.testing.assert_array_equal(
      hope.consumer_tap_fractions(np.zeros((1, 1, 8, 8)), "conv"), ones)
  # Other spatial sizes also get all-ones fractions, at kernel shape.
  fives = hope.consumer_tap_fractions(np.zeros((5, 5, 2, 2)), "conv2_x")
  assert fives.shape == (5, 5, 1) and np.all(fives == 1.0)
  # 3x3 consumer in stage conv2 (H = 56): chi = outer(r, r), r = 55/56.
  chi = hope.consumer_tap_fractions(np.zeros((3, 3, 8, 8)),
                                    "conv2_block1_2_conv")
  r = 55.0 / 56.0
  np.testing.assert_allclose(chi[:, :, 0], np.outer([r, 1.0, r],
                                                    [r, 1.0, r]))
  assert chi.shape == (3, 3, 1) and chi[1, 1, 0] == 1.0
  # Unknown prefixes fall back to H = 14.
  chi_default = hope.consumer_tap_fractions(np.zeros((3, 3, 8, 8)),
                                            "unknown_layer")
  assert abs(chi_default[0, 0, 0] - (13.0 / 14.0) ** 2) < 1e-15
  # A custom map resolves by stage prefix, and an exact layer-name entry
  # wins over the prefix entry.
  chi_prefix = hope.consumer_tap_fractions(
      np.zeros((3, 3, 8, 8)), "conv9_block1_2_conv", {"conv9": 10.0})
  assert abs(chi_prefix[0, 0, 0] - 0.9**2) < 1e-15
  chi_exact = hope.consumer_tap_fractions(
      np.zeros((3, 3, 8, 8)), "conv9_block1_2_conv",
      {"conv9": 10.0, "conv9_block1_2_conv": 4.0})
  assert abs(chi_exact[0, 0, 0] - 0.75**2) < 1e-15


def test_neuron_capacities():
  """Capacities are ||w_out,i|| * sqrt(v_i), computed row by row."""
  weights, quadruplets = _synthetic()
  joint, fan_in, bn = hope.read_quadruplet(weights, quadruplets[0])
  capacities = hope.neuron_capacities(joint, fan_in, bn)
  variances = hope.relu_variance(np.abs(bn["gamma"]), bn["beta"])
  expected = np.linalg.norm(joint[:, fan_in:], axis=1) * np.sqrt(
      np.maximum(0.0, variances))
  np.testing.assert_allclose(capacities, expected, rtol=1e-6)
  assert capacities.dtype == joint.dtype  # float32 in, float32 out
  # Zero outgoing weights or a surely-dead neuron (beta << 0) give ~0.
  joint_zero = joint.copy()
  joint_zero[0, fan_in:] = 0.0
  assert hope.neuron_capacities(joint_zero, fan_in, bn)[0] == 0.0


# ==============================================================================
# 3. Pair geometry and the merge cache
# ==============================================================================


def test_merge_pair_geometry_invariants():
  """psi* is a unit-capacity operator; capacities match the ledger."""
  joint, fan_in, bn = _random_layer()
  geometry = hope.merge_pair_geometry(0, 1, joint, fan_in, bn,
                                       return_directions=True)
  assert geometry["is_valid"]
  # psi* at scale s = 1 has (ideally) exactly unit centred capacity.
  assert abs(geometry["psi_capacity"] - 1.0) < 1e-8
  assert abs(geometry["psi_sq_norm"] - 1.0) < 1e-8
  # <psi*, f_i + f_j>_o decomposes and equals the alignment b.
  assert abs(geometry["psi_inner_i"] + geometry["psi_inner_j"]
             - geometry["alignment"]) < 1e-8
  # a = ||f_i||_o^2 + ||f_j||_o^2, consistent with neuron_capacities.
  capacities = hope.neuron_capacities(joint, fan_in, bn)
  assert abs(geometry["capacity_i"] - capacities[0]) < 1e-9
  assert abs(geometry["capacity_j"] - capacities[1]) < 1e-9
  assert abs(geometry["sum_sq_capacity"]
             - (capacities[0] ** 2 + capacities[1] ** 2)) < 1e-9
  # Directions: u_c in R^(fan_in + 1) (weights + bias), both unit norm.
  assert geometry["u_c"].shape == (fan_in + 1,)
  assert abs(np.linalg.norm(geometry["u_c"]) - 1.0) < 1e-10
  assert abs(np.linalg.norm(geometry["v_star"]) - 1.0) < 1e-10
  assert -1.0 <= geometry["rho_hat"] <= 1.0
  # Without return_directions the direction vectors are withheld.
  plain = hope.merge_pair_geometry(0, 1, joint, fan_in, bn)
  assert "u_c" not in plain and "v_star" not in plain
  # A fully zero pair is degenerate and reported invalid, not an error.
  assert hope.merge_pair_geometry(
      0, 1, np.zeros_like(joint), fan_in, bn) == {"is_valid": False}


def test_build_merge_cache_matches_scalar_geometry():
  """The vectorized all-pairs cache agrees with the scalar geometry."""
  joint, fan_in, bn = _random_layer()
  cache = hope.build_merge_cache(joint, fan_in, bn)
  num = joint.shape[0]
  for key, array in cache.items():
    assert array.shape == (num, num), key
  # Diagonal is empty: no self-pairs, zero sums.
  assert not np.any(np.diag(cache["admissible"]))
  assert np.all(np.diag(cache["sum_sq_capacity"]) == 0.0)
  scalar_fields = {
      "sum_sq_capacity": lambda g: g["sum_sq_capacity"],
      "capacity_sum": lambda g: g["capacity_i"] + g["capacity_j"],
      "alignment": lambda g: g["alignment"],
      "psi_capacity": lambda g: g["psi_capacity"],
      "children_sq_norm": lambda g: g["sq_norm_i"] + g["sq_norm_j"],
      "psi_sq_norm": lambda g: g["psi_sq_norm"],
      "psi_inner_sum": lambda g: g["psi_inner_i"] + g["psi_inner_j"],
  }
  for i in range(num):
    for j in range(i + 1, num):
      geometry = hope.merge_pair_geometry(i, j, joint, fan_in, bn)
      assert geometry["is_valid"]
      assert cache["admissible"][i, j] and cache["admissible"][j, i]
      for key, extract in scalar_fields.items():
        assert abs(cache[key][i, j] - extract(geometry)) < 1e-6, (
            key, i, j)
        assert cache[key][i, j] == cache[key][j, i]  # symmetric


def test_merge_cache_on_synthetic_float32_layer():
  """Cache/scalar agreement on the synthetic float32 network (tol 1e-4).

  Uses the second quadruplet of block 1 and both quadruplets of block 2,
  whose neuron pairs are well conditioned. The first quadruplet's
  near-duplicate pair (0, 1) is covered separately below, because its
  psi statistics are ill-conditioned (see the lane report).
  """
  weights, quadruplets = _synthetic()
  scalar_fields = ("sum_sq_capacity", "capacity_sum", "alignment",
                   "psi_capacity", "children_sq_norm", "psi_sq_norm",
                   "psi_inner_sum")
  for quad in quadruplets[1:]:
    joint, fan_in, bn = hope.read_quadruplet(weights, quad)
    cache = hope.build_merge_cache(joint, fan_in, bn)
    for i in range(joint.shape[0]):
      for j in range(i + 1, joint.shape[0]):
        geometry = hope.merge_pair_geometry(i, j, joint, fan_in, bn)
        if not geometry["is_valid"]:
          assert not cache["admissible"][i, j]
          continue
        expected = {
            "sum_sq_capacity": geometry["sum_sq_capacity"],
            "capacity_sum": geometry["capacity_i"] + geometry["capacity_j"],
            "alignment": geometry["alignment"],
            "psi_capacity": geometry["psi_capacity"],
            "children_sq_norm": geometry["sq_norm_i"] + geometry["sq_norm_j"],
            "psi_sq_norm": geometry["psi_sq_norm"],
            "psi_inner_sum": geometry["psi_inner_i"] + geometry["psi_inner_j"],
        }
        for key in scalar_fields:
          assert abs(cache[key][i, j] - expected[key]) < 1e-4, (
              quad[0], key, i, j)


def test_merge_cache_near_duplicate_pair_capacity_fields():
  """Capacity-side cache fields stay accurate for near-duplicate pairs.

  Neurons 0 and 1 of the first synthetic quadruplet are near-duplicates
  (Gram determinant ~1e-15). The capacity/alignment fields of the cache
  still agree with the scalar geometry; the psi fields of this pair are
  ill-conditioned and intentionally not asserted here (documented in
  the lane report as a follow-up finding).
  """
  weights, quadruplets = _synthetic()
  joint, fan_in, bn = hope.read_quadruplet(weights, quadruplets[0])
  cache = hope.build_merge_cache(joint, fan_in, bn)
  geometry = hope.merge_pair_geometry(0, 1, joint, fan_in, bn)
  assert geometry["is_valid"] and cache["admissible"][0, 1]
  assert abs(cache["sum_sq_capacity"][0, 1]
             - geometry["sum_sq_capacity"]) < 1e-4
  assert abs(cache["capacity_sum"][0, 1]
             - (geometry["capacity_i"] + geometry["capacity_j"])) < 1e-4
  assert abs(cache["alignment"][0, 1] - geometry["alignment"]) < 1e-4
  assert abs(cache["children_sq_norm"][0, 1]
             - (geometry["sq_norm_i"] + geometry["sq_norm_j"])) < 1e-4


def test_update_merge_cache_row_recomputes_from_scalar_geometry():
  """After a row changes, its cache entries track the scalar geometry."""
  joint, fan_in, bn = _random_layer()
  cache = hope.build_merge_cache(joint, fan_in, bn)
  joint_changed = joint.copy()
  joint_changed[0, fan_in:] *= 2.0  # neuron 0's outgoing weights double
  hope.update_merge_cache_row(cache, 0, [1, 2, 3, 4], joint_changed, fan_in,
                              bn)
  for k in (1, 2, 3, 4):
    lo, hi = (0, k) if 0 < k else (k, 0)
    geometry = hope.merge_pair_geometry(lo, hi, joint_changed, fan_in, bn)
    assert cache["admissible"][0, k] and cache["admissible"][k, 0]
    assert abs(cache["capacity_sum"][0, k]
               - (geometry["capacity_i"] + geometry["capacity_j"])) < 1e-9
    assert abs(cache["alignment"][0, k] - geometry["alignment"]) < 1e-9
    assert cache["capacity_sum"][0, k] == cache["capacity_sum"][k, 0]
  # Pairs not touching neuron 0 keep their original (stale-free) values.
  fresh = hope.build_merge_cache(joint, fan_in, bn)
  assert cache["sum_sq_capacity"][1, 2] == fresh["sum_sq_capacity"][1, 2]


def test_hope_cost_manager_ledger():
  """CostManager registration, removal and neighbour listing."""
  _, _, bn = _random_layer()
  del bn
  manager = hope.CostManager()
  capacities = np.array([1.0, 2.0, 3.0])
  manager.register_layer("layer", 3, 6.0, capacities, {})
  assert manager.live_capacity["layer"] == 6.0
  assert manager.initial_capacity["layer"] == 6.0
  assert manager.active_neighbors("layer", 0) == [1, 2]
  # The manager copies the capacity array at registration time.
  capacities[:] = 0.0
  assert np.array_equal(manager.neuron_capacity["layer"],
                        np.array([1.0, 2.0, 3.0]))
  manager.remove_neuron("layer", 1)
  assert manager.active_neighbors("layer", 0) == [2]
  # Removing twice (or an unknown index) is a no-op, not an error.
  manager.remove_neuron("layer", 1)
  manager.remove_neuron("layer", 99)
  assert manager.active_neighbors("layer", 2) == [0]


# ==============================================================================
# 4. InstructionPacket serialization and decoder edge cases
# ==============================================================================


def test_instruction_packet_defaults_and_fields():
  """A minimal packet gets the documented default field values."""
  packet = hope.InstructionPacket(
      action="PRUNE", indices=(-1, 3, "conv_a"),
      quadruplet=("a", "b", "c", None), fan_in=16)
  assert packet.producer_bn_before == ()
  assert packet.consumer_mean_before is None
  assert packet.joint_rows_before == ()
  assert packet.victim_row_before is None
  assert packet.saved_block_tensors is None
  assert packet.norm == 0.0 and packet.e_b == 0.0 and packet.n_active == 0


def test_instruction_packet_pickle_roundtrip():
  """Packets (including array payloads) survive pickle, deeply."""
  merge = hope.InstructionPacket(
      action="MERGE", indices=(0, 1, "conv_a"),
      quadruplet=("a", "b", "c", "d"), fan_in=4,
      producer_bn_before=(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0),
      consumer_mean_before=np.array([0.1, 0.2]),
      joint_rows_before=(np.arange(6.0), np.arange(6.0) + 10.0),
      norm=1.5, e_b=2.5, n_active=7)
  restored = pickle.loads(pickle.dumps(merge))
  assert restored.action == "MERGE" and restored.indices == (0, 1, "conv_a")
  assert restored.producer_bn_before == merge.producer_bn_before
  assert restored.norm == 1.5 and restored.e_b == 2.5
  assert restored.n_active == 7
  for got, want in zip(restored.joint_rows_before, merge.joint_rows_before):
    np.testing.assert_array_equal(got, want)
  np.testing.assert_array_equal(restored.consumer_mean_before,
                                merge.consumer_mean_before)
  # The unpickled packet owns its arrays: mutating it is safe.
  restored.joint_rows_before[0][0] = -999.0
  assert merge.joint_rows_before[0][0] == 0.0
  evict = hope.InstructionPacket(
      action="EVICT", indices=("c1", "c2", "c3"),
      quadruplet=("a", "b", "c", "d"), fan_in=0,
      saved_block_tensors={"x/kernel": np.ones((2, 2), dtype=np.float32)})
  restored_evict = pickle.loads(pickle.dumps(evict))
  np.testing.assert_array_equal(
      restored_evict.saved_block_tensors["x/kernel"],
      evict.saved_block_tensors["x/kernel"])


def test_hope_decode_num_steps_edges():
  """num_steps = 0, an over-large count, and empty curricula."""
  weights, quadruplets = _synthetic()
  compressed, curriculum, _ = hope.hope_encode(weights, quadruplets,
                                               eval_fn=None)
  snapshot = {k: v.copy() for k, v in compressed.items()}
  decoded_zero = hope.hope_decode(compressed, curriculum, num_steps=0)
  for key in compressed:
    np.testing.assert_array_equal(decoded_zero[key], compressed[key])
    assert decoded_zero[key] is not compressed[key]  # copies, not aliases
  decoded_all = hope.hope_decode(compressed, curriculum, num_steps=None)
  decoded_over = hope.hope_decode(
      compressed, curriculum, num_steps=len(curriculum) + 100)
  for key in compressed:
    np.testing.assert_array_equal(decoded_over[key], decoded_all[key])
  # An empty curriculum decodes to an identical copy for any num_steps.
  same = hope.hope_decode(compressed, [], num_steps=None)
  for key in compressed:
    np.testing.assert_array_equal(same[key], snapshot[key])
  # Decoding never modifies the compressed input dictionary.
  for key in compressed:
    np.testing.assert_array_equal(compressed[key], snapshot[key])


def test_hope_decode_negative_num_steps_raises():
  """Documents current behaviour: negative num_steps raises ValueError.

  ``hope_decode`` forwards ``num_steps`` to ``itertools.islice``, which
  rejects negative stop values. (A friendlier error naming ``num_steps``
  would be an improvement; this test pins the current contract so a
  change is noticed deliberately.)
  """
  weights, quadruplets = _synthetic()
  compressed, curriculum, _ = hope.hope_encode(weights, quadruplets,
                                               eval_fn=None)
  _assert_raises(ValueError, hope.hope_decode, compressed, curriculum, -1)


def test_hope_decode_in_stages_matches_full_decode():
  """Decoding k steps, then the rest, restores the original weights."""
  weights, quadruplets = _synthetic()
  compressed, curriculum, _ = hope.hope_encode(weights, quadruplets,
                                               eval_fn=None)
  staged = hope.hope_decode(compressed, curriculum, num_steps=5)
  staged = hope.hope_decode(staged, curriculum[5:])
  for key, reference in weights.items():
    got = staged[key]
    assert got.dtype == reference.dtype and got.shape == reference.shape
    if key.endswith("_1_conv/kernel") or key.endswith("_2_conv/kernel"):
      np.testing.assert_allclose(got, reference, rtol=1e-6, atol=1e-7)
    else:
      np.testing.assert_array_equal(got, reference)


def test_decode_step_evict_restores_saved_tensors_in_place():
  """A hand-built EVICT packet restores its saved block tensors in place."""
  weights = {
      "x/kernel": np.zeros((2, 2), dtype=np.float32),
      "y/kernel": np.ones(3, dtype=np.float32),
  }
  saved = {
      "x/kernel": np.full((2, 2), 7.0, dtype=np.float32),
      "y/kernel": np.full(3, -2.0, dtype=np.float32),
  }
  packet = hope.InstructionPacket(
      action="EVICT", indices=("c1", "c2", "c3"),
      quadruplet=("a", "b", "c", "d"), fan_in=0, saved_block_tensors=saved)
  out = hope.hope_decode_step(weights, packet)
  assert out is weights  # in place
  np.testing.assert_array_equal(weights["x/kernel"], saved["x/kernel"])
  np.testing.assert_array_equal(weights["y/kernel"], saved["y/kernel"])


# ==============================================================================
# 5. Baselines
# ==============================================================================


def _baseline_cost_inputs():
  weights, quadruplets = _synthetic()
  joint, fan_in, bn = hope.read_quadruplet(weights, quadruplets[0])
  return joint.astype(np.float64), fan_in, {
      k: v.astype(np.float64) for k, v in bn.items()
  }


def test_compute_baseline_prune_cost_hand_computed():
  """Every method's prune cost equals its defining score / (norm + eps)."""
  joint, fan_in, bn = _baseline_cost_inputs()
  i = 0
  w_in, w_out = joint[i, :fan_in], joint[i, fan_in:]
  guard = 1e-9
  expected = {
      "L1_IN": np.sum(np.abs(w_in)) / (10.0 + guard),
      "L1_JOINT": np.sum(np.abs(joint[i])) / (20.0 + guard),
      "BN_SCALE": abs(bn["gamma"][i]) / (4.0 + guard),
      "SYNFLOW": (np.sum(np.abs(w_in)) * np.sum(np.abs(w_out)))
                 / (5.0 + guard),
  }
  norms = {"L1_IN": 10.0, "L1_JOINT": 20.0, "BN_SCALE": 4.0,
           "SYNFLOW": 5.0}
  for method, want in expected.items():
    got = hope_baselines.compute_baseline_prune_cost(
        method, i, joint, fan_in, bn, norms)
    assert abs(got - want) < 1e-12, method
  # WS_COSINE_MERGE prune cost is ||w~_in|| * ||w_out|| with the folded
  # bias appended to the input vector.
  scale_i = float(bn["gamma"][i]) / math.sqrt(
      max(float(bn["var"][i]), 0.0) + hope.BN_EPSILON)
  b_eff = float(bn["beta"][i]) - scale_i * float(bn["mean"][i])
  a_i = math.sqrt(float(w_in @ w_in) + b_eff * b_eff)
  want_ws = a_i * float(np.linalg.norm(w_out)) / (2.0 + guard)
  got_ws = hope_baselines.compute_baseline_prune_cost(
      "WS_COSINE_MERGE", i, joint, fan_in, bn, {"WS_COSINE_MERGE": 2.0})
  assert abs(got_ws - want_ws) < 1e-12
  # With no norms supplied the default denominator is (1 + guard), and a
  # zero score stays zero.
  got_default = hope_baselines.compute_baseline_prune_cost(
      "L1_IN", i, joint, fan_in, bn)
  assert abs(got_default - np.sum(np.abs(w_in)) / (1.0 + guard)) < 1e-12


def test_compute_baseline_prune_cost_biascorr_alias_and_fpgm():
  """The BIASCORR/P1A alias pair and both FPGM score paths agree."""
  joint, fan_in, bn = _baseline_cost_inputs()
  joint_norm = {"L1_JOINT_BIASCORR": 20.0}
  biascorr = hope_baselines.compute_baseline_prune_cost(
      "L1_JOINT_BIASCORR", 0, joint, fan_in, bn, joint_norm)
  alias = hope_baselines.compute_baseline_prune_cost(
      "L1_JOINT_P1A", 0, joint, fan_in, bn, {"L1_JOINT_P1A": 20.0})
  plain = hope_baselines.compute_baseline_prune_cost(
      "L1_JOINT", 0, joint, fan_in, bn, {"L1_JOINT": 20.0})
  # Same raw score (joint L1) under every name.
  assert biascorr == alias == plain
  # The BIASCORR lookup falls back to the P1A norm entry when its own
  # key is absent.
  fallback = hope_baselines.compute_baseline_prune_cost(
      "L1_JOINT_BIASCORR", 0, joint, fan_in, bn, {"L1_JOINT_P1A": 20.0})
  assert fallback == plain
  # FPGM: precomputed scores are used verbatim; without them the cost is
  # the sum of distances to every neuron's input row.
  scores = np.full(joint.shape[0], 7.0)
  got_pre = hope_baselines.compute_baseline_prune_cost(
      "FPGM", 0, joint, fan_in, bn, {"FPGM": 2.0}, fpgm_scores=scores)
  assert abs(got_pre - 7.0 / (2.0 + 1e-9)) < 1e-12
  direct = float(
      np.sum(np.linalg.norm(joint[:, :fan_in] - joint[0, :fan_in], axis=1)))
  got_direct = hope_baselines.compute_baseline_prune_cost(
      "FPGM", 0, joint, fan_in, bn, {"FPGM": 1.0})
  assert abs(got_direct - direct / (1.0 + 1e-9)) < 1e-9
  # Unknown methods are rejected, in both entry points.
  _assert_raises(ValueError, hope_baselines.compute_baseline_prune_cost,
                 "NOPE", 0, joint, fan_in, bn)
  weights, quadruplets = _synthetic()
  _assert_raises(ValueError, hope_baselines.baseline_encode, weights,
                 quadruplets, "NOPE")


def test_baseline_cost_manager_heap_semantics():
  """Min-heap order, inf/NaN rejection, lazy invalidation, compaction."""
  manager = hope_baselines.BaselineCostManager()
  manager.register_layer("layer", 3, 1.0)
  manager.push_action(0.5, "PRUNE", "layer", (1,), 4)
  manager.push_action(0.1, "PRUNE", "layer", (2,), 4)
  manager.push_action(float("inf"), "PRUNE", "layer", (0,), 4)
  manager.push_action(float("nan"), "PRUNE", "layer", (0,), 4)
  assert len(manager.heap) == 2  # inf and NaN costs are never queued
  best = manager.pop_best_valid_action()
  assert best.cost == 0.1 and best.location == (2,)
  # Removing neuron 1 makes its queued 0.5 action stale: the heap is
  # then empty of valid actions.
  manager.remove_neuron_idx("layer", 1)
  assert manager.pop_best_valid_action() is None
  # Invalidating a neuron (without removing it) also stales its actions.
  manager.push_action(0.3, "MERGE", "layer", (0, 2), 4)
  manager.invalidate_neuron("layer", 0)
  assert manager.pop_best_valid_action() is None
  # clean_heap physically drops stale entries and keeps the valid ones.
  manager.push_action(0.2, "PRUNE", "layer", (2,), 4)
  manager.push_action(0.4, "PRUNE", "layer", (1,), 4)  # neuron 1 is gone
  manager.clean_heap()
  assert [entry.location for entry in manager.heap] == [(2,)]
  # Entries order by cost alone.
  cheap = hope_baselines.BaselineActionEntry(0.1, "MERGE", "layer", (1,),
                                             (0,), 4)
  dear = hope_baselines.BaselineActionEntry(0.2, "PRUNE", "layer", (0,),
                                            (0,), 4)
  assert sorted([dear, cheap])[0] is cheap


def test_baseline_encode_partial_steps_and_alias_object():
  """partial_encoding_steps caps the curriculum; the class alias holds."""
  assert hope_baselines.HOPE_Baseline_Encoder is \
      hope_baselines.baseline_encode
  weights, quadruplets = _synthetic()
  _, curriculum, _ = hope_baselines.baseline_encode(
      weights, quadruplets, benchmark_method="L1_IN", eval_fn=None,
      eval_stride=0, partial_encoding_steps=3, output_file="")
  assert len(curriculum) == 3
  # A cap above the total action count behaves like no cap (4 quadruplets
  # x MID=8 neurons, as in the end-to-end suite).
  _, full, _ = hope_baselines.baseline_encode(
      weights, quadruplets, benchmark_method="L1_IN", eval_fn=None,
      eval_stride=0, partial_encoding_steps=1000, output_file="")
  assert len(full) == 32


def test_run_all_baselines_subset_and_no_files_without_eval():
  """run_all_baselines honours `methods` and writes nothing w/o eval_fn."""
  weights, quadruplets = _synthetic()
  with tempfile.TemporaryDirectory() as tmpdir:
    results = hope_baselines.run_all_baselines(
        weights, quadruplets, eval_fn=None, eval_stride=0,
        output_dir=tmpdir, methods=["BN_SCALE"])
    assert list(results.keys()) == ["BN_SCALE"]
    # Without an eval_fn there is nothing to record, so the returned
    # (density, accuracy) history for the method is empty.
    assert results["BN_SCALE"] == []
    assert os.listdir(tmpdir) == []


def test_biascorr_compensates_consumer_bn_mean():
  """L1_JOINT_BIASCORR: same actions as L1_JOINT, different BN means.

  The bias-corrected baseline absorbs each pruned neuron's mean shift
  into the downstream BN running mean (P1a). On the synthetic network
  the two methods choose the identical action sequence, and their final
  weights differ *only* in the terminal consumer BN moving means.
  """
  weights, quadruplets = _synthetic()
  final_plain, curriculum_plain, _ = hope_baselines.baseline_encode(
      weights, quadruplets, benchmark_method="L1_JOINT", eval_fn=None,
      eval_stride=0, output_file="")
  final_corr, curriculum_corr, _ = hope_baselines.baseline_encode(
      weights, quadruplets, benchmark_method="L1_JOINT_BIASCORR",
      eval_fn=None, eval_stride=0, output_file="")
  assert [p.action for p in curriculum_plain] == \
      [p.action for p in curriculum_corr]
  differing = {
      key for key in weights
      if not np.array_equal(final_plain[key], final_corr[key])
  }
  assert differing == {
      "conv2_block1_3_bn/moving_mean",
      "conv2_block2_3_bn/moving_mean",
  }
  for key in differing:
    assert np.max(np.abs(final_plain[key] - final_corr[key])) > 0.1


# ==============================================================================
# 6. Plot utilities: the data loader and curve assembly
# ==============================================================================


def _write(path, text):
  with open(path, "w") as f:
    f.write(text)


def test_load_density_accuracy_file_formats():
  """The loader parses every documented format and skips junk lines."""
  with tempfile.TemporaryDirectory() as tmpdir:
    path = os.path.join(tmpdir, "two.csv")
    _write(path, "density,accuracy\n1.0,0.9\n0.5,0.8\n")
    arr = plot_compression_curves.load_density_accuracy_file(path)
    np.testing.assert_array_equal(arr, np.array([[1.0, 0.9], [0.5, 0.8]]))
    assert arr.dtype == np.float64

    path = os.path.join(tmpdir, "three.csv")
    _write(path, "step,density,top1_accuracy\n0,1.0,0.9\n1,0.5,0.75\n")
    arr = plot_compression_curves.load_density_accuracy_file(path)
    np.testing.assert_array_equal(arr, np.array([[1.0, 0.9], [0.5, 0.75]]))

    path = os.path.join(tmpdir, "pipe.txt")
    _write(path, "# comment\n\n1.0 | 0.9\n0.25|0.5\n")
    arr = plot_compression_curves.load_density_accuracy_file(path)
    np.testing.assert_array_equal(arr, np.array([[1.0, 0.9], [0.25, 0.5]]))

    path = os.path.join(tmpdir, "mixed.csv")
    _write(path, "garbage line\n0.9,0.1\n1,2,3,4\n")
    arr = plot_compression_curves.load_density_accuracy_file(path)
    # The 4-column row contributes its (2nd, 3rd) entries, per the
    # documented >= 3 column rule.
    np.testing.assert_array_equal(arr, np.array([[0.9, 0.1], [2.0, 3.0]]))

    path = os.path.join(tmpdir, "header_only.csv")
    _write(path, "density,accuracy\n")
    assert plot_compression_curves.load_density_accuracy_file(path) is None
    assert plot_compression_curves.load_density_accuracy_file(
        os.path.join(tmpdir, "missing.csv")) is None


def test_plot_compression_curves_no_data():
  """An empty data directory renders nothing and creates no file."""
  with tempfile.TemporaryDirectory() as tmpdir:
    out = os.path.join(tmpdir, "plot.png")
    fig, ax = plot_compression_curves.plot_compression_curves(
        data_dir=tmpdir, save_path=out)
    assert fig is None and ax is None
    assert not os.path.exists(out)


def test_plot_compression_curves_with_matplotlib():
  """With matplotlib: curves render, overrides apply, rcParams restored."""
  try:
    import matplotlib.pyplot as plt
  except ImportError:
    return  # Environment without matplotlib: loader tests still cover I/O.
  with tempfile.TemporaryDirectory() as tmpdir:
    custom = os.path.join(tmpdir, "custom.csv")
    _write(custom, "density,accuracy\n1.0,0.9\n0.5,0.8\n")
    _write(os.path.join(tmpdir, "results_HOPE.csv"),
           "density,accuracy\n1.0,0.7\n0.5,0.6\n")
    out = os.path.join(tmpdir, "plot.png")
    before_rc = dict(plt.rcParams)
    fig, ax = plot_compression_curves.plot_compression_curves(
        data_dir=tmpdir, save_path=out,
        file_overrides={"L1_In": custom},
        curve_specs=[("HOPE", "results_HOPE.csv", "blue"),
                     ("L1_In", "results_L1_IN.csv", "red")])
    assert fig is not None and ax is not None
    assert len(ax.get_lines()) == 2
    assert [line.get_label() for line in ax.get_lines()] == ["HOPE", "L1_In"]
    assert ax.get_xlabel() == "Model Density"
    assert ax.get_ylabel() == "Test Accuracy"
    assert os.path.isfile(out) and os.path.getsize(out) > 0
    assert dict(plt.rcParams) == before_rc
    plt.close("all")


def test_plot_hope_fallback_filename_and_main():
  """results_HOPE.csv falls back to density_accuracy.csv; main() is 0."""
  with tempfile.TemporaryDirectory() as tmpdir:
    _write(os.path.join(tmpdir, "density_accuracy.csv"),
           "density,accuracy\n1.0,0.9\n0.5,0.8\n")
    out = os.path.join(tmpdir, "plot.png")
    try:
      import matplotlib  # noqa: F401
      has_mpl = True
    except ImportError:
      has_mpl = False
    fig, _ = plot_compression_curves.plot_compression_curves(
        data_dir=tmpdir, save_path=out,
        curve_specs=[("HOPE", "results_HOPE.csv", "blue")])
    if has_mpl:
      # The fallback file was found and rendered as the HOPE curve.
      assert fig is not None and os.path.isfile(out)
    else:
      assert fig is None
    rc = plot_compression_curves.main(
        ["--data_dir", tmpdir, "--output", os.path.join(tmpdir, "m.png")])
    assert rc == 0


def _run_all() -> int:
  """Runs every ``test_*`` function in this module; returns an exit code."""
  tests = [
      (name, fn) for name, fn in sorted(globals().items())
      if name.startswith("test_") and callable(fn)
  ]
  failures = 0
  for name, fn in tests:
    try:
      fn()
    except Exception:  # pylint: disable=broad-except
      failures += 1
      print(f"FAIL {name}")
      traceback.print_exc()
    else:
      print(f"PASS {name}")
  print(f"{len(tests) - failures}/{len(tests)} tests passed")
  return 1 if failures else 0


if __name__ == "__main__":
  sys.exit(_run_all())
