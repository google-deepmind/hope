# Copyright 2026 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Licensed under the Apache License, Version 2.0 (the "License");
# http://www.apache.org/licenses/LICENSE-2.0
"""HOPE: Hilbert Operator for Progressive Encoding.

Strictly data-free, progressive compression of neural networks (Keras tensor
names, 4D Conv2D kernels in (h, w, c_in, c_out) layout and 2D Dense kernels in
(c_in, c_out) layout), together with a decoder that undoes the compression step
by step.

Method
------
Each hidden neuron of a bottleneck block (a convolution filter with its batch
normalization (BN), its ReLU and its
outgoing consumer weights) is treated as a rank-1 Hilbert-Schmidt operator f_i =
g_i (x) w_out,i. Norms and inner
products of these operators are evaluated in closed form under a Gaussian
surrogate of the layer input built from
the stored BN statistics, under which the pre-activation of neuron i is y_i ~
N(beta_i, gamma_i^2). The encoder
repeatedly executes the admissible action with the smallest distortion rate DR =
J / dP_init:

  PRUNE  one neuron i:
         J_prune = N ||f_i||_o / (E_a - ||f_i||_o)
  MERGE  two neurons (i, j) into one synthesized parent f_p = s* psi*:
         J_merge = N sqrt(||f_i - f_p||_o^2 + ||f_j - f_p||_o^2) / (E_a -
         ||f_i||_o - ||f_j||_o + ||f_p||_o)
  EVICT  the residual branch of a bottleneck block (internal layers l = 1, 2):
         J_evict = sum_{l in {1,2}} N_l (E_a^(l) / E_init^(l))

until no admissible action is left, and records for every action an
`InstructionPacket` from which the decoder
restores the weights as they were before that action.

Components of the method:
  [P1-Centred]  capacities, costs and parent synthesis on the zero-mean
  fluctuation subspace H0-perp, i.e. with the
                centred kernels v_i = C(i,i) = K(i,i) - m_i^2 and C(i,j) =
                K(i,j) - m_i m_j;
  [P3]          the exact biased-ReLU cross-kernel K(i,j) = E[ReLU(y_i)
  ReLU(y_j)] via the bivariate normal CDF
                Phi_2(x, y; rho) = 0.5 (Phi(x) + Phi(y)) - T(x, a_x) - T(y, a_y)
                - 0.5 I(x y < 0) (Owen's T);
  [P1a]         consumer BN mean compensation: the mean shift caused by a prune
  or merge is absorbed into the
                running mean of the downstream BN:
                  prune:  mu^b <- mu^b - m_i sum_t chi_t w_{t,i}
                  merge:  mu^b <- mu^b + sum_t chi_t (m_p w'_{t,p} - m_i w_{t,i}
                  - m_j w_{t,j})
  [P1b-Wiener]  at block eviction, BN_3's beta keeps the coherence-weighted
  Wiener shift
                  beta_3 <- delta_3* = alpha_3 beta_3,   alpha_3 = beta_3^2 /
                  (gamma_3^2 + beta_3^2).

Code map
--------
  1. Gaussian and ReLU kernels            m_i, v_i = K(i,i) - m_i^2, Phi_2,
  C(i,j) = K(i,j) - m_i m_j
  2. ResNet quadruplets and BN folding    joint rows [w_in,i^eff | w_out,i], tap
  fractions chi_t
  3. Pair geometry and parent synthesis   rho_hat, u_c, v*, a, b, R_F, s*, f_p =
  s* psi*
  4. Capacity ledger and decoupled cache  ||f_i||_o, CostManager, all-pairs
  merge cache
  5. Encoder                              hope_encode (greedy DR = J / dP_init
  loop)
  6. Decoder                              hope_decode_step, hope_decode

Notation: N is the live width of a layer, n (`fan_in`) the length of the input
part of a joint weight row,
||f_i||_o = ||w_out,i|| sqrt(v_i) the centred capacity of neuron i, E_a = sum_{k
in active} ||f_k||_o the live
centred capacity of a layer and E_init its value before compression. Companion
code to the paper
"Hilbert Operator for Progressive Encoding (HOPE): A Mathematical Framework for
Deconstructing Learned
Representations in Deep Networks" by H. Mobahi and P. L. Bartlett.

Example (weights: dict of Keras tensor name -> numpy array):

  quadruplets = find_bottleneck_quadruplets(weights)
  compressed, curriculum, history = hope_encode(weights, quadruplets)
  # The model after the first k encoder actions (raw producer filters up to
  rounding):
  model_k = hope_decode(compressed, curriculum, num_steps=len(curriculum) - k)

Requires numpy >= 2.0 and scipy. The arithmetic follows NumPy 2's scalar
promotion rules (NEP 50); NumPy 1.x
rounds some intermediate products differently.
"""

import copy
import dataclasses
import itertools
import logging
import math
import time
import types
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Mapping,
    NamedTuple,
    Optional,
    Sequence,
    Set,
    Tuple,
    Union,
)

import numpy as np
from scipy.special import ndtr
from scipy.special import owens_t

__all__ = [
    "Quadruplet",
    "InstructionPacket",
    "find_bottleneck_quadruplets",
    "hope_encode",
    "hope_decode_step",
    "hope_decode",
]

logger = logging.getLogger(__name__)

# Normal-distribution constants.
_INV_SQRT2 = 1.0 / math.sqrt(2.0)
_INV_SQRT_2PI = 1.0 / math.sqrt(2.0 * math.pi)

# BN epsilon (eps in w_in,i^eff = (gamma_i / sqrt(sigma_i^2 + eps)) w_in,i),
# used consistently by the encoder (BN folding; parent variance anchoring
# sigma_p^2 = max(0, gamma_p^2 - eps)) and by the decoder. (The BN layers of
# Keras' ResNet-50 use 1.001e-5.)
BN_EPSILON = 1e-5

# Division guard used when raw producer kernels are recovered from BN-folded
# weights: w_raw = w_eff / (gamma / sqrt(var + eps) + RAW_WEIGHT_DIV_GUARD).
RAW_WEIGHT_DIV_GUARD = 1e-9

# Floor eps = 1e-12 for layer capacities and cost denominators (e.g.
# E_rem = max(E_a - ||f_i||_o - ||f_j||_o, eps)).
CAPACITY_FLOOR = 1e-12

# Feature-map size H of each ResNet-50 stage for 224 x 224 inputs, keyed by the
# stage prefix of the Keras layer names. Used for the zero-padding tap fractions
# chi = r r^T with r = ((H - 1)/H, 1, (H - 1)/H); any other prefix gets 14.
RESNET50_FEATURE_MAP_SIZE = types.MappingProxyType(
    {"conv2": 56.0, "conv3": 28.0, "conv4": 14.0, "conv5": 7.0}
)
_DEFAULT_FEATURE_MAP_SIZE = 14.0

# Parameter footprints: every neuron of a granular action carries 4 BN
# parameters (gamma, beta, mean, variance), so dP_init = n + c + 4; block
# eviction also removes 3 parameters per channel of BN_3 (gamma_3, mu_3,
# sigma_3^2; beta_3 is kept as delta_3* = alpha_3 beta_3), so
# dP_evict = sum_{l in {1,2}} N_l dP_l^init + 3 d_amb.
_BN_PARAMS_PER_NEURON = 4
_BN3_PARAMS_REMOVED_PER_CHANNEL = 3

# (conv_a, bn_a, conv_b, bn_b): layer names of a producer layer and its BN,
# and of the consumer layer and its optional BN (`None` when the consumer has
# no BN, e.g. a terminal classifier head), i.e. the coupled quadruplet
# (K_A, BN_a, K_B, BN_b).
Quadruplet = Tuple[str, str, str, Optional[str]]


@dataclasses.dataclass
class InstructionPacket:
  """One decoding instruction: information needed to undo one encoder action.

  Attributes:
    action: "PRUNE", "MERGE" or "EVICT".
    indices: PRUNE: (-1, i, conv_a). MERGE: (i, j, conv_a) with i < j; the
      parent was written into slot i and slot j was zeroed. EVICT: (conv_1,
      conv_2, conv_3) of the evicted bottleneck block.
    quadruplet: the quadruplet acted upon (for EVICT, the block's first
      quadruplet Q1).
    fan_in: length n of the input part of a joint weight row (see
      `read_quadruplet`).
    producer_bn_before: producer BN entries (gamma, beta, moving_mean,
      moving_variance) of the affected neuron(s) before the action: 4 values for
      PRUNE, 8 for MERGE (neuron i, then neuron j), empty for EVICT.
    consumer_mean_before: copy of the consumer BN moving mean before the action
      (PRUNE and MERGE when `bn_b` is present).
    joint_rows_before: MERGE only: the joint weight rows of neurons i and j
      before the action.
    victim_row_before: PRUNE only: the joint weight row of the pruned neuron
      before the action.
    saved_block_tensors: EVICT only: copies of all kernels and BN tensors of the
      block before the action.
    norm: centred capacity ||f_i||_o of the pruned neuron (PRUNE) or combined
      child capacity ||f_i||_o + ||f_j||_o (MERGE).
    e_b: post-action live layer capacity E_b (floored at `CAPACITY_FLOOR`).
    n_active: active neuron count N of the layer immediately before the action.
  """

  action: str
  indices: Union[Tuple[int, int, str], Tuple[str, str, str]]
  quadruplet: Quadruplet
  fan_in: int
  producer_bn_before: Tuple[Any, ...] = ()
  consumer_mean_before: Optional[np.ndarray] = None
  joint_rows_before: Tuple[np.ndarray, ...] = ()
  victim_row_before: Optional[np.ndarray] = None
  saved_block_tensors: Optional[Dict[str, np.ndarray]] = None
  norm: float = 0.0
  e_b: float = 0.0
  n_active: int = 0


# ==============================================================================
# 1. Gaussian and ReLU kernels
# ==============================================================================
# Under the BN surrogate the pre-activation of neuron i is
# y_i ~ N(beta_i, gamma_i^2), so the moments of ReLU(y_i) depend only on
# (|gamma_i|, beta_i). The kernel functions take |gamma| and beta;
# c = beta / |gamma| is the standardized bias. All quantities are centred
# (P1-Centred): v_i = C(i,i) = K(i,i) - m_i^2 and C(i,j) = K(i,j) - m_i m_j
# rather than K(i,i), K(i,j).


def _normal_cdf_scalar(z: float) -> float:
  """Standard normal CDF Phi(z)."""
  return 0.5 * (1.0 + math.erf(float(z) * _INV_SQRT2))


def _normal_pdf_scalar(z: float) -> float:
  """Standard normal density phi(z)."""
  zf = float(z)
  return _INV_SQRT_2PI * math.exp(-0.5 * zf * zf)


def _normal_cdf(z: np.ndarray) -> np.ndarray:
  """Vectorized standard normal CDF Phi(z)."""
  return ndtr(z)


def _normal_pdf(z: np.ndarray) -> np.ndarray:
  """Vectorized standard normal density phi(z)."""
  return _INV_SQRT_2PI * np.exp(-0.5 * np.square(z))


def relu_mean_scalar(gamma_abs: float, beta: float) -> float:
  """First moment m_i = E[ReLU(y_i)] for y_i ~ N(beta_i, gamma_i^2).

  Evaluates m_i = |gamma_i| phi(c_i) + beta_i Phi(c_i).

  Args:
    gamma_abs: |gamma_i|, the standard deviation of the surrogate
      pre-activation.
    beta: beta_i, its mean.

  Returns:
    m_i = |gamma_i| phi(c_i) + beta_i Phi(c_i) with c_i = beta_i / |gamma_i|,
    clipped at 0 (|gamma_i| + 1e-12 is used for |gamma_i| in both c_i and the
    |gamma_i| phi(c_i) term).
  """
  gamma_safe = abs(float(gamma_abs)) + 1e-12
  beta_f = float(beta)
  c = beta_f / gamma_safe
  m = gamma_safe * _normal_pdf_scalar(c) + beta_f * _normal_cdf_scalar(c)
  return max(0.0, float(m))


def relu_variance_scalar(gamma_abs: float, beta: float) -> float:
  """Centred self-kernel v_i = C(i,i) = K(i,i) - m_i^2 = Var[ReLU(y_i)].

  K(i,i) = (gamma_i^2 + beta_i^2) Phi(c_i) + beta_i |gamma_i| phi(c_i) is the
  second moment of ReLU(y_i) (with |gamma_i| + 1e-12 in c_i and in the
  phi(c_i) term); the final variance is clipped at 0.

  Args:
    gamma_abs: |gamma_i|, the standard deviation of the surrogate
      pre-activation.
    beta: beta_i, its mean.

  Returns:
    Centred self-kernel v_i = Var[ReLU(y_i)] as a Python float, clipped at 0.
  """
  gamma_abs_f = abs(float(gamma_abs))
  gamma_safe = gamma_abs_f + 1e-12
  beta_f = float(beta)
  c = beta_f / gamma_safe
  cdf_c = _normal_cdf_scalar(c)
  pdf_c = _normal_pdf_scalar(c)
  second_moment = (gamma_abs_f * gamma_abs_f + beta_f * beta_f) * cdf_c + (
      beta_f * gamma_safe * pdf_c
  )
  m = max(0.0, gamma_safe * pdf_c + beta_f * cdf_c)
  return max(0.0, float(second_moment - m * m))


def relu_variance(gamma_abs: np.ndarray, beta: np.ndarray) -> np.ndarray:
  """Vectorized centred self-kernel v_i = C(i,i) = K(i,i) - m_i^2.

  Evaluated in the floating dtype of the inputs; clips K(i,i), m_i and the
  resulting variance at 0.

  Args:
    gamma_abs: Array of |gamma_i|, surrogate pre-activation standard deviations.
    beta: Array of beta_i, surrogate pre-activation means.

  Returns:
    Array of centred self-kernels v_i in the floating dtype of the inputs.
  """
  gamma_safe = np.abs(gamma_abs) + 1e-12
  c = beta / gamma_safe
  cdf_c = _normal_cdf(c)
  pdf_c = _normal_pdf(c)
  second_moment = (np.square(gamma_abs) + np.square(beta)) * cdf_c + (
      beta * gamma_safe * pdf_c
  )
  second_moment = np.maximum(0.0, second_moment)
  m = np.maximum(0.0, gamma_safe * pdf_c + beta * cdf_c)
  return np.maximum(0.0, second_moment - np.square(m))


def bivariate_normal_cdf(
    x: np.ndarray, y: np.ndarray, rho: np.ndarray
) -> np.ndarray:
  """Bivariate standard normal CDF Phi_2(x, y; rho) via Owen's T function.

  Computes:
    Phi_2(x, y; rho) = 0.5 (Phi(x) + Phi(y)) - T(x, a_x) - T(y, a_y)
                       - (0.5 if x y < 0 else 0),
    where a_x = (y - rho x) / (x sqrt(1 - rho^2)),
          a_y = (x - rho y) / (y sqrt(1 - rho^2)).

  Evaluated in float64; agrees with scipy.stats.multivariate_normal.cdf to about
  2e-16. rho is clipped to +-(1 - 1e-12); an argument of magnitude below 1e-15
  is replaced by +-1e-15 with its sign (+1e-15 for zero) to avoid division by
  zero at the coordinate axes; the result is clipped to [0, 1].

  Args:
    x: First upper integration limit(s).
    y: Second upper integration limit(s).
    rho: Correlation coefficient(s) between the two standard normal variables.

  Returns:
    Bivariate standard normal CDF values in float64, clipped to [0, 1].
  """
  x_arr = np.asarray(x, dtype=np.float64)
  y_arr = np.asarray(y, dtype=np.float64)
  rho_arr = np.clip(
      np.asarray(rho, dtype=np.float64), -1.0 + 1e-12, 1.0 - 1e-12
  )
  x_safe = np.where(
      np.abs(x_arr) < 1e-15, np.where(x_arr < 0.0, -1e-15, 1e-15), x_arr
  )
  y_safe = np.where(
      np.abs(y_arr) < 1e-15, np.where(y_arr < 0.0, -1e-15, 1e-15), y_arr
  )
  sqrt_1m_rho2 = np.sqrt(1.0 - np.square(rho_arr))
  a_x = (y_safe - rho_arr * x_safe) / (x_safe * sqrt_1m_rho2)
  a_y = (x_safe - rho_arr * y_safe) / (y_safe * sqrt_1m_rho2)
  half_if_opposite_signs = np.where((x_safe * y_safe) < 0.0, 0.5, 0.0)
  cdf2 = (
      0.5 * (ndtr(x_safe) + ndtr(y_safe))
      - owens_t(x_safe, a_x)
      - owens_t(y_safe, a_y)
      - half_if_opposite_signs
  )
  return np.clip(cdf2, 0.0, 1.0)


def relu_covariance(
    gamma_abs_i: np.ndarray,
    beta_i: np.ndarray,
    gamma_abs_j: np.ndarray,
    beta_j: np.ndarray,
    rho_hat: np.ndarray,
) -> np.ndarray:
  """Centred cross-kernel C(i,j) = K(i,j) - m_i m_j = Cov[ReLU(y_i), ReLU(y_j)].

  Exact cross-kernel (P3). For bivariate normal (y_i, y_j) with means
  (beta_i, beta_j), standard deviations (|gamma_i|, |gamma_j|), standardized
  biases c_i = beta_i / |gamma_i|, c_j = beta_j / |gamma_j| and correlation
  rho_hat:

    K(i,j) = E[ReLU(y_i) ReLU(y_j)]
           = |gamma_i| |gamma_j| * (
               (c_i c_j + rho_hat) Phi_2(c_i, c_j; rho_hat)
               + c_i phi(c_j) Phi(c_{i|j}) + c_j phi(c_i) Phi(c_{j|i})
               + (sqrt(1 - rho_hat^2) / (2 pi))
               * exp(-(c_i^2 - 2 rho_hat c_i c_j + c_j^2)
                     / (2 (1 - rho_hat^2)))
             ),
    where c_{i|j} = (c_i - rho_hat c_j) / sqrt(1 - rho_hat^2).

  Evaluated in float64, element-wise over broadcast inputs. Numerical guards:
  rho_hat is clipped to +-0.9999, |gamma| + 1e-12 is used for |gamma|, and
  K(i,j) is clipped at 0. C(i,j) is signed.

  Args:
    gamma_abs_i: Standard deviation(s) |gamma_i| of y_i.
    beta_i: Mean(s) beta_i of y_i.
    gamma_abs_j: Standard deviation(s) |gamma_j| of y_j.
    beta_j: Mean(s) beta_j of y_j.
    rho_hat: Pre-activation correlation(s) between y_i and y_j.

  Returns:
    Signed centred cross-kernel values C(i,j) in float64.
  """
  gamma_i_safe = np.abs(np.asarray(gamma_abs_i, dtype=np.float64)) + 1e-12
  gamma_j_safe = np.abs(np.asarray(gamma_abs_j, dtype=np.float64)) + 1e-12
  beta_i = np.asarray(beta_i, dtype=np.float64)
  beta_j = np.asarray(beta_j, dtype=np.float64)
  rho = np.clip(np.asarray(rho_hat, dtype=np.float64), -0.9999, 0.9999)

  c_i = beta_i / gamma_i_safe
  c_j = beta_j / gamma_j_safe

  sqrt_1m_rho2 = np.sqrt(1.0 - np.square(rho))
  c_i_given_j = (
      c_i - rho * c_j
  ) / sqrt_1m_rho2  # conditional thresholds c_{i|j}, c_{j|i}
  c_j_given_i = (c_j - rho * c_i) / sqrt_1m_rho2

  cdf2 = bivariate_normal_cdf(c_i, c_j, rho)
  pdf_i, pdf_j = _normal_pdf(c_i), _normal_pdf(c_j)
  cdf_i_given_j, cdf_j_given_i = _normal_cdf(c_i_given_j), _normal_cdf(
      c_j_given_i
  )

  # (1 - rho^2) phi_2(c_i, c_j; rho) = (sqrt(1 - rho^2) / (2 pi))
  #   * exp(-(c_i^2 - 2 rho c_i c_j + c_j^2) / (2 (1 - rho^2))).
  exponent = -(np.square(c_i) - 2.0 * rho * c_i * c_j + np.square(c_j)) / (
      2.0 * np.square(sqrt_1m_rho2)
  )
  density_term = (sqrt_1m_rho2 / (2.0 * np.pi)) * np.exp(exponent)

  term1 = (c_i * c_j + rho) * cdf2
  term2 = c_i * pdf_j * cdf_i_given_j
  term3 = c_j * pdf_i * cdf_j_given_i
  second_moment = np.maximum(
      0.0, gamma_i_safe * gamma_j_safe * (term1 + term2 + term3 + density_term)
  )

  m_i = np.maximum(0.0, gamma_i_safe * pdf_i + beta_i * _normal_cdf(c_i))
  m_j = np.maximum(0.0, gamma_j_safe * pdf_j + beta_j * _normal_cdf(c_j))
  return second_moment - m_i * m_j


def relu_covariance_scalar(
    gamma_abs_i: float,
    beta_i: float,
    gamma_abs_j: float,
    beta_j: float,
    rho_hat: float,
) -> float:
  """Scalar `relu_covariance` C(i,j), as a Python float."""
  value = relu_covariance(
      np.array([gamma_abs_i]),
      np.array([beta_i]),
      np.array([gamma_abs_j]),
      np.array([beta_j]),
      np.array([rho_hat]),
  )[0]
  return float(value)


# ==============================================================================
# 2. ResNet quadruplets and BN folding
# ==============================================================================


def find_bottleneck_quadruplets(layer_names: Iterable[str]) -> List[Quadruplet]:
  """Quadruplets (K_A, BN_a, K_B, BN_b) of bottleneck blocks, in network order.

  For every block X with layers X_1_conv, X_1_bn, X_2_conv, X_2_bn, X_3_conv,
  X_3_bn (Keras ResNet naming), returns
  Q1 = (X_1_conv, X_1_bn, X_2_conv, X_2_bn) and Q2 = (X_2_conv, X_2_bn,
  X_3_conv, X_3_bn). Names are matched by
  substring ("_1_conv" / "_2_conv"), as suits the Keras ResNet naming scheme.

  Args:
    layer_names: Keras layer names or weight paths such as
      "conv2_block1_1_conv/kernel"; anything after the first "/" is ignored and
      duplicates are dropped (first occurrence kept).

  Returns:
    The quadruplets whose four layers are all present, ordered by the first
    occurrence of their producer layer.
  """
  names: List[str] = []
  seen = set()
  for raw_name in layer_names:
    name = raw_name.split("/", 1)[0]
    if name not in seen:
      seen.add(name)
      names.append(name)

  quadruplets = []
  for name in names:
    if "_1_conv" in name:
      block_prefix = name.replace("_1_conv", "")
      c1, b1 = f"{block_prefix}_1_conv", f"{block_prefix}_1_bn"
      c2, b2 = f"{block_prefix}_2_conv", f"{block_prefix}_2_bn"
      if all(layer in seen for layer in [c1, b1, c2, b2]):
        quadruplets.append((c1, b1, c2, b2))
    elif "_2_conv" in name:
      block_prefix = name.replace("_2_conv", "")
      c2, b2 = f"{block_prefix}_2_conv", f"{block_prefix}_2_bn"
      c3, b3 = f"{block_prefix}_3_conv", f"{block_prefix}_3_bn"
      if all(layer in seen for layer in [c2, b2, c3, b3]):
        quadruplets.append((c2, b2, c3, b3))
  return quadruplets


def read_quadruplet(
    weights: Dict[str, np.ndarray], quadruplet: Quadruplet
) -> Tuple[np.ndarray, int, Dict[str, np.ndarray]]:
  """Joint weight rows [w_in,i^eff | w_out,i] and producer BN of a quadruplet.

  Read from the current tensors. Row i is the joint parameter vector of neuron
  i: [w_in,i^eff | w_out,i], where
  w_in,i^eff = (gamma_i / sqrt(max(sigma_i^2, 0) + eps)) * kernel_a[:, :, :, i]
  is the BN-folded producer filter (with folded bias
  b_i = beta_i - (gamma_i / sqrt(max(sigma_i^2, 0) + eps)) * mu_i) and
  w_out,i = kernel_b[:, :, i, :] is the neuron's slice of the consumer kernel,
  both flattened in C order. Convolution biases are not folded in (the
  bottleneck convolutions of the pretrained Keras ResNet-50 have zero biases).

  Args:
    weights: Mapping from Keras tensor name to parameter array.
    quadruplet: Layer names (conv_a, bn_a, conv_b, bn_b) of the quadruplet.

  Returns:
    (joint_weights, fan_in, producer_bn): joint_weights of shape
    (N, n + h_b * w_b * c_out) in the dtype of the tensors,
    fan_in = n = h_a * w_a * c_in, and producer_bn = copies of the producer BN
    tensors under the keys "gamma", "beta", "mean" (moving mean) and "var"
    (moving variance).
  """
  conv_a, bn_a, conv_b, bn_b = quadruplet

  kernel_a = weights[f"{conv_a}/kernel"]
  kernel_b = weights[f"{conv_b}/kernel"]
  kernel_a_4d = (
      kernel_a
      if kernel_a.ndim == 4
      else kernel_a.reshape(1, 1, kernel_a.shape[0], kernel_a.shape[1])
  )
  kernel_b_4d = (
      kernel_b
      if kernel_b.ndim == 4
      else kernel_b.reshape(1, 1, kernel_b.shape[0], kernel_b.shape[1])
  )

  gamma = weights[f"{bn_a}/gamma"].copy()
  beta = weights[f"{bn_a}/beta"].copy()
  moving_mean = weights[f"{bn_a}/moving_mean"].copy()
  moving_var = weights[f"{bn_a}/moving_variance"].copy()

  inv_std = 1.0 / np.sqrt(np.maximum(moving_var, 0.0) + BN_EPSILON)
  bn_scale = gamma * inv_std

  w_in_eff = kernel_a_4d * bn_scale.reshape(1, 1, 1, -1)
  w_in_rows = w_in_eff.reshape(-1, w_in_eff.shape[-1]).T
  w_out_rows = kernel_b_4d.transpose(2, 0, 1, 3).reshape(
      kernel_b_4d.shape[2], -1
  )

  joint_weights = np.hstack([w_in_rows, w_out_rows])
  fan_in = w_in_rows.shape[1]

  # Touch the consumer BN scale and variance when `bn_b` is present so
  # incomplete consumer BNs fail early with the same KeyError as the reference
  # implementation.
  if bn_b is not None:
    _ = weights[f"{bn_b}/gamma"], weights[f"{bn_b}/moving_variance"]

  producer_bn = {
      "gamma": gamma,
      "beta": beta,
      "mean": moving_mean,
      "var": moving_var,
  }

  return joint_weights, fan_in, producer_bn


def consumer_tap_fractions(
    kernel_b: np.ndarray,
    conv_b: str,
    feature_map_sizes: Optional[Mapping[str, float]] = None,
) -> np.ndarray:
  """In-bounds tap fractions chi_t of the consumer layer.

  chi_t is the fraction of output positions at which tap t of a zero-padded
  convolution reads inside the feature map: 1 for a 1x1 or Dense consumer and,
  for a 3x3 consumer (padding 1, stride 1) on an H x H map, chi = outer(r, r)
  with r = ((H - 1) / H, 1, (H - 1) / H). H is looked up in `feature_map_sizes`
  (defaulting to `RESNET50_FEATURE_MAP_SIZE`, ResNet-50 stage sizes for
  224 x 224 inputs) by exact layer name `conv_b` first and then by prefix
  `conv_b[:5]`, falling back to 14.0.

  Args:
    kernel_b: Consumer kernel of shape (h_b, w_b, c_in, c_out) or (c_in, c_out).
    conv_b: Name of the consumer layer.
    feature_map_sizes: Optional mapping from consumer layer name (or 5-character
      stage prefix) to spatial resolution H.

  Returns:
    Float64 array of tap fractions chi_t with shape (h_b, w_b, 1) (or (1, 1, 1)
    for 2D Dense kernels).
  """
  if kernel_b.ndim != 4:
    return np.ones((1, 1, 1), dtype=np.float64)
  h_b, w_b = kernel_b.shape[0], kernel_b.shape[1]
  if h_b == 1 and w_b == 1:
    return np.ones((1, 1, 1), dtype=np.float64)
  size_map = (
      RESNET50_FEATURE_MAP_SIZE
      if feature_map_sizes is None
      else feature_map_sizes
  )
  if conv_b in size_map:
    size = float(size_map[conv_b])
  else:
    size = float(size_map.get(conv_b[:5], _DEFAULT_FEATURE_MAP_SIZE))
  edge_fraction = (size - 1.0) / size
  if h_b == 3 and w_b == 3:
    chi_1d = np.array([edge_fraction, 1.0, edge_fraction], dtype=np.float64)
    return np.outer(chi_1d, chi_1d)[:, :, np.newaxis]
  return np.ones((h_b, w_b, 1), dtype=np.float64)


# ==============================================================================
# 3. Pair geometry and parent synthesis
# ==============================================================================
# For a merge candidate (i, j), u_c is the polarity-checked principal right
# singular vector of A = w_out,i w~_in,i^T + w_out,j w~_in,j^T, a unit vector in
# span(w~_in,i, w~_in,j) with w~_in = [w_in^eff; b]:
#   u_c = argmax_{u in {+u_hat, -u_hat}}
#     ||C(u,i) w_out,i + C(u,j) w_out,j|| / sqrt(C(u,u)),
# and v* = (C(u_c,i) w_out,i + C(u_c,j) w_out,j)
#   / ||C(u_c,i) w_out,i + C(u_c,j) w_out,j|| is the unit output direction.
# psi* denotes the parent at scale s = 1; the deployed parent is f_p = s* psi*.
# The cached constants a = ||f_i||_o^2 + ||f_j||_o^2 and b = <psi*, f_i + f_j>_o
# give s* in O(1) for any live capacity E_a.


def span_coefficients(
    w_i: np.ndarray, w_j: np.ndarray, target: np.ndarray
) -> Tuple[float, float]:
  """Least-squares coefficients (c1, c2) of `target` in span(w_i, w_j).

  Computed from the 2x2 normal equations to express a parent's effective input
  weights in the span of its children's:
  w_in,p^eff = c_1 w_in,i^eff + c_2 w_in,j^eff.

  Args:
    w_i: First basis vector (child i effective input weights).
    w_j: Second basis vector (child j effective input weights).
    target: Target vector to project onto span(w_i, w_j).

  Returns:
    Tuple (c1, c2) of least-squares coefficients as Python floats, or
    (0.5, 0.5) when the Gram determinant is below 1e-12 in absolute value.
  """
  g11 = float(np.dot(w_i, w_i))
  g12 = float(np.dot(w_i, w_j))
  g22 = float(np.dot(w_j, w_j))
  rhs_i = float(np.dot(w_i, target))
  rhs_j = float(np.dot(w_j, target))

  det = g11 * g22 - g12 * g12
  if abs(det) < 1e-12:
    return 0.5, 0.5

  c1 = (g22 * rhs_i - g12 * rhs_j) / det
  c2 = (g11 * rhs_j - g12 * rhs_i) / det
  return float(c1), float(c2)


def warped_correlation(
    w_in_i: np.ndarray, w_in_j: np.ndarray, gamma_i: float, gamma_j: float
) -> float:
  """Pairwise warped correlation rho_hat_ij of surrogate pre-activations.

  With rho_eff = <w_in,i^eff, w_in,j^eff> / (||w_in,i^eff|| ||w_in,j^eff||):
    kappa = (rho_eff / (1 - rho_eff^2)) * (|gamma_i| / ||w_in,i^eff||)
      * (|gamma_j| / ||w_in,j^eff||),
    rho_hat = 2 kappa / (1 + sqrt(1 + 4 kappa^2)).
  ("warped" is the paper's name for this transform of rho_eff.) Numerical
  guards: 0 is returned if a weight norm or |gamma| is below 1e-9, rho_eff is
  clipped to +-(1 - 1e-7), and |kappa| > 1e8 gives sign(kappa).

  Args:
    w_in_i: Effective (BN-folded) input weight vector of neuron i.
    w_in_j: Effective (BN-folded) input weight vector of neuron j.
    gamma_i: Producer BN scale parameter gamma_i of neuron i.
    gamma_j: Producer BN scale parameter gamma_j of neuron j.

  Returns:
    Warped correlation rho_hat_ij between y_i and y_j as a Python float.
  """
  norm_i = float(np.linalg.norm(w_in_i))
  norm_j = float(np.linalg.norm(w_in_j))
  if (
      norm_i < 1e-9
      or norm_j < 1e-9
      or abs(gamma_i) < 1e-9
      or abs(gamma_j) < 1e-9
  ):
    return 0.0
  rho_eff = float(np.dot(w_in_i, w_in_j)) / (norm_i * norm_j)
  rho_eff = float(np.clip(rho_eff, -1.0 + 1e-7, 1.0 - 1e-7))
  std_ratio_i = abs(float(gamma_i)) / norm_i
  std_ratio_j = abs(float(gamma_j)) / norm_j
  kappa = (rho_eff / (1.0 - rho_eff * rho_eff)) * std_ratio_i * std_ratio_j
  if abs(kappa) > 1e8:
    return float(np.sign(kappa) * 1.0)
  return float((2.0 * kappa) / (1.0 + math.sqrt(1.0 + 4.0 * kappa * kappa)))


def _combined_preactivation(
    c1: float,
    c2: float,
    gamma_i: float,
    beta_i: float,
    gamma_j: float,
    beta_j: float,
    rho_hat: float,
) -> Tuple[float, float]:
  """Surrogate (std, mean) of y = c1 y_i + c2 y_j.

  Computes:
    gamma = sqrt(c1^2 gamma_i^2 + c2^2 gamma_j^2
                 + 2 c1 c2 rho_hat |gamma_i| |gamma_j|),
    beta = c1 beta_i + c2 beta_j.

  Args:
    c1: Weight of y_i.
    c2: Weight of y_j.
    gamma_i: Scale parameter gamma_i of y_i.
    beta_i: Shift parameter beta_i of y_i.
    gamma_j: Scale parameter gamma_j of y_j.
    beta_j: Shift parameter beta_j of y_j.
    rho_hat: Surrogate correlation between y_i and y_j.

  Returns:
    Tuple (gamma, beta) of combined pre-activation standard deviation and mean.
  """
  gamma = math.sqrt(
      max(
          0.0,
          (c1 * gamma_i) ** 2
          + (c2 * gamma_j) ** 2
          + 2.0 * c1 * c2 * rho_hat * abs(gamma_i) * abs(gamma_j),
      )
  )
  beta = c1 * beta_i + c2 * beta_j
  return gamma, beta


def _preactivation_correlations(
    c1: float,
    c2: float,
    gamma_u: float,
    gamma_i: float,
    gamma_j: float,
    rho_hat: float,
) -> Tuple[float, float]:
  """Correlations of y_u = c1 y_i + c2 y_j (std gamma_u) with y_i and y_j.

  Computes rho_{u,k} = Cov(y_u, y_k) / (gamma_u |gamma_k|), clipped to [-1, 1].

  Args:
    c1: Weight of y_i in y_u.
    c2: Weight of y_j in y_u.
    gamma_u: Standard deviation of y_u.
    gamma_i: Scale parameter gamma_i of y_i.
    gamma_j: Scale parameter gamma_j of y_j.
    rho_hat: Surrogate correlation between y_i and y_j.

  Returns:
    Tuple (rho_ui, rho_uj) of correlations in [-1, 1].
  """
  preact_cov_ui = c1 * (gamma_i**2) + c2 * rho_hat * abs(gamma_i) * abs(gamma_j)
  preact_cov_uj = c1 * rho_hat * abs(gamma_i) * abs(gamma_j) + c2 * (gamma_j**2)
  rho_ui = float(
      np.clip(preact_cov_ui / (gamma_u * abs(gamma_i) + 1e-12), -1.0, 1.0)
  )
  rho_uj = float(
      np.clip(preact_cov_uj / (gamma_u * abs(gamma_j) + 1e-12), -1.0, 1.0)
  )
  return rho_ui, rho_uj


def merge_pair_geometry(
    i: int,
    j: int,
    joint_weights: np.ndarray,
    fan_in: int,
    producer_bn: Dict[str, np.ndarray],
    return_directions: bool = False,
) -> Dict[str, Any]:
  """Merge geometry of one pair (i, j) on H0-perp, scalar path.

  Used to synthesize the parent of the winning merge (return_directions=True)
  and to refresh the merge-cache row of
  a merged slot. Computes:
    - u_hat, the principal right singular vector of A = W_out W_in^T restricted
    to span(w~_in,i, w~_in,j), via the
      QR factors of W_out = [w_out,i | w_out,j] and W_in = [w~_in,i | w~_in,j];
    - the polarity u_c = argmax_{u in {+u_hat, -u_hat}} ||C(u,i) w_out,i +
    C(u,j) w_out,j|| / sqrt(C(u,u))
      (+u_hat wins ties) and unit output direction v* = (C(u_c,i) w_out,i +
      C(u_c,j) w_out,j) / ||...||;
    - the cached constants a = ||f_i||_o^2 + ||f_j||_o^2 and b = ||C(u_c,i)
    w_out,i + C(u_c,j) w_out,j|| / sqrt(K_self)
      with K_self = C(u_c, u_c);
    - the Frobenius ratio R_F = ||W_in||_F / ||W_out||_F; and
    - psi*, the parent at scale s = 1 in the software gauge w~_in = sqrt(s R_F)
    u_c,
      w_out = sqrt(s / R_F) K_self^(-1/2) v* (splitting K_self^(-1/2)
      symmetrically as K_self^(-1/4) on both factors
      defines the same rank-1 operator).
  The surrogate statistics of psi* are obtained from least-squares span
  coefficients of its input weights (weight
  part only, bias excluded).

  Args:
    i: First neuron index (i < j).
    j: Second neuron index (i < j).
    joint_weights: (N, n + c) joint weight rows (see `read_quadruplet`).
    fan_in: n, length of the input part of a joint weight row.
    producer_bn: Producer BN dict ("gamma", "beta", "mean", "var").
    return_directions: Whether to also return u_c and v*.

  Returns:
    {"is_valid": False} if the pair is degenerate (||W_in||_F or ||W_out||_F
    below 1e-9, or C(u,u) <= 1e-12 for
    both polarities). Otherwise {"is_valid": True} and Python floats:
      k_self                     K_self = C(u_c, u_c);
      frobenius_ratio            R_F;
      sum_sq_capacity            a = ||f_i||_o^2 + ||f_j||_o^2;
      alignment                  b = <psi*, f_i + f_j>_o = ||sum_k C(u_c,k)
      w_out,k|| / sqrt(K_self);
      capacity_i, capacity_j     ||f_i||_o, ||f_j||_o;
      sq_norm_i, sq_norm_j       ||f_i||_o^2, ||f_j||_o^2 (Gram form ||w_out||^2
      v);
      psi_capacity, psi_sq_norm  ||psi*||_o, ||psi*||_o^2 (ideally 1);
      psi_inner_i, psi_inner_j   <psi*, f_i>_o, <psi*, f_j>_o;
      rho_hat                    rho_hat_ij;
      psi_c1, psi_c2             span coefficients of psi*'s effective input
      weights;
    plus, with return_directions, u_c (unit vector in R^(n+1): input weights,
    then bias) and v_star.
  """
  bn_scale_i = producer_bn["gamma"][i] / np.sqrt(
      max(float(producer_bn["var"][i]), 0.0) + BN_EPSILON
  )
  bn_scale_j = producer_bn["gamma"][j] / np.sqrt(
      max(float(producer_bn["var"][j]), 0.0) + BN_EPSILON
  )

  w_in_i, w_in_j = joint_weights[i][:fan_in], joint_weights[j][:fan_in]
  w_out_i, w_out_j = joint_weights[i][fan_in:], joint_weights[j][fan_in:]

  bias_i = float(
      producer_bn["beta"][i] - bn_scale_i * producer_bn["mean"][i]
  )  # folded bias b_i = beta_i - bn_scale_i * mu_i
  bias_j = float(producer_bn["beta"][j] - bn_scale_j * producer_bn["mean"][j])
  gamma_i, beta_i = float(producer_bn["gamma"][i]), float(
      producer_bn["beta"][i]
  )
  gamma_j, beta_j = float(producer_bn["gamma"][j]), float(
      producer_bn["beta"][j]
  )

  w_in_aug_i, w_in_aug_j = np.append(w_in_i, bias_i), np.append(w_in_j, bias_j)
  frob_in = float(
      np.linalg.norm(np.stack([w_in_aug_i, w_in_aug_j], axis=1), "fro")
  )
  frob_out = float(np.linalg.norm(np.stack([w_out_i, w_out_j], axis=1), "fro"))

  if frob_in < 1e-9 or frob_out < 1e-9:
    return {"is_valid": False}

  # u_hat from the rank-2 SVD of R_out R_in^T (the same direction is used under
  # centring).
  w_out_pair = np.column_stack([w_out_i, w_out_j])
  w_in_aug_pair = np.column_stack([w_in_aug_i, w_in_aug_j])
  _, r_out = np.linalg.qr(w_out_pair)
  q_in, r_in = np.linalg.qr(w_in_aug_pair)
  _, _, vh = np.linalg.svd(r_out @ r_in.T)
  u_hat = vh[0] @ q_in.T

  rho_hat = warped_correlation(w_in_i, w_in_j, gamma_i, gamma_j)
  var_i = relu_variance_scalar(abs(gamma_i), beta_i)
  var_j = relu_variance_scalar(abs(gamma_j), beta_j)

  out_sq_i = float(np.dot(w_out_i, w_out_i))
  out_sq_j = float(np.dot(w_out_j, w_out_j))

  sq_norm_i = out_sq_i * var_i
  sq_norm_j = out_sq_j * var_j

  # Polarity check:
  # u_c = argmax_{u in {+u_hat, -u_hat}}
  #   ||C(u,i) w_out,i + C(u,j) w_out,j|| / sqrt(C(u,u)).
  best_score, u_c, k_self, best_v_unnorm = -np.inf, None, None, None

  for sign in [1.0, -1.0]:
    u_cand = sign * u_hat
    c1_u, c2_u = span_coefficients(w_in_i, w_in_j, u_cand[:-1])
    gamma_u, beta_u = _combined_preactivation(
        c1_u, c2_u, gamma_i, beta_i, gamma_j, beta_j, rho_hat
    )

    var_u = relu_variance_scalar(abs(gamma_u), beta_u)  # C(u, u)
    if var_u <= 1e-12:
      continue

    rho_ui, rho_uj = _preactivation_correlations(
        c1_u, c2_u, gamma_u, gamma_i, gamma_j, rho_hat
    )
    cov_ui = relu_covariance_scalar(
        abs(gamma_u), beta_u, abs(gamma_i), beta_i, rho_ui
    )  # C(u, i)
    cov_uj = relu_covariance_scalar(
        abs(gamma_u), beta_u, abs(gamma_j), beta_j, rho_uj
    )

    v_unnorm = cov_ui * w_out_i + cov_uj * w_out_j
    score = float(np.linalg.norm(v_unnorm)) / math.sqrt(var_u)
    if score > best_score:
      best_score, u_c, k_self = score, u_cand, var_u
      best_v_unnorm = v_unnorm

  if u_c is None:
    return {"is_valid": False}

  capacity_i = math.sqrt(max(0.0, out_sq_i)) * math.sqrt(max(0.0, var_i))
  capacity_j = math.sqrt(max(0.0, out_sq_j)) * math.sqrt(max(0.0, var_j))
  a = capacity_i**2 + capacity_j**2

  v_norm = float(np.linalg.norm(best_v_unnorm))
  v_star = best_v_unnorm / (v_norm + 1e-12)
  b = v_norm / math.sqrt(k_self)

  frobenius_ratio = frob_in / frob_out
  psi_w_in_aug = math.sqrt(frobenius_ratio) * u_c
  psi_w_out = (1.0 / math.sqrt(frobenius_ratio)) * (v_star / math.sqrt(k_self))

  psi_c1, psi_c2 = span_coefficients(w_in_i, w_in_j, psi_w_in_aug[:-1])
  psi_gamma, psi_beta = _combined_preactivation(
      psi_c1, psi_c2, gamma_i, beta_i, gamma_j, beta_j, rho_hat
  )

  psi_var = relu_variance_scalar(abs(psi_gamma), psi_beta)

  rho_pi, rho_pj = _preactivation_correlations(
      psi_c1, psi_c2, psi_gamma, gamma_i, gamma_j, rho_hat
  )
  psi_cov_i = relu_covariance_scalar(
      abs(psi_gamma), psi_beta, abs(gamma_i), beta_i, rho_pi
  )
  psi_cov_j = relu_covariance_scalar(
      abs(psi_gamma), psi_beta, abs(gamma_j), beta_j, rho_pj
  )

  result = {
      "is_valid": True,
      "k_self": float(k_self),
      "frobenius_ratio": float(frobenius_ratio),
      "sum_sq_capacity": float(a),
      "alignment": float(b),
      "capacity_i": float(capacity_i),
      "capacity_j": float(capacity_j),
      "psi_capacity": float(
          np.linalg.norm(psi_w_out) * math.sqrt(max(0.0, psi_var))
      ),
      "sq_norm_i": float(sq_norm_i),
      "sq_norm_j": float(sq_norm_j),
      "psi_sq_norm": float(np.dot(psi_w_out, psi_w_out) * psi_var),
      "psi_inner_i": float(np.dot(psi_w_out, w_out_i) * psi_cov_i),
      "psi_inner_j": float(np.dot(psi_w_out, w_out_j) * psi_cov_j),
      "rho_hat": float(rho_hat),
      "psi_c1": float(psi_c1),
      "psi_c2": float(psi_c2),
  }

  if return_directions:
    result["u_c"] = u_c
    result["v_star"] = v_star

  return result


class _Parent(NamedTuple):
  """Physical parameters of a merge parent (see `_synthesize_parent`)."""

  w_in_eff: (
      np.ndarray
  )  # effective (BN-folded) input weights w_in,p^eff (float64)
  bias: float  # effective bias b_p
  w_out: np.ndarray  # outgoing weights w_out,p (flattened consumer slice)
  gamma: float  # gamma_p
  beta: float  # beta_p
  mean: float  # mu_p (BN moving mean)
  var: float  # sigma_p^2 (BN moving variance)
  bn_scale: float  # gamma_p / sqrt(sigma_p^2 + eps)


def _synthesize_parent(
    geometry: Dict[str, Any],
    s_star: float,
    gamma_i: float,
    beta_i: float,
    gamma_j: float,
    beta_j: float,
) -> _Parent:
  """Physical parameters of the merge parent f_p = s* psi*.

  Input/output scaling uses the software gauge w~_in = sqrt(s* R_F) u_c and
  w_out = sqrt(s* / R_F) K_self^(-1/2) v*, instead of splitting K_self^(-1/2)
  symmetrically as K_self^(-1/4) between input and output (same operator and
  capacity). The effective bias b_p is read from the last entry of w~_in (equal
  to c1 b_i + c2 b_j in exact arithmetic when the children's input weight
  vectors are linearly independent). The span coefficients
  (c1, c2) = sqrt(s*) (psi_c1, psi_c2) come from a least-squares fit of the
  weight part (see `merge_pair_geometry`) and give the combined pre-activation
  moments:
    gamma_p = sqrt(c1^2 gamma_i^2 + c2^2 gamma_j^2
                   + 2 c1 c2 rho_hat |gamma_i| |gamma_j|),
    beta_p = c1 beta_i + c2 beta_j.
  The variance is anchored at sigma_p^2 = max(0, gamma_p^2 - eps), giving
  bn_scale_p = gamma_p / sqrt(sigma_p^2 + eps) (= 1 when gamma_p^2 >= eps) and
  mu_p = (beta_p - b_p) / (bn_scale_p + 1e-12). Even when gamma_p^2 < eps, this
  implementation applies the same formulas, protected by the small division
  guards.

  Args:
    geometry: Pair geometry dictionary (with `u_c` and `v_star`) from
      `merge_pair_geometry`.
    s_star: Optimal parent scale s*.
    gamma_i: Producer BN scale gamma_i of child i.
    beta_i: Producer BN shift beta_i of child i.
    gamma_j: Producer BN scale gamma_j of child j.
    beta_j: Producer BN shift beta_j of child j.

  Returns:
    `_Parent` tuple containing the synthesized parent's physical parameters.
  """
  w_in_aug = math.sqrt(s_star * geometry["frobenius_ratio"]) * geometry["u_c"]
  w_out = math.sqrt(s_star / geometry["frobenius_ratio"]) * (
      geometry["v_star"] / math.sqrt(geometry["k_self"])
  )
  w_in_eff, bias = w_in_aug[:-1], w_in_aug[-1]

  c1, c2 = (
      math.sqrt(s_star) * geometry["psi_c1"],
      math.sqrt(s_star) * geometry["psi_c2"],
  )
  gamma, beta = _combined_preactivation(
      c1, c2, gamma_i, beta_i, gamma_j, beta_j, geometry["rho_hat"]
  )
  var = max(0.0, gamma**2 - BN_EPSILON)

  bn_scale = gamma / math.sqrt(var + BN_EPSILON)
  mean = (beta - bias) / (bn_scale + 1e-12)
  return _Parent(
      w_in_eff=w_in_eff,
      bias=bias,
      w_out=w_out,
      gamma=gamma,
      beta=beta,
      mean=mean,
      var=var,
      bn_scale=bn_scale,
  )


# ==============================================================================
# 4. Capacity ledger and decoupled merge cache
# ==============================================================================


def neuron_capacities(
    joint_weights: np.ndarray, fan_in: int, producer_bn: Dict[str, np.ndarray]
) -> np.ndarray:
  """Centred capacities ||f_i||_o = ||w_out,i|| sqrt(v_i) of all layer neurons.

  Evaluated in the dtype of the tensors (float32 for the pretrained ResNet-50).

  Args:
    joint_weights: (N, n + c) joint weight rows of the layer.
    fan_in: Length n of the input part of each joint weight row.
    producer_bn: Producer BN parameter dictionary ("gamma", "beta", "mean",
      "var").

  Returns:
    1D array of centred neuron capacities ||f_i||_o.
  """
  out_norms = np.linalg.norm(joint_weights[:, fan_in:], axis=1)
  variances = relu_variance(np.abs(producer_bn["gamma"]), producer_bn["beta"])
  return out_norms * np.sqrt(np.maximum(0.0, variances))


def build_merge_cache(
    joint_weights: np.ndarray, fan_in: int, producer_bn: Dict[str, np.ndarray]
) -> Dict[str, np.ndarray]:
  """Vectorized merge geometry of all pairs of a layer (float64).

  Builds the decoupled all-pairs merge cache. Evaluates, for all N(N-1)/2 pairs
  i < j at once, the E_a-independent quantities of `merge_pair_geometry`.
  Instead of a rank-2 SVD per pair it uses closed-form 2x2 QR factors of the
  Gram matrices and the eigen-angle of the 2x2 matrix M^T M (M = R_out R_in^T),
  and it evaluates norms and inner products from Gram matrices; the two paths
  are mathematically equivalent but round differently. a and b are evaluated
  twice, in the form used for s* (sum_sq_capacity, alignment) and in the form
  used for the distortion (children_sq_norm, psi_inner_sum).

  Args:
    joint_weights: (N, n + c) joint weight rows of the layer.
    fan_in: Length n of the input part of each joint weight row.
    producer_bn: Producer BN parameter dictionary ("gamma", "beta", "mean",
      "var").

  Returns:
    Symmetric (N, N) arrays indexed by neuron pairs (diagonal False / 0):
      admissible        the pair is a merge candidate (non-degenerate geometry);
      capacity_sum      S = ||f_i||_o + ||f_j||_o;
      sum_sq_capacity   a = ||f_i||_o^2 + ||f_j||_o^2;
      alignment         b = <psi*, f_i + f_j>_o;
      psi_capacity      ||psi*||_o;
      children_sq_norm  ||f_i||_o^2 + ||f_j||_o^2 (Gram form);
      psi_sq_norm       ||psi*||_o^2;
      psi_inner_sum     <psi*, f_i + f_j>_o (Gram form).
  """
  num_neurons = joint_weights.shape[0]
  gamma = np.asarray(producer_bn["gamma"], dtype=np.float64)
  beta = np.asarray(producer_bn["beta"], dtype=np.float64)
  var = np.asarray(producer_bn["var"], dtype=np.float64)
  mean = np.asarray(producer_bn["mean"], dtype=np.float64)
  bn_scale = gamma / np.sqrt(np.maximum(var, 0.0) + BN_EPSILON)
  bias = beta - bn_scale * mean  # folded bias b_i = beta_i - bn_scale_i * mu_i

  joint64 = np.asarray(joint_weights, dtype=np.float64)
  w_in = joint64[:, :fan_in]
  w_out = joint64[:, fan_in:]
  w_in_aug = np.hstack([w_in, bias[:, None]])

  gram_in = w_in @ w_in.T
  gram_in_aug = w_in_aug @ w_in_aug.T
  gram_out = w_out @ w_out.T

  var_all = relu_variance(np.abs(gamma), beta)  # v_i of every neuron
  ii, jj = np.triu_indices(num_neurons, k=1)  # all pairs i < j, row-major

  g11, g22, g12 = gram_in[ii, ii], gram_in[jj, jj], gram_in[ii, jj]
  aug_ii, aug_jj, aug_ij = (
      gram_in_aug[ii, ii],
      gram_in_aug[jj, jj],
      gram_in_aug[ii, jj],
  )
  out_ii, out_jj, out_ij = gram_out[ii, ii], gram_out[jj, jj], gram_out[ii, jj]

  frob_in = np.sqrt(np.maximum(0.0, aug_ii + aug_jj))
  frob_out = np.sqrt(np.maximum(0.0, out_ii + out_jj))
  valid_in_out = (frob_in >= 1e-9) & (frob_out >= 1e-9)
  frobenius_ratio = frob_in / np.maximum(frob_out, 1e-12)

  # 2x2 R factors of W_in = [w~_in,i | w~_in,j] and W_out = [w_out,i | w_out,j]
  # from their Gram matrices.
  r11 = np.sqrt(np.maximum(aug_ii, 1e-24))
  r12 = aug_ij / r11
  r22_sq = aug_jj - np.square(r12)
  r22 = np.sqrt(np.maximum(r22_sq, 1e-24))

  ro11 = np.sqrt(np.maximum(out_ii, 1e-24))
  ro12 = out_ij / ro11
  ro22 = np.sqrt(np.maximum(out_jj - np.square(ro12), 1e-24))

  # M = R_out R_in^T = [[ro11*r11 + ro12*r12, ro12*r22], [ro22*r12, ro22*r22]].
  # Its principal right singular vector (cos theta, sin theta) gives
  # u_hat = W_in R_in^-1 (cos theta, sin theta)
  #       = u_coef_i w~_in,i + u_coef_j w~_in,j.
  m00 = ro11 * r11 + ro12 * r12
  m01 = ro12 * r22
  m10 = ro22 * r12
  m11 = ro22 * r22
  mtm00 = m00 * m00 + m10 * m10
  mtm01 = m00 * m01 + m10 * m11
  mtm11 = m01 * m01 + m11 * m11
  theta = 0.5 * np.arctan2(2.0 * mtm01, mtm00 - mtm11)
  eig0, eig1 = np.cos(theta), np.sin(theta)
  u_coef_j = eig1 / r22
  u_coef_i = (eig0 - r12 * u_coef_j) / r11

  # Span coefficients of the weight part of u_hat (as in `span_coefficients`).
  rhs_i = u_coef_i * g11 + u_coef_j * g12
  rhs_j = u_coef_i * g12 + u_coef_j * g22
  det = g11 * g22 - np.square(g12)
  det_valid = np.abs(det) >= 1e-12
  safe_det = np.where(det_valid, det, 1.0)
  c1_pos = np.where(det_valid, (g22 * rhs_i - g12 * rhs_j) / safe_det, 0.5)
  c2_pos = np.where(det_valid, (g11 * rhs_j - g12 * rhs_i) / safe_det, 0.5)

  # Warped correlation rho_hat_ij (as in `warped_correlation`).
  norm_i, norm_j = np.sqrt(np.maximum(0.0, g11)), np.sqrt(np.maximum(0.0, g22))
  gamma_i, beta_i = gamma[ii], beta[ii]
  gamma_j, beta_j = gamma[jj], beta[jj]
  abs_gi, abs_gj = np.abs(gamma_i), np.abs(gamma_j)
  rho_eff = np.clip(
      g12 / np.maximum(norm_i * norm_j, 1e-24), -1.0 + 1e-7, 1.0 - 1e-7
  )
  std_ratio_i = abs_gi / np.maximum(norm_i, 1e-24)
  std_ratio_j = abs_gj / np.maximum(norm_j, 1e-24)
  kappa = (rho_eff / (1.0 - np.square(rho_eff))) * std_ratio_i * std_ratio_j
  rho_hat = np.where(
      (norm_i < 1e-9) | (norm_j < 1e-9) | (abs_gi < 1e-9) | (abs_gj < 1e-9),
      0.0,
      np.where(
          np.abs(kappa) > 1e8,
          np.sign(kappa) * 1.0,
          (2.0 * kappa) / (1.0 + np.sqrt(1.0 + 4.0 * np.square(kappa))),
      ),
  )

  var_i, var_j = var_all[ii], var_all[jj]

  sq_norm_i, sq_norm_j = out_ii * var_i, out_jj * var_j

  # Polarity check:
  # u_c = argmax_{u in {+u_hat, -u_hat}}
  #   ||C(u,i) w_out,i + C(u,j) w_out,j|| / sqrt(C(u,u)); +u_hat wins ties.
  best_score = np.full(len(ii), -np.inf)
  best_sign = np.ones(len(ii))
  k_self = np.zeros(len(ii))
  best_v_norm = np.zeros(len(ii))
  best_cov_ui = np.zeros(len(ii))
  best_cov_uj = np.zeros(len(ii))

  for sign in [1.0, -1.0]:
    c1_u = np.where(det_valid, sign * c1_pos, 0.5)
    c2_u = np.where(det_valid, sign * c2_pos, 0.5)
    # (|gamma_u|, beta_u) of y_u = c1 y_i + c2 y_j and its correlations with
    # y_i, y_j (as in `_combined_preactivation` and
    # `_preactivation_correlations`).
    gamma_u = np.sqrt(
        np.maximum(
            0.0,
            np.square(c1_u * gamma_i)
            + np.square(c2_u * gamma_j)
            + 2.0 * c1_u * c2_u * rho_hat * abs_gi * abs_gj,
        )
    )
    beta_u = c1_u * beta_i + c2_u * beta_j
    var_u = relu_variance(np.abs(gamma_u), beta_u)  # C(u, u)

    preact_cov_ui = c1_u * np.square(gamma_i) + c2_u * rho_hat * abs_gi * abs_gj
    preact_cov_uj = c1_u * rho_hat * abs_gi * abs_gj + c2_u * np.square(gamma_j)
    rho_ui = np.clip(preact_cov_ui / (gamma_u * abs_gi + 1e-12), -1.0, 1.0)
    rho_uj = np.clip(preact_cov_uj / (gamma_u * abs_gj + 1e-12), -1.0, 1.0)

    cov_ui = relu_covariance(
        np.abs(gamma_u), beta_u, abs_gi, beta_i, rho_ui
    )  # C(u, i)
    cov_uj = relu_covariance(np.abs(gamma_u), beta_u, abs_gj, beta_j, rho_uj)

    # ||C(u,i) w_out,i + C(u,j) w_out,j|| from the output Gram entries.
    v_norm = np.sqrt(
        np.maximum(
            0.0,
            np.square(cov_ui) * out_ii
            + np.square(cov_uj) * out_jj
            + 2.0 * cov_ui * cov_uj * out_ij,
        )
    )
    score = np.where(
        var_u > 1e-12, v_norm / np.sqrt(np.maximum(var_u, 1e-24)), -np.inf
    )
    better = score > best_score
    best_score = np.where(better, score, best_score)
    best_sign = np.where(better, sign, best_sign)
    k_self = np.where(better, var_u, k_self)
    best_v_norm = np.where(better, v_norm, best_v_norm)
    best_cov_ui = np.where(better, cov_ui, best_cov_ui)
    best_cov_uj = np.where(better, cov_uj, best_cov_uj)

  admissible = valid_in_out & (k_self > 1e-12)
  capacity_i = np.sqrt(np.maximum(0.0, out_ii)) * np.sqrt(
      np.maximum(0.0, var_i)
  )
  capacity_j = np.sqrt(np.maximum(0.0, out_jj)) * np.sqrt(
      np.maximum(0.0, var_j)
  )
  a = np.square(capacity_i) + np.square(capacity_j)
  b = best_v_norm / np.sqrt(np.maximum(k_self, 1e-24))

  # psi*, the parent at s = 1 (gauge of `merge_pair_geometry`).
  sqrt_rf = np.sqrt(frobenius_ratio)
  inv_sqrt_rf = 1.0 / np.maximum(sqrt_rf, 1e-12)
  psi_c1 = np.where(det_valid, sqrt_rf * best_sign * c1_pos, 0.5)
  psi_c2 = np.where(det_valid, sqrt_rf * best_sign * c2_pos, 0.5)
  psi_gamma = np.sqrt(
      np.maximum(
          0.0,
          np.square(psi_c1 * gamma_i)
          + np.square(psi_c2 * gamma_j)
          + 2.0 * psi_c1 * psi_c2 * rho_hat * abs_gi * abs_gj,
      )
  )
  psi_beta = psi_c1 * beta_i + psi_c2 * beta_j
  psi_var = relu_variance(np.abs(psi_gamma), psi_beta)

  preact_cov_pi = (
      psi_c1 * np.square(gamma_i) + psi_c2 * rho_hat * abs_gi * abs_gj
  )
  preact_cov_pj = psi_c1 * rho_hat * abs_gi * abs_gj + psi_c2 * np.square(
      gamma_j
  )
  rho_pi = np.clip(preact_cov_pi / (psi_gamma * abs_gi + 1e-12), -1.0, 1.0)
  rho_pj = np.clip(preact_cov_pj / (psi_gamma * abs_gj + 1e-12), -1.0, 1.0)

  psi_cov_i = relu_covariance(
      np.abs(psi_gamma), psi_beta, abs_gi, beta_i, rho_pi
  )
  psi_cov_j = relu_covariance(
      np.abs(psi_gamma), psi_beta, abs_gj, beta_j, rho_pj
  )

  psi_out_sq = 1.0 / np.maximum(
      frobenius_ratio * k_self, 1e-24
  )  # ||w_out(psi*)||^2 = 1 / (R_F K_self)
  psi_capacity = np.sqrt(psi_out_sq) * np.sqrt(np.maximum(0.0, psi_var))
  psi_sq_norm = psi_out_sq * psi_var

  v_scale = inv_sqrt_rf / (
      np.sqrt(np.maximum(k_self, 1e-24)) * (best_v_norm + 1e-12)
  )
  psi_out_dot_i = v_scale * (
      best_cov_ui * out_ii + best_cov_uj * out_ij
  )  # <w_out(psi*), w_out,i>
  psi_out_dot_j = v_scale * (best_cov_ui * out_ij + best_cov_uj * out_jj)
  psi_inner_i = psi_out_dot_i * psi_cov_i
  psi_inner_j = psi_out_dot_j * psi_cov_j

  def symmetric(values, fill=0.0, dtype=np.float64):
    mat = np.full((num_neurons, num_neurons), fill, dtype=dtype)
    mat[ii, jj] = values
    mat[jj, ii] = values
    return mat

  return {
      "admissible": symmetric(admissible, fill=False, dtype=bool),
      "capacity_sum": symmetric(capacity_i + capacity_j),
      "sum_sq_capacity": symmetric(a),
      "alignment": symmetric(b),
      "psi_capacity": symmetric(psi_capacity),
      "children_sq_norm": symmetric(sq_norm_i + sq_norm_j),
      "psi_sq_norm": symmetric(psi_sq_norm),
      "psi_inner_sum": symmetric(psi_inner_i + psi_inner_j),
  }


def update_merge_cache_row(
    merge_cache: Dict[str, np.ndarray],
    i: int,
    neighbors: List[int],
    joint_weights: np.ndarray,
    fan_in: int,
    producer_bn: Dict[str, np.ndarray],
) -> None:
  """Recomputes row/column i of a merge cache with the scalar geometry.

  Used after a merge (i, j) -> i. Pairs that turn degenerate are only marked
  inadmissible (their other entries keep stale values, which are never read).

  Args:
    merge_cache: All-pairs merge geometry cache dictionary to update in place.
    i: Neuron index whose row and column are recomputed.
    neighbors: Live neuron indices other than `i`.
    joint_weights: (N, n + c) joint weight rows of the layer.
    fan_in: Length n of the input part of each joint weight row.
    producer_bn: Producer BN parameter dictionary ("gamma", "beta", "mean",
      "var").
  """
  merge_cache["admissible"][i, :] = False
  merge_cache["admissible"][:, i] = False
  for k in neighbors:
    lo, hi = (i, k) if i < k else (k, i)
    geometry = merge_pair_geometry(
        lo, hi, joint_weights, fan_in, producer_bn, return_directions=False
    )
    if not geometry.get("is_valid", False):
      continue
    for row, col in [(lo, hi), (hi, lo)]:
      merge_cache["admissible"][row, col] = True
      merge_cache["capacity_sum"][row, col] = (
          geometry["capacity_i"] + geometry["capacity_j"]
      )
      merge_cache["sum_sq_capacity"][row, col] = geometry["sum_sq_capacity"]
      merge_cache["alignment"][row, col] = geometry["alignment"]
      merge_cache["psi_capacity"][row, col] = geometry["psi_capacity"]
      merge_cache["children_sq_norm"][row, col] = (
          geometry["sq_norm_i"] + geometry["sq_norm_j"]
      )
      merge_cache["psi_sq_norm"][row, col] = geometry["psi_sq_norm"]
      merge_cache["psi_inner_sum"][row, col] = (
          geometry["psi_inner_i"] + geometry["psi_inner_j"]
      )


class CostManager:
  """Capacity ledger and decoupled merge cache of the encoder.

  Per producer layer, keyed by its convolution name in registration order:
    active_neurons    set of live neuron indices (N = its size);
    live_capacity     E_a = sum_{k in active} ||f_k||_o, the sum of the live
    centred capacities; updated in O(1)
                      after each prune or merge in the layer, and set to
                      CAPACITY_FLOOR when the block is evicted;
    initial_capacity  E_init, the uncompressed capacity anchor used in J_evict =
    sum_l N_l (E_a^(l) / E_init^(l));
    neuron_capacity   ||f_i||_o of every neuron (0 once eliminated);
    merge_cache       all-pairs merge geometry (see `build_merge_cache`).
  """

  def __init__(self) -> None:
    self.active_neurons: Dict[str, Set[int]] = {}
    self.live_capacity: Dict[str, float] = {}
    self.initial_capacity: Dict[str, float] = {}
    self.neuron_capacity: Dict[str, np.ndarray] = {}
    self.merge_cache: Dict[str, Dict[str, np.ndarray]] = {}

  def register_layer(
      self,
      layer: str,
      num_neurons: int,
      initial_capacity: float,
      neuron_capacity: np.ndarray,
      merge_cache: Dict[str, np.ndarray],
  ) -> None:
    """Registers a layer with all neurons live and anchors E_init = E_a."""
    self.active_neurons[layer] = set(range(num_neurons))
    self.live_capacity[layer] = float(initial_capacity)
    self.initial_capacity[layer] = float(initial_capacity)
    self.neuron_capacity[layer] = neuron_capacity.copy()
    self.merge_cache[layer] = merge_cache

  def remove_neuron(self, layer: str, idx: int) -> None:
    """Removes a neuron from the live set; no-op if it is not live."""
    if idx in self.active_neurons[layer]:
      self.active_neurons[layer].remove(idx)

  def active_neighbors(self, layer: str, exclude_idx: int) -> List[int]:
    """Live neurons of `layer` other than `exclude_idx`, in increasing order."""
    return [i for i in sorted(self.active_neurons[layer]) if i != exclude_idx]


@dataclasses.dataclass
class _EncoderState:
  """Mutable state of one `hope_encode` run."""

  weights: Dict[
      str, np.ndarray
  ]  # live tensors (copies of the input), written by the actions
  quadruplet_of: Dict[
      str, Quadruplet
  ]  # producer convolution name -> its quadruplet
  # Producer convolution name -> {"joint_weights", "fan_in", "producer_bn"}
  # (see `read_quadruplet`): the cached layer state from which actions are
  # scored and synthesized, updated locally in O(1) / O(N) after each action.
  layer_cache: Dict[str, Dict[str, Any]]
  cost_manager: CostManager
  curriculum: List[
      InstructionPacket
  ]  # one packet per executed action, in encoding order
  feature_map_sizes: Optional[Mapping[str, float]] = None


# ==============================================================================
# 5. Encoder
# ==============================================================================


def _write_density_accuracy_csv(
    path: str, rows: Sequence[Tuple[float, float]]
) -> None:
  """Writes `(density, accuracy)` rows to `path` in two-column CSV format."""
  with open(path, "w") as f:
    f.write("density,accuracy\n")
    for density, accuracy in rows:
      f.write(f"{float(density):.6f},{float(accuracy):.6f}\n")


def hope_encode(
    weights: Dict[str, np.ndarray],
    quadruplets: Sequence[Quadruplet],
    eval_fn: Optional[Callable[[Dict[str, np.ndarray]], float]] = None,
    eval_stride: int = 10,
    output_file: Optional[str] = None,
    feature_map_sizes: Optional[Mapping[str, float]] = None,
) -> Tuple[
    Dict[str, np.ndarray], List[InstructionPacket], List[Tuple[float, float]]
]:
  """Runs the HOPE progressive encoder until no admissible action is left.

  There is no target density: for finite weights, the loop ends when every
  registered layer has been emptied (by
  prunes, merges or block evictions). Intermediate models are recovered by
  decoding part of the curriculum
  (`hope_decode`).

  Args:
    weights: tensor name -> array, Keras names such as
      "conv2_block1_1_conv/kernel" or "conv2_block1_1_bn/gamma". Not modified.
    quadruplets: the quadruplets to compress, e.g.
      `find_bottleneck_quadruplets(weights)`. Layers are registered and scanned
      in this order; quadruplets whose producer kernel is missing are skipped.
    eval_fn: optional callback, called every `eval_stride` steps with the live
      compressed weights (which it must not modify); its return value is
      recorded in the history.
    eval_stride: evaluation period, in encoding steps.
    output_file: optional path to save `(density, accuracy)` pairs as a
      two-column CSV (`density,accuracy`). When provided (non-empty string) with
      `eval_stride > 0`, records the initial network (`density = 1.0`), every
      `eval_stride` steps, and the final compressed network.
    feature_map_sizes: optional mapping from consumer layer name (or 5-character
      prefix) to spatial feature-map resolution H for 3x3 zero-padding tap
      fractions; defaults to `RESNET50_FEATURE_MAP_SIZE`.

  Returns:
    (compressed_weights, curriculum, history): the fully compressed weights; the
    InstructionPackets in decoding
    order (last encoder action first), so that `hope_decode(compressed_weights,
    curriculum, k)` undoes the last k
    actions; and the (density, eval_fn value) pairs recorded every `eval_stride`
    steps, with density = number of
    nonzero entries over all tensors / its initial value (the value is 0.0 when
    eval_fn is None).

  Raises:
    ValueError: If a registered neuron's BN scale gamma / sqrt(var + eps)
      is zero while its raw producer filter is nonzero: the neuron's folded
      filter row is identically zero, so no PRUNE or MERGE packet could
      restore the filter and the returned curriculum would not decode back
      to the input weights.
  """
  t_start = time.time()
  state = _EncoderState(
      weights={k: v.copy() for k, v in weights.items()},
      quadruplet_of={q[0]: q for q in quadruplets},
      layer_cache={},
      cost_manager=CostManager(),
      curriculum=[],
      feature_map_sizes=feature_map_sizes,
  )
  initial_nonzeros = sum(np.count_nonzero(v) for v in weights.values())

  num_neurons = _initialize_caches(state, quadruplets)
  logger.info(
      "HOPE encoder initialization complete in %.2fs (%d neurons).",
      time.time() - t_start,
      num_neurons,
  )

  step = 0
  history: List[Tuple[float, float]] = []
  csv_path = output_file if (output_file and eval_stride > 0) else None
  csv_rows: List[Tuple[float, float]] = []
  if csv_path is not None:
    initial_val = float(eval_fn(state.weights)) if eval_fn is not None else 0.0
    csv_rows.append((1.0, initial_val))
    _write_density_accuracy_csv(csv_path, csv_rows)

  while True:
    best_action = _greedy_scan(state)
    if best_action is None:
      break

    action, conv_a, indices = best_action
    if action == "MERGE":
      _execute_merge(state, conv_a, indices[0], indices[1], indices[2])
    elif action == "PRUNE":
      _execute_prune(state, conv_a, indices[0])
    elif action == "EVICT":
      _execute_evict(state, *indices)

    step += 1

    if eval_stride > 0 and step % eval_stride == 0:
      nonzeros = sum(np.count_nonzero(v) for v in state.weights.values())
      density = float(nonzeros) / float(initial_nonzeros)
      value = float(eval_fn(state.weights)) if eval_fn is not None else 0.0
      history.append((np.float64(density), np.float64(value)))
      if csv_path is not None:
        csv_rows.append((density, value))
        _write_density_accuracy_csv(csv_path, csv_rows)

  if csv_path is not None:
    if step % eval_stride != 0:
      nonzeros = sum(np.count_nonzero(v) for v in state.weights.values())
      final_density = float(nonzeros) / float(initial_nonzeros)
      final_val = float(eval_fn(state.weights)) if eval_fn is not None else 0.0
      csv_rows.append((final_density, final_val))
    _write_density_accuracy_csv(csv_path, csv_rows)

  state.curriculum.reverse()
  return state.weights, state.curriculum, history


def _check_zero_bn_scale(
    weights: Dict[str, np.ndarray],
    quadruplet: Quadruplet,
    producer_bn: Dict[str, np.ndarray],
) -> None:
  """Rejects registered neurons whose zero BN scale makes encoding lossy.

  A neuron's BN-folded filter row is (gamma / sqrt(max(var, 0) + eps)) times
  its raw filter, so a zero BN scale folds the row to zero however large the
  raw filter is. PRUNE and MERGE packets store only the folded row, so
  `hope_decode_step` cannot recover such a neuron's filter (see its
  docstring): without this check `hope_encode` would silently return a
  curriculum that does not decode back to the input weights. A zero-scale
  neuron whose raw filter is already zero loses nothing and is accepted.

  Args:
    weights: The encoder's live weights (not modified).
    quadruplet: The quadruplet being registered.
    producer_bn: Producer BN tensors of the quadruplet, as returned by
      `read_quadruplet`.

  Raises:
    ValueError: If a neuron of the quadruplet has a zero BN scale and a
      nonzero raw producer filter.
  """
  conv_a, bn_a, _, _ = quadruplet
  gamma = np.asarray(producer_bn["gamma"], dtype=np.float64)
  var = np.asarray(producer_bn["var"], dtype=np.float64)
  bn_scale = gamma / np.sqrt(np.maximum(var, 0.0) + BN_EPSILON)
  zero_scale = np.flatnonzero(bn_scale == 0.0)
  if zero_scale.size == 0:
    return
  kernel_a = weights[f"{conv_a}/kernel"]
  kernel_a_4d = (
      kernel_a
      if kernel_a.ndim == 4
      else kernel_a.reshape(1, 1, kernel_a.shape[0], kernel_a.shape[1])
  )
  for i in zero_scale:
    num_nonzero = int(np.count_nonzero(kernel_a_4d[..., int(i)]))
    if num_nonzero:
      raise ValueError(
          f"hope_encode cannot losslessly encode neuron {int(i)} of"
          f" '{conv_a}' (batch norm '{bn_a}'): its BN scale"
          " gamma / sqrt(var + eps) is zero"
          f" (gamma = {float(gamma[i])}), so its BN-folded filter row is"
          " identically zero and a PRUNE or MERGE packet cannot restore"
          f" its raw filter ({num_nonzero} nonzero entries would be lost"
          " on decode). Give the neuron a nonzero BN gamma, or exclude"
          " this quadruplet, before encoding."
      )


def _initialize_caches(
    state: _EncoderState, quadruplets: Sequence[Quadruplet]
) -> int:
  """Initializes neuron capacities ||f_i||_o, E_init, and layer merge caches.

  Registers every present producer layer of `quadruplets` in `state`.

  Args:
    state: Mutable encoder state to populate.
    quadruplets: Bottleneck quadruplets to register in order.

  Returns:
    Total number of registered neurons across all layers.
  """
  total_neurons = 0
  for quadruplet in quadruplets:
    conv_a, _, _, _ = quadruplet
    if f"{conv_a}/kernel" not in state.weights:
      continue

    joint_weights, fan_in, producer_bn = read_quadruplet(
        state.weights, quadruplet
    )
    _check_zero_bn_scale(state.weights, quadruplet, producer_bn)
    num_neurons = joint_weights.shape[0]
    total_neurons += num_neurons

    capacities = neuron_capacities(joint_weights, fan_in, producer_bn)
    initial_capacity = float(np.sum(capacities))

    merge_cache = build_merge_cache(joint_weights, fan_in, producer_bn)

    state.cost_manager.register_layer(
        conv_a, num_neurons, initial_capacity, capacities, merge_cache
    )
    state.layer_cache[conv_a] = {
        "joint_weights": joint_weights.copy(),
        "fan_in": fan_in,
        "producer_bn": copy.deepcopy(producer_bn),
    }
  return total_neurons


def _greedy_scan(
    state: _EncoderState,
) -> Optional[Tuple[str, str, Tuple[Any, ...]]]:
  """Finds the admissible action with smallest distortion rate DR = J / dP_init.

  Evaluates DR(a) = J(a) / dP_init(a). Layers are visited in registration order
  and, within a layer, prune before merge before evict. A candidate replaces the
  incumbent only if its DR is strictly smaller, so ties go to the earlier layer
  and action type and, within a type, to the lowest neuron index (pair). The
  static footprint dP_init = n + c + 4 per neuron is taken from the tensor
  shapes (convolution biases are not counted), which equals the initial nonzero
  parameter count when the initial weights contain no exact zeros.

  Args:
    state: Current mutable encoder state.

  Returns:
    ("PRUNE", conv_a, (i,)), ("MERGE", conv_a, (i, j, s_star)),
    ("EVICT", conv_1, (conv_1, conv_2, conv_3)), or None if no action is
    admissible.
  """
  cost_manager = state.cost_manager
  best_rate = np.inf
  best_action = None

  for layer, active_set in cost_manager.active_neurons.items():
    n_live = len(active_set)
    if n_live == 0:
      continue
    active_idx = np.array(sorted(list(active_set)), dtype=np.int64)

    delta_p = (
        state.layer_cache[layer]["joint_weights"].shape[1]
        + _BN_PARAMS_PER_NEURON
    )
    live_capacity = cost_manager.live_capacity[layer]

    rate, i = _prune_candidate(
        cost_manager.neuron_capacity[layer],
        active_idx,
        live_capacity,
        n_live,
        delta_p,
    )
    if rate < best_rate:
      best_rate = float(rate)
      best_action = ("PRUNE", layer, (i,))

    if n_live >= 2:
      candidate = _merge_candidate(
          cost_manager.merge_cache[layer],
          active_idx,
          live_capacity,
          n_live,
          delta_p,
      )
      if candidate is not None:
        rate, i, j, s_star = candidate
        if rate < best_rate:
          best_rate = float(rate)
          best_action = ("MERGE", layer, (i, j, s_star))

    if "_1_conv" in layer:
      candidate = _evict_candidate(state, layer)
      if candidate is not None:
        rate, block = candidate
        if rate < best_rate:
          best_rate = float(rate)
          best_action = ("EVICT", block[0], block)

  return best_action


def _prune_candidate(
    neuron_capacity: np.ndarray,
    active_idx: np.ndarray,
    live_capacity: float,
    n_live: int,
    delta_p: int,
) -> Tuple[np.floating, int]:
  """Cheapest PRUNE candidate of a layer.

  Evaluates J_prune(i) = N ||f_i||_o / (E_a - ||f_i||_o) and
  DR = J_prune / dP_init over live neurons.

  Args:
    neuron_capacity: Per-neuron centred capacities ||f_i||_o of the layer.
    active_idx: Sorted 1D integer array of live neuron indices.
    live_capacity: Current live centred capacity E_a of the layer.
    n_live: Number of live neurons N in the layer.
    delta_p: Static parameter footprint dP_init per neuron.

  Returns:
    Tuple (rate, neuron) where `rate` is a NumPy scalar in the dtype of the
    capacities (float32 for ResNet-50) and `neuron` is the victim index.
  """
  capacity = neuron_capacity[active_idx]
  e_post = np.maximum(live_capacity - capacity, CAPACITY_FLOOR)
  cost = (capacity * n_live) / e_post
  rate = cost / delta_p
  k = int(np.argmin(rate))
  return rate[k], int(active_idx[k])


def _merge_candidate(
    merge_cache: Dict[str, np.ndarray],
    active_idx: np.ndarray,
    live_capacity: float,
    n_live: int,
    delta_p: int,
) -> Optional[Tuple[np.floating, int, int, float]]:
  """Cheapest MERGE of a layer, evaluated from the cache in O(1) per pair.

  For every admissible pair i < j of live neurons: E_rem = max(E_a - S, eps),
  s* = min(max(0, (a + b E_rem) / (2 E_rem + b)), S) (pairs with
  2 E_rem + b <= 1e-12 or a + b E_rem <= 0 are skipped), f_p = s* psi*,
  D^2 = ||f_i - f_p||_o^2 + ||f_j - f_p||_o^2 and
  J_merge = N D / max(E_a - S + ||f_p||_o, eps).

  Args:
    merge_cache: Decoupled all-pairs merge cache of the layer.
    active_idx: Sorted 1D integer array of live neuron indices.
    live_capacity: Current live centred capacity E_a of the layer.
    n_live: Number of live neurons N in the layer.
    delta_p: Static parameter footprint dP_init per neuron.

  Returns:
    (rate, i, j, s_star) of the cheapest pair, or None if no pair qualifies.
  """
  rel_i, rel_j = np.triu_indices(n_live, k=1)
  rows, cols = active_idx[rel_i], active_idx[rel_j]
  admissible = merge_cache["admissible"][rows, cols]
  if not np.any(admissible):
    return None
  pos = np.nonzero(admissible)[0]
  rows, cols = rows[pos], cols[pos]

  capacity_sum = merge_cache["capacity_sum"][rows, cols]
  e_rem = np.maximum(live_capacity - capacity_sum, CAPACITY_FLOOR)
  a = merge_cache["sum_sq_capacity"][rows, cols]
  b = merge_cache["alignment"][rows, cols]

  s_den = 2.0 * e_rem + b
  s_num = a + b * e_rem
  s_ok = (s_den > 1e-12) & (s_num > 0.0)
  if not np.any(s_ok):
    return None
  pos = np.nonzero(s_ok)[0]
  rows, cols = rows[pos], cols[pos]
  capacity_sum = capacity_sum[pos]
  s_num = s_num[pos]
  s_den = s_den[pos]

  s_star = np.minimum(np.maximum(0.0, s_num / s_den), capacity_sum)
  parent_sq_norm = np.square(s_star) * merge_cache["psi_sq_norm"][rows, cols]
  parent_inner = s_star * merge_cache["psi_inner_sum"][rows, cols]
  children_sq_norm = merge_cache["children_sq_norm"][rows, cols]

  distortion_sq = np.maximum(
      0.0, 2.0 * parent_sq_norm + children_sq_norm - 2.0 * parent_inner
  )

  distortion = np.sqrt(distortion_sq)
  parent_capacity = s_star * merge_cache["psi_capacity"][rows, cols]
  e_post = np.maximum(
      live_capacity - capacity_sum + parent_capacity, CAPACITY_FLOOR
  )
  cost = (distortion * n_live) / e_post
  rate = cost / delta_p

  k = int(np.argmin(rate))
  return rate[k], int(rows[k]), int(cols[k]), float(s_star[k])


def _evict_candidate(
    state: _EncoderState, layer: str
) -> Optional[Tuple[float, Tuple[str, str, str]]]:
  """Eviction of the bottleneck block whose first producer layer is `layer`.

  Evaluates J_evict = sum_{l in {1,2}} N_l (E_a^(l) / E_init^(l)) over the
  block's two internal layers and
  dP_evict = sum_{l in {1,2}} N_l dP_l^init + 3 d_amb, returning
  DR = J_evict / dP_evict. Requires the block's second producer layer to be
  registered and its X_3 kernel to exist.

  Args:
    state: Current mutable encoder state.
    layer: Name of the block's first producer convolution (`*_1_conv`).

  Returns:
    (rate, (conv_1, conv_2, conv_3)), or None if the block cannot be evicted.
  """
  cost_manager = state.cost_manager
  block_prefix = layer.rsplit("_", 2)[0]
  conv_1 = f"{block_prefix}_1_conv"
  conv_2 = f"{block_prefix}_2_conv"
  conv_3 = f"{block_prefix}_3_conv"
  if (
      conv_2 not in cost_manager.active_neurons
      or f"{conv_3}/kernel" not in state.weights
  ):
    return None
  active_1 = cost_manager.active_neurons[conv_1]
  active_2 = cost_manager.active_neurons[conv_2]
  if len(active_1) == 0:
    return None

  cost_1 = len(active_1) * (
      cost_manager.live_capacity[conv_1]
      / max(cost_manager.initial_capacity[conv_1], CAPACITY_FLOOR)
  )
  cost_2 = len(active_2) * (
      cost_manager.live_capacity[conv_2]
      / max(cost_manager.initial_capacity[conv_2], CAPACITY_FLOOR)
  )
  cost = cost_1 + cost_2

  delta_p_1 = (
      state.layer_cache[conv_1]["joint_weights"].shape[1]
      + _BN_PARAMS_PER_NEURON
  )
  delta_p_2 = (
      state.layer_cache[conv_2]["joint_weights"].shape[1]
      + _BN_PARAMS_PER_NEURON
  )
  d_amb = state.weights[f"{conv_3}/kernel"].shape[-1]
  delta_p = (
      (len(active_1) * delta_p_1)
      + (len(active_2) * delta_p_2)
      + _BN3_PARAMS_REMOVED_PER_CHANNEL * d_amb
  )
  return cost / delta_p, (conv_1, conv_2, conv_3)


def _neuron_capacity_scalar(
    gamma_abs: float, beta: float, w_out: np.ndarray
) -> float:
  """Scalar centred capacity ||f||_o = sqrt(max(0, ||w_out||^2 v))."""
  return math.sqrt(
      max(
          0.0,
          float(np.dot(w_out, w_out)) * relu_variance_scalar(gamma_abs, beta),
      )
  )


def _execute_merge(
    state: _EncoderState, conv_a: str, i: int, j: int, s_star: float
) -> None:
  """Executes MERGE (i, j) -> i and updates row/col i locally in O(N).

  Writes the synthesized parent into slot i, zeroes slot j, compensates the
  consumer BN mean, records an `InstructionPacket`, and refreshes the ledger.

  Args:
    state: Current mutable encoder state.
    conv_a: Producer convolution layer name.
    i: Surviving neuron index (i < j) receiving the synthesized parent.
    j: Eliminated neuron index (i < j) to zero out.
    s_star: Optimal parent scale s*.
  """
  quadruplet = state.quadruplet_of[conv_a]
  _, bn_a, conv_b, bn_b = quadruplet
  cached = state.layer_cache[conv_a]
  weights = state.weights
  has_consumer_bn = bn_b is not None and f"{bn_b}/moving_mean" in weights
  consumer_mean_before = (
      weights[f"{bn_b}/moving_mean"].copy() if has_consumer_bn else None
  )

  live_joint, _, _ = read_quadruplet(weights, quadruplet)
  row_i_before = live_joint[i].copy()
  row_j_before = live_joint[j].copy()

  geometry = merge_pair_geometry(
      i,
      j,
      cached["joint_weights"],
      cached["fan_in"],
      cached["producer_bn"],
      return_directions=True,
  )

  producer_bn = cached["producer_bn"]
  gamma_i, beta_i = float(producer_bn["gamma"][i]), float(
      producer_bn["beta"][i]
  )
  gamma_j, beta_j = float(producer_bn["gamma"][j]), float(
      producer_bn["beta"][j]
  )
  parent = _synthesize_parent(
      geometry, s_star, gamma_i, beta_i, gamma_j, beta_j
  )

  producer_bn_before = (
      weights[f"{bn_a}/gamma"][i],
      weights[f"{bn_a}/beta"][i],
      weights[f"{bn_a}/moving_mean"][i],
      weights[f"{bn_a}/moving_variance"][i],
      weights[f"{bn_a}/gamma"][j],
      weights[f"{bn_a}/beta"][j],
      weights[f"{bn_a}/moving_mean"][j],
      weights[f"{bn_a}/moving_variance"][j],
  )

  kernel_a, kernel_b = weights[f"{conv_a}/kernel"], weights[f"{conv_b}/kernel"]
  kernel_b_4d = (
      kernel_b
      if kernel_b.ndim == 4
      else kernel_b.reshape(1, 1, kernel_b.shape[0], kernel_b.shape[1])
  )
  w_out = (
      parent.w_out
      if np.issubdtype(kernel_b.dtype, np.floating)
      else parent.w_out.astype(np.float64)
  )
  w_out_taps = w_out.reshape(
      kernel_b_4d.shape[0:2] + (-1,)
  )  # parent tap weights w'_{t,p}

  # P1a: mu^b += sum_t chi_t (m_p w'_{t,p} - m_i w_{t,i} - m_j w_{t,j}).
  if has_consumer_bn:
    m_p = relu_mean_scalar(abs(parent.gamma), parent.beta)
    m_i = relu_mean_scalar(abs(gamma_i), beta_i)
    m_j = relu_mean_scalar(abs(gamma_j), beta_j)
    chi = consumer_tap_fractions(kernel_b_4d, conv_b, state.feature_map_sizes)
    tap_mean_change = (
        m_p * w_out_taps
        - m_i * kernel_b_4d[:, :, i, :]
        - m_j * kernel_b_4d[:, :, j, :]
    )
    _compensate_consumer_mean(
        state, quadruplet, np.sum(chi * tap_mean_change, axis=(0, 1))
    )

  # Raw parent kernel by inverting the BN folding
  # w_in,p^eff = bn_scale_p * w_raw.
  w_raw = parent.w_in_eff / (parent.bn_scale + RAW_WEIGHT_DIV_GUARD)
  kernel_a[..., i] = w_raw.reshape(kernel_a.shape[:-1])
  kernel_a[..., j] = 0.0
  if kernel_b.ndim == 4:
    kernel_b[:, :, i, :] = w_out_taps
    kernel_b[:, :, j, :] = 0.0
  else:
    kernel_b[i, :] = w_out_taps.reshape(-1)
    kernel_b[j, :] = 0.0

  weights[f"{bn_a}/gamma"][i] = parent.gamma
  weights[f"{bn_a}/beta"][i] = parent.beta
  weights[f"{bn_a}/moving_mean"][i] = parent.mean
  weights[f"{bn_a}/moving_variance"][i] = parent.var
  for param in ["gamma", "beta", "moving_mean", "moving_variance"]:
    weights[f"{bn_a}/{param}"][j] = 0.0

  cost_manager = state.cost_manager
  n_active_before = len(cost_manager.active_neurons[conv_a])
  combined_norm = float(geometry["capacity_i"] + geometry["capacity_j"])
  cap_p_scaled = float(s_star * geometry["psi_capacity"])
  remaining_eb = float(
      max(
          cost_manager.live_capacity[conv_a] - combined_norm + cap_p_scaled,
          CAPACITY_FLOOR,
      )
  )
  cost_manager.remove_neuron(conv_a, j)
  cost_manager.merge_cache[conv_a]["admissible"][j, :] = False
  cost_manager.merge_cache[conv_a]["admissible"][:, j] = False
  cost_manager.neuron_capacity[conv_a][j] = 0.0

  state.curriculum.append(
      InstructionPacket(
          action="MERGE",
          indices=(i, j, conv_a),
          quadruplet=quadruplet,
          fan_in=cached["fan_in"],
          producer_bn_before=producer_bn_before,
          consumer_mean_before=consumer_mean_before,
          joint_rows_before=(row_i_before, row_j_before),
          norm=combined_norm,
          e_b=remaining_eb,
          n_active=n_active_before,
      )
  )

  cap_p = _neuron_capacity_scalar(abs(parent.gamma), parent.beta, parent.w_out)
  flux_removed = geometry["capacity_i"] + geometry["capacity_j"] - cap_p
  cost_manager.live_capacity[conv_a] = max(
      cost_manager.live_capacity[conv_a] - flux_removed, CAPACITY_FLOOR
  )

  fan_in = cached["fan_in"]
  cached["joint_weights"][i][:fan_in] = parent.w_in_eff
  cached["joint_weights"][i][fan_in:] = parent.w_out
  cached["producer_bn"]["gamma"][i], cached["producer_bn"]["beta"][i] = (
      parent.gamma,
      parent.beta,
  )
  cached["producer_bn"]["mean"][i], cached["producer_bn"]["var"][i] = (
      parent.mean,
      parent.var,
  )

  cached["joint_weights"][j][:] = 0.0
  cached["producer_bn"]["gamma"][j], cached["producer_bn"]["beta"][j] = 0.0, 0.0
  cached["producer_bn"]["mean"][j], cached["producer_bn"]["var"][j] = 0.0, 1.0

  cost_manager.neuron_capacity[conv_a][i] = _neuron_capacity_scalar(
      abs(float(cached["producer_bn"]["gamma"][i])),
      float(cached["producer_bn"]["beta"][i]),
      cached["joint_weights"][i, fan_in:],
  )
  update_merge_cache_row(
      cost_manager.merge_cache[conv_a],
      i,
      cost_manager.active_neighbors(conv_a, i),
      cached["joint_weights"],
      fan_in,
      cached["producer_bn"],
  )


def _execute_prune(state: _EncoderState, conv_a: str, i: int) -> None:
  """Executes PRUNE of neuron i and updates local state in O(1).

  Zeroes neuron i's producer filter, BN entries and consumer slice, compensates
  the consumer BN mean, and records an `InstructionPacket`.

  Args:
    state: Current mutable encoder state.
    conv_a: Producer convolution layer name.
    i: Neuron index to prune.
  """
  quadruplet = state.quadruplet_of[conv_a]
  _, bn_a, conv_b, bn_b = quadruplet
  cached = state.layer_cache[conv_a]
  weights = state.weights
  cost_manager = state.cost_manager
  flux_removed = float(cost_manager.neuron_capacity[conv_a][i])

  has_consumer_bn = bn_b is not None and f"{bn_b}/moving_mean" in weights
  consumer_mean_before = (
      weights[f"{bn_b}/moving_mean"].copy() if has_consumer_bn else None
  )

  live_joint, _, _ = read_quadruplet(weights, quadruplet)
  victim_row_before = live_joint[i].copy()
  producer_bn = cached["producer_bn"]
  gamma_i, beta_i = float(producer_bn["gamma"][i]), float(
      producer_bn["beta"][i]
  )
  producer_bn_before = (
      weights[f"{bn_a}/gamma"][i],
      weights[f"{bn_a}/beta"][i],
      weights[f"{bn_a}/moving_mean"][i],
      weights[f"{bn_a}/moving_variance"][i],
  )

  kernel_a, kernel_b = weights[f"{conv_a}/kernel"], weights[f"{conv_b}/kernel"]
  kernel_b_4d = (
      kernel_b
      if kernel_b.ndim == 4
      else kernel_b.reshape(1, 1, kernel_b.shape[0], kernel_b.shape[1])
  )

  # P1a: mu^b -= m_i sum_t chi_t w_{t,i}.
  if has_consumer_bn:
    m_i = relu_mean_scalar(abs(gamma_i), beta_i)
    chi = consumer_tap_fractions(kernel_b_4d, conv_b, state.feature_map_sizes)
    w_chi_i = np.sum(chi * kernel_b_4d[:, :, i, :], axis=(0, 1))
    _compensate_consumer_mean(state, quadruplet, -(m_i * w_chi_i))

  kernel_a[..., i] = 0.0
  if kernel_b.ndim == 4:
    kernel_b[:, :, i, :] = 0.0
  else:
    kernel_b[i, :] = 0.0
  for param in ["gamma", "beta", "moving_mean", "moving_variance"]:
    weights[f"{bn_a}/{param}"][i] = 0.0

  n_active_before = len(cost_manager.active_neurons[conv_a])
  remaining_eb = float(
      max(cost_manager.live_capacity[conv_a] - flux_removed, CAPACITY_FLOOR)
  )
  cost_manager.remove_neuron(conv_a, i)
  cost_manager.merge_cache[conv_a]["admissible"][i, :] = False
  cost_manager.merge_cache[conv_a]["admissible"][:, i] = False
  cost_manager.neuron_capacity[conv_a][i] = 0.0

  state.curriculum.append(
      InstructionPacket(
          action="PRUNE",
          indices=(-1, i, conv_a),
          quadruplet=quadruplet,
          fan_in=cached["fan_in"],
          producer_bn_before=producer_bn_before,
          consumer_mean_before=consumer_mean_before,
          victim_row_before=victim_row_before,
          norm=flux_removed,
          e_b=remaining_eb,
          n_active=n_active_before,
      )
  )

  cost_manager.live_capacity[conv_a] = max(
      cost_manager.live_capacity[conv_a] - flux_removed, CAPACITY_FLOOR
  )
  cached["joint_weights"][i][:] = 0.0
  cached["producer_bn"]["gamma"][i], cached["producer_bn"]["beta"][i] = 0.0, 0.0
  cached["producer_bn"]["mean"][i], cached["producer_bn"]["var"][i] = 0.0, 1.0


def _execute_evict(
    state: _EncoderState, conv_1: str, conv_2: str, conv_3: str
) -> None:
  """EVICT the residual branch of a bottleneck block.

  All kernels and BN tensors of the block are zeroed, except that BN_3's beta
  keeps the coherence-weighted Wiener shift delta_3* = alpha_3 beta_3 with
  alpha_3 = beta_3^2 / (gamma_3^2 + beta_3^2) (P1b-Wiener), evaluated in float64
  with +1e-12 in the denominator. The block's two internal layers are then
  collapsed in the ledger.

  Args:
    state: Current mutable encoder state.
    conv_1: First convolution layer name (`*_1_conv`) of the block.
    conv_2: Second convolution layer name (`*_2_conv`) of the block.
    conv_3: Third convolution layer name (`*_3_conv`) of the block.
  """
  weights = state.weights
  block_prefix = conv_1.rsplit("_", 2)[0]
  bn_3 = f"{block_prefix}_3_bn"

  block_layers = [
      (conv_1, f"{block_prefix}_1_bn"),
      (conv_2, f"{block_prefix}_2_bn"),
      (conv_3, bn_3),
  ]

  saved_block_tensors = {}
  for conv, bn in block_layers:
    if f"{conv}/kernel" in weights:
      saved_block_tensors[f"{conv}/kernel"] = weights[f"{conv}/kernel"].copy()
      weights[f"{conv}/kernel"][:] = 0.0
    for param in ["gamma", "beta", "moving_mean", "moving_variance"]:
      key = f"{bn}/{param}"
      if key in weights:
        saved_block_tensors[key] = weights[key].copy()
        if bn == bn_3 and param == "beta":
          # P1b-Wiener: beta_3 <- delta_3* = alpha_3 beta_3, with
          # alpha_3 = beta_3^2 / (gamma_3^2 + beta_3^2).
          gamma_3 = saved_block_tensors[f"{bn_3}/gamma"].astype(np.float64)
          beta_3 = saved_block_tensors[f"{bn_3}/beta"].astype(np.float64)
          alpha_3 = (beta_3 * beta_3) / (
              gamma_3 * gamma_3 + beta_3 * beta_3 + 1e-12
          )
          weights[key][:] = (alpha_3 * beta_3).astype(weights[key].dtype)
          continue
        weights[key][:] = 0.0

  state.curriculum.append(
      InstructionPacket(
          action="EVICT",
          indices=(conv_1, conv_2, conv_3),
          quadruplet=state.quadruplet_of[conv_1],
          fan_in=state.layer_cache[conv_1]["fan_in"],
          saved_block_tensors=saved_block_tensors,
      )
  )

  _collapse_evicted_layers(state, [conv_1, conv_2])


def _compensate_consumer_mean(
    state: _EncoderState, quadruplet: Quadruplet, delta_mu_b: np.ndarray
) -> None:
  """Adds delta_mu_b to the running mean of the consumer BN (P1a).

  Channels of the consumer that are already eliminated in its own quadruplet
  (when the consumer is itself a registered producer, i.e. Q1's consumer W_2)
  are left unchanged, so no eliminated channel is revived. The moving-mean array
  is replaced by a new array, and the cached producer-BN mean of the consumer
  layer is updated in place.

  Args:
    state: Current mutable encoder state.
    quadruplet: Quadruplet `(conv_a, bn_a, conv_b, bn_b)` whose consumer BN is
      updated.
    delta_mu_b: Shift to add to the consumer BN moving mean.
  """
  _, _, conv_b, bn_b = quadruplet
  cost_manager = state.cost_manager
  if conv_b in cost_manager.active_neurons:
    active_b = cost_manager.active_neurons[conv_b]
    dead_b = [k for k in range(len(delta_mu_b)) if k not in active_b]
    if dead_b:
      delta_mu_b[dead_b] = 0.0
  key = f"{bn_b}/moving_mean"
  state.weights[key] = state.weights[key] + delta_mu_b.astype(
      state.weights[key].dtype
  )
  if conv_b in state.layer_cache:
    state.layer_cache[conv_b]["producer_bn"]["mean"][:] = state.weights[key]


def _collapse_evicted_layers(state: _EncoderState, layers: List[str]) -> None:
  """Collapses each evicted internal layer in the ledger and caches.

  Sets E_a <- eps and clears `active_neurons` for each layer in `layers`.

  Args:
    state: Current mutable encoder state.
    layers: Producer convolution layer names of the evicted block.
  """
  cost_manager = state.cost_manager
  for conv in layers:
    if conv in cost_manager.active_neurons:
      cost_manager.live_capacity[conv] = CAPACITY_FLOOR
      cost_manager.active_neurons[conv].clear()
      cost_manager.neuron_capacity[conv][:] = 0.0
      cost_manager.merge_cache[conv]["admissible"][:, :] = False
      if conv in state.layer_cache:
        cached = state.layer_cache[conv]
        cached["joint_weights"][:] = 0.0
        for param in ["gamma", "beta"]:
          cached["producer_bn"][param][:] = 0.0
        cached["producer_bn"]["mean"][:] = 0.0
        cached["producer_bn"]["var"][:] = 1.0


# ==============================================================================
# 6. Decoder
# ==============================================================================


def hope_decode_step(
    weights: Dict[str, np.ndarray], packet: InstructionPacket
) -> Dict[str, np.ndarray]:
  """Undoes one encoder action, in place; returns `weights`.

  EVICT restores the saved block tensors. PRUNE and MERGE restore the consumer
  BN moving mean, the producer BN entries and the joint rows of the affected
  neuron(s). Outgoing weights are restored exactly; the raw producer filter is
  recovered by inverting the BN folding
  w_in^eff = (gamma / sqrt(var + eps)) * kernel:
  kernel = w_in^eff / (gamma / sqrt(var + eps) + RAW_WEIGHT_DIV_GUARD), which is
  exact up to rounding for ordinary neurons (5.96e-8 maximum error on the
  pretrained ResNet-50). It cannot recover the filter of a neuron whose BN scale
  gamma / sqrt(var + eps) is zero or not large compared with the guard, and it
  is inexact (NaN below -eps) for a negative stored moving variance, which the
  encoder's BN folding clamps at 0.

  Args:
    weights: Mapping from Keras tensor name to array, modified in place.
    packet: Decoding instruction for the encoder action to undo.

  Returns:
    The modified `weights` dictionary.
  """
  action = packet.action
  if action == "EVICT":
    for path, saved in packet.saved_block_tensors.items():
      weights[path][:] = saved
    return weights

  conv_a, bn_a, conv_b, bn_b = packet.quadruplet
  fan_in = packet.fan_in

  if (
      bn_b is not None
      and packet.consumer_mean_before is not None
      and f"{bn_b}/moving_mean" in weights
  ):
    weights[f"{bn_b}/moving_mean"][:] = packet.consumer_mean_before

  kernel_a = weights[f"{conv_a}/kernel"]
  kernel_b = weights[f"{conv_b}/kernel"]
  gamma_a = weights[f"{bn_a}/gamma"]
  beta_a = weights[f"{bn_a}/beta"]
  mean_a = weights[f"{bn_a}/moving_mean"]
  var_a = weights[f"{bn_a}/moving_variance"]

  if action == "MERGE":
    i, j, _ = packet.indices
    gamma_i, beta_i, mean_i, var_i, gamma_j, beta_j, mean_j, var_j = (
        packet.producer_bn_before
    )
    gamma_a[i], beta_a[i], mean_a[i], var_a[i] = gamma_i, beta_i, mean_i, var_i
    gamma_a[j], beta_a[j], mean_a[j], var_a[j] = gamma_j, beta_j, mean_j, var_j

    row_i, row_j = packet.joint_rows_before
    w_in_i, w_out_i = row_i[:fan_in], row_i[fan_in:]
    w_in_j, w_out_j = row_j[:fan_in], row_j[fan_in:]

    bn_scale_i = gamma_i / np.sqrt(var_i + BN_EPSILON)
    bn_scale_j = gamma_j / np.sqrt(var_j + BN_EPSILON)

    kernel_a[..., i] = (w_in_i / (bn_scale_i + RAW_WEIGHT_DIV_GUARD)).reshape(
        kernel_a.shape[:-1]
    )
    kernel_a[..., j] = (w_in_j / (bn_scale_j + RAW_WEIGHT_DIV_GUARD)).reshape(
        kernel_a.shape[:-1]
    )
    if kernel_b.ndim == 4:
      kernel_b[:, :, i, :] = w_out_i.reshape(kernel_b.shape[0:2] + (-1,))
      kernel_b[:, :, j, :] = w_out_j.reshape(kernel_b.shape[0:2] + (-1,))
    else:
      kernel_b[i, :] = w_out_i
      kernel_b[j, :] = w_out_j

  elif action == "PRUNE":
    _, i, _ = packet.indices
    gamma_i, beta_i, mean_i, var_i = packet.producer_bn_before
    gamma_a[i], beta_a[i], mean_a[i], var_a[i] = gamma_i, beta_i, mean_i, var_i

    row = packet.victim_row_before
    w_in, w_out = row[:fan_in], row[fan_in:]
    bn_scale = gamma_i / np.sqrt(var_i + BN_EPSILON)
    kernel_a[..., i] = (w_in / (bn_scale + RAW_WEIGHT_DIV_GUARD)).reshape(
        kernel_a.shape[:-1]
    )
    if kernel_b.ndim == 4:
      kernel_b[:, :, i, :] = w_out.reshape(kernel_b.shape[0:2] + (-1,))
    else:
      kernel_b[i, :] = w_out

  return weights


def hope_decode(
    weights: Dict[str, np.ndarray],
    curriculum: Sequence[InstructionPacket],
    num_steps: Optional[int] = None,
) -> Dict[str, np.ndarray]:
  """Applies the first `num_steps` packets of `curriculum` to a `weights` copy.

  With the curriculum returned by `hope_encode` (decoding order), decoding k
  steps restores the model as it was before the last k encoder actions; decoding
  all steps (`num_steps=None`) restores the input model.

  Args:
    weights: Mapping from Keras tensor name to compressed parameter array (not
      modified).
    curriculum: Sequence of `InstructionPacket`s in decoding order.
    num_steps: Number of packets from the start of `curriculum` to apply, or
      None to apply all packets.

  Returns:
    New dictionary of decoded weight arrays.
  """
  decoded = {k: v.copy() for k, v in weights.items()}
  for packet in itertools.islice(curriculum, num_steps):
    hope_decode_step(decoded, packet)
  return decoded
