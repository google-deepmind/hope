# Copyright 2026 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Licensed under the Apache License, Version 2.0 (the "License");
# http://www.apache.org/licenses/LICENSE-2.0
"""HOPE (Hilbert Operator for Progressive Encoding): Data-Free Structured Compression Baselines.

Implements the seven strictly data-free structured neuron compression baselines
evaluated against
HOPE on ResNet-50 (Section 10.1.3 of https://arxiv.org/abs/2607.21366):

  1. L1_IN             L1-Norm Input Filter Pruning (Li et al., ICLR 2017)
                       s_i = ||w_in,i^eff||_1
  2. L1_JOINT          L1-Norm Joint Coupled Pruning
                       s_i = ||w_in,i^eff||_1 + ||w_out,i||_1
  3. BN_SCALE          BatchNorm Scaling Factor Pruning / Network Slimming (Liu
  et al., ICCV 2017)
                       s_i = |gamma_i|
  4. FPGM              Filter Pruning via Geometric Median (He et al., CVPR
  2019)
                       s_i = sum_{j=1}^{N_l} ||w_in,i^eff - w_in,j^eff||_2
  5. L1_JOINT_BIASCORR Bias-Corrected / BN-Mean-Compensated Joint L1 Pruning
  (Nagel et al., ICCV 2019)
                       Same importance score as L1_JOINT, paired with
                       closed-form Gaussian ReLU
                       consumer BatchNorm running-mean compensation upon
                       pruning:
                         mu^b <- mu^b - m_i sum_t chi_t w_{t,i}
  6. WS_COSINE_MERGE   Data-Free Weight-Space Cosine Neuron Merging (Srinivas &
  Babu, BMVC 2015;
                       Kim et al., 2020)
                       Replaces positively aligned pairs (rho_ij^ws > 0) of
                       augmented input vectors
                       w~_in,i = [w_in,i^eff; b_i] by a single merged parent
                       neuron in Euclidean
                       weight space, competing in a unified priority queue
                       against single-neuron
                       pruning with Frobenius norm cost ||w_out (w~_in)^T||_F =
                       a_i b_i.
  7. SYNFLOW           Data-Free Synaptic Flow Pruning (Tanaka et al., NeurIPS
  2020)
                       Multiplicative L1 path-norm saliency through the
                       bottleneck neuron:
                         s_i = ||w_in,i^eff||_1 * ||w_out,i||_1

To enable cross-layer tournament selection without calibration data, each neuron
or pair score in
layer l is normalized by the layer's initial total score sum_{k=1}^{N_l} s_k^(l)
(controlled by
`use_normalized_baselines=True`). Every executed action emits an
`InstructionPacket` compatible with
`hope.hope_decode_step` and `hope.hope_decode`, so baseline curricula support
the same
lossless progressive decoding as HOPE.
"""

import dataclasses
import gc
import heapq
import logging
import math
import os
import time
from typing import Callable, Dict, List, Optional, Sequence, Set, Tuple

import hope
import numpy as np
from scipy.spatial import distance

__all__ = [
    "BASELINE_METHODS",
    "DEFAULT_BASELINE_FILES",
    "BaselineActionEntry",
    "BaselineCostManager",
    "compute_baseline_prune_cost",
    "baseline_encode",
    "HOPE_Baseline_Encoder",
    "run_all_baselines",
]

logger = logging.getLogger(__name__)

# Canonical baseline identifiers in paper order (Section 10.1.3).
BASELINE_METHODS: Tuple[str, ...] = (
    "L1_IN",
    "L1_JOINT",
    "BN_SCALE",
    "FPGM",
    "L1_JOINT_BIASCORR",
    "WS_COSINE_MERGE",
    "SYNFLOW",
)

# Default two-column CSV filenames (`density,accuracy`) written by each
# baseline.
DEFAULT_BASELINE_FILES: Dict[str, str] = {
    "L1_IN": "results_L1_IN.csv",
    "L1_JOINT": "results_L1_JOINT.csv",
    "BN_SCALE": "results_BN_SCALE.csv",
    "FPGM": "results_FPGM.csv",
    "L1_JOINT_BIASCORR": "results_L1_JOINT_BIASCORR.csv",
    "L1_JOINT_P1A": "results_L1_JOINT_BIASCORR.csv",
    "WS_COSINE_MERGE": "results_WS_COSINE_MERGE.csv",
    "SYNFLOW": "results_SYNFLOW.csv",
}

# Small additive constant preventing division by zero when normalizing layer
# scores.
_NORM_GUARD = 1e-9
_UNIT_VEC_GUARD = 1e-12


# =====================================================================================================================
# 1. Priority-queue cost manager with lazy invalidation
# =====================================================================================================================


@dataclasses.dataclass(order=True)
class BaselineActionEntry:
  """One candidate compression action stored in the global min-heap.

  Ordered by `cost` (with ties broken by insertion order via Python's tuple
  comparison only if `cost`
  is equal; non-comparison fields are excluded). Version tags on the
  participating neurons allow
  lazy deletion when a neuron is pruned or merged.
  """

  cost: float
  action_type: str = dataclasses.field(compare=False)
  layer_name: str = dataclasses.field(compare=False)
  location: Tuple[int, ...] = dataclasses.field(compare=False)
  versions: Tuple[int, ...] = dataclasses.field(compare=False)
  fan_in: int = dataclasses.field(compare=False)


class BaselineCostManager:
  """Min-heap cost manager with per-neuron version tracking and layer-normalization statistics."""

  def __init__(self) -> None:
    self.heap: List[BaselineActionEntry] = []
    self.neuron_versions: Dict[str, Dict[int, int]] = {}
    self.active_neurons: Dict[str, Set[int]] = {}
    self.layer_energies: Dict[str, float] = {}
    self.baseline_norms: Dict[str, Dict[str, float]] = {}

  def register_layer(
      self,
      layer_name: str,
      num_neurons: int,
      initial_energy: float,
      baseline_stats: Optional[Dict[str, float]] = None,
  ) -> None:
    """Registers a producer layer with `num_neurons` initially active channels."""
    self.neuron_versions[layer_name] = {i: 0 for i in range(num_neurons)}
    self.active_neurons[layer_name] = set(range(num_neurons))
    self.layer_energies[layer_name] = initial_energy
    if baseline_stats:
      self.baseline_norms[layer_name] = baseline_stats

  def push_action(
      self,
      cost: float,
      action_type: str,
      layer_name: str,
      location: Tuple[int, ...],
      fan_in: int,
  ) -> None:
    """Pushes a candidate `PRUNE` or `MERGE` action onto the min-heap with current neuron versions."""
    if cost == np.inf or math.isnan(cost):
      return
    current_vs = tuple(
        self.neuron_versions[layer_name][idx] for idx in location
    )
    entry = BaselineActionEntry(
        cost=float(cost),
        action_type=action_type,
        layer_name=layer_name,
        location=location,
        versions=current_vs,
        fan_in=fan_in,
    )
    heapq.heappush(self.heap, entry)

  def _is_entry_valid(self, entry: BaselineActionEntry) -> bool:
    """Returns True iff all neurons referenced by `entry` are still active at their recorded versions."""
    if entry.layer_name not in self.neuron_versions:
      return False
    active_set = self.active_neurons[entry.layer_name]
    versions_map = self.neuron_versions[entry.layer_name]
    for idx, stored_v in zip(entry.location, entry.versions):
      if idx not in active_set or versions_map[idx] != stored_v:
        return False
    return True

  def pop_best_valid_action(self) -> Optional[BaselineActionEntry]:
    """Pops and returns the lowest-cost valid candidate action, discarding stale heap entries."""
    while self.heap:
      entry = heapq.heappop(self.heap)
      if self._is_entry_valid(entry):
        return entry
    return None

  def clean_heap(self) -> None:
    """Compacts the min-heap in place by removing stale entries."""
    self.heap = [entry for entry in self.heap if self._is_entry_valid(entry)]
    heapq.heapify(self.heap)
    gc.collect()

  def invalidate_neuron(self, layer_name: str, idx: int) -> None:
    """Increments the version counter of neuron `idx`, invalidating queued actions that reference it."""
    if idx in self.neuron_versions[layer_name]:
      self.neuron_versions[layer_name][idx] += 1

  def remove_neuron_idx(self, layer_name: str, idx: int) -> None:
    """Marks neuron `idx` as inactive and invalidates queued actions referencing it."""
    self.invalidate_neuron(layer_name, idx)
    self.active_neurons[layer_name].discard(idx)


# =====================================================================================================================
# 2. Baseline pruning scores and weight-space cosine merging geometry
# =====================================================================================================================


def compute_baseline_prune_cost(
    benchmark_method: str,
    i: int,
    joint_weights: np.ndarray,
    fan_in: int,
    producer_bn: Dict[str, np.ndarray],
    baseline_norms: Optional[Dict[str, float]] = None,
    fpgm_scores: Optional[np.ndarray] = None,
) -> float:
  """Computes the layer-normalized single-neuron pruning cost for neuron `i`.

  Args:
    benchmark_method: One of `BASELINE_METHODS` (or alias `"L1_JOINT_P1A"`).
    i: Neuron index in `0 .. N_l - 1`.
    joint_weights: `(N_l, fan_in + c_out)` array of `[w_in,i^eff | w_out,i]`
      rows from `read_quadruplet`.
    fan_in: Length `n` of the BN-folded incoming filter slice `w_in,i^eff`.
    producer_bn: Producer BatchNorm dictionary (`"gamma"`, `"beta"`, `"mean"`,
      `"var"`).
    baseline_norms: Optional dictionary mapping method name to the layer's
      initial score sum.
    fpgm_scores: Optional precomputed vector of pairwise Euclidean distance sums
      for `"FPGM"`.

  Returns:
    The normalized pruning cost `s_i / (norm + 1e-9)` as a Python float.
  """
  baseline_norms = baseline_norms or {}
  if benchmark_method == "L1_IN":
    w_in = joint_weights[i, :fan_in]
    raw_score = np.sum(np.abs(w_in))
    norm = baseline_norms.get("L1_IN", 1.0)
    return float(raw_score / (norm + _NORM_GUARD))

  if benchmark_method == "L1_JOINT":
    w_joint = joint_weights[i]
    raw_score = np.sum(np.abs(w_joint))
    norm = baseline_norms.get("L1_JOINT", 1.0)
    return float(raw_score / (norm + _NORM_GUARD))

  if benchmark_method == "BN_SCALE":
    gamma = producer_bn["gamma"][i]
    raw_score = np.abs(gamma)
    norm = baseline_norms.get("BN_SCALE", 1.0)
    return float(raw_score / (norm + _NORM_GUARD))

  if benchmark_method == "FPGM":
    if fpgm_scores is not None:
      raw_score = float(fpgm_scores[i])
    else:
      w_in_all = joint_weights[:, :fan_in].astype(np.float64)
      raw_score = float(np.sum(np.linalg.norm(w_in_all - w_in_all[i], axis=1)))
    norm = baseline_norms.get("FPGM", 1.0)
    return float(raw_score / (norm + _NORM_GUARD))

  if benchmark_method in ("L1_JOINT_BIASCORR", "L1_JOINT_P1A"):
    w_joint = joint_weights[i]
    raw_score = np.sum(np.abs(w_joint))
    norm = baseline_norms.get(
        "L1_JOINT_BIASCORR", baseline_norms.get("L1_JOINT_P1A", 1.0)
    )
    return float(raw_score / (norm + _NORM_GUARD))

  if benchmark_method == "WS_COSINE_MERGE":
    w_in_eff = joint_weights[i, :fan_in].astype(np.float64)
    w_out = joint_weights[i, fan_in:].astype(np.float64)
    scale_i = float(producer_bn["gamma"][i]) / math.sqrt(
        max(float(producer_bn["var"][i]), 0.0) + hope.BN_EPSILON
    )
    b_eff_i = float(producer_bn["beta"][i]) - scale_i * float(
        producer_bn["mean"][i]
    )
    a_i = math.sqrt(float(np.dot(w_in_eff, w_in_eff)) + b_eff_i * b_eff_i)
    b_i = float(np.linalg.norm(w_out))
    raw_score = a_i * b_i
    norm = baseline_norms.get("WS_COSINE_MERGE", 1.0)
    return float(raw_score / (norm + _NORM_GUARD))

  if benchmark_method == "SYNFLOW":
    w_in = joint_weights[i, :fan_in]
    w_out = joint_weights[i, fan_in:]
    raw_score = float(
        np.sum(np.abs(w_in), dtype=np.float64)
        * np.sum(np.abs(w_out), dtype=np.float64)
    )
    norm = baseline_norms.get("SYNFLOW", 1.0)
    return float(raw_score / (norm + _NORM_GUARD))

  raise ValueError(
      f"Unknown benchmark_method: {benchmark_method!r}. Expected one of"
      f" {BASELINE_METHODS}."
  )


def _compute_ws_cosine_layer_features(
    joint_weights: np.ndarray, fan_in: int, producer_bn: Dict[str, np.ndarray]
) -> Dict[str, np.ndarray]:
  """Computes weight-space cosine merging features for a layer.

  For each neuron `i`:
    - `w~_in,i = [w_in,i^eff, b_i]` where `b_i = beta_i - scale_i * mu_i` is the
      BN-folded bias,
    - `a_i = ||w~_in,i||_2`, `b_i = ||w_out,i||_2`,
      `e_i = a_i * b_i = ||w_out,i (w~_in,i)^T||_F`,
    - `u_i = w~_in,i / (a_i + 1e-12)`.

  Args:
    joint_weights: `(N_l, fan_in + c_out)` array of `[w_in,i^eff | w_out,i]`
      rows.
    fan_in: Length of the BN-folded incoming filter slice `w_in,i^eff`.
    producer_bn: Producer BatchNorm dictionary (`"gamma"`, `"beta"`, `"mean"`,
      `"var"`).

  Returns:
    Dictionary containing augmented incoming weights, outgoing weights, norms,
    energies, and unit incoming directions.
  """
  w_in = joint_weights[:, :fan_in].astype(np.float64)
  w_out = joint_weights[:, fan_in:].astype(np.float64)
  gamma = producer_bn["gamma"].astype(np.float64)
  beta = producer_bn["beta"].astype(np.float64)
  mean = producer_bn["mean"].astype(np.float64)
  var = producer_bn["var"].astype(np.float64)

  scale = gamma / np.sqrt(np.maximum(var, 0.0) + hope.BN_EPSILON)
  b_eff = beta - scale * mean
  w_in_aug = np.column_stack([w_in, b_eff])

  a_vec = np.linalg.norm(w_in_aug, axis=1)
  b_vec = np.linalg.norm(w_out, axis=1)
  e_vec = a_vec * b_vec
  u_in = w_in_aug / (a_vec[:, None] + _UNIT_VEC_GUARD)
  return {
      "W_in_aug": w_in_aug,
      "W_out": w_out,
      "a_vec": a_vec,
      "b_vec": b_vec,
      "e_vec": e_vec,
      "U_in": u_in,
  }


def _push_ws_cosine_all_pairs(
    conv_a_name: str,
    fan_in: int,
    feats: Dict[str, np.ndarray],
    norm: float,
    cost_manager: BaselineCostManager,
) -> None:
  """Pushes all positively aligned weight-space merge candidates in a layer.

  For every pair `(i, j)` with `i < j` and positive input cosine similarity
  `rho_ij^ws = u_i^T u_j > 0`, synthesizes the energy-weighted parent input
  vector `w~_in,p = omega_i w~_in,i + omega_j w~_in,j`
  (`omega_k = e_k / (e_i + e_j)`), projects the children onto `u_p`
  (`r_k = <w~_in,k, u_p> / a_p`), sets `w_out,p = r_i w_out,i + r_j w_out,j`,
  and evaluates the exact Frobenius residual norm:
    `||w_out,i (w~_in,i - r_i w~_in,p)^T + w_out,j (w~_in,j - r_j
    w~_in,p)^T||_F`.

  Args:
    conv_a_name: Producer convolution layer name.
    fan_in: Length of the BN-folded incoming filter slice.
    feats: Cached weight-space layer features from
      `_compute_ws_cosine_layer_features`.
    norm: Layer normalization denominator for `WS_COSINE_MERGE`.
    cost_manager: Global min-heap cost manager receiving candidate actions.
  """
  a_vec = feats["a_vec"]
  b_vec = feats["b_vec"]
  e_vec = feats["e_vec"]
  u_in = feats["U_in"]
  w_out = feats["W_out"]

  num_neurons = a_vec.shape[0]
  rho_mat = np.clip(u_in @ u_in.T, -1.0, 1.0)
  g_out = w_out @ w_out.T

  idx_i, idx_j = np.triu_indices(num_neurons, k=1)
  rho_ij = rho_mat[idx_i, idx_j]
  e_i = e_vec[idx_i]
  e_j = e_vec[idx_j]

  valid = (rho_ij > 0.0) & (e_i > _UNIT_VEC_GUARD) & (e_j > _UNIT_VEC_GUARD)
  if not np.any(valid):
    return

  vi = idx_i[valid]
  vj = idx_j[valid]
  rho = rho_ij[valid]
  ai = a_vec[vi]
  aj = a_vec[vj]
  bi_sq = b_vec[vi] ** 2
  bj_sq = b_vec[vj] ** 2
  ei = e_vec[vi]
  ej = e_vec[vj]
  g_out_ij = g_out[vi, vj]

  denom_e = ei + ej + _UNIT_VEC_GUARD
  omega_i = ei / denom_e
  omega_j = ej / denom_e

  alpha_i = omega_i * ai
  alpha_j = omega_j * aj
  ap_sq = np.maximum(
      0.0, alpha_i**2 + alpha_j**2 + 2.0 * alpha_i * alpha_j * rho
  )
  ap = np.sqrt(ap_sq)
  ap_denom_sq = (ap + _UNIT_VEC_GUARD) ** 2

  dot_ip = ai * (alpha_i + alpha_j * rho)
  dot_jp = aj * (alpha_i * rho + alpha_j)
  ri = dot_ip / ap_denom_sq
  rj = dot_jp / ap_denom_sq

  delta_i_sq = np.maximum(0.0, ai**2 - 2.0 * ri * dot_ip + (ri**2) * ap_sq)
  delta_j_sq = np.maximum(0.0, aj**2 - 2.0 * rj * dot_jp + (rj**2) * ap_sq)
  delta_ij = ai * aj * rho - ri * dot_jp - rj * dot_ip + ri * rj * ap_sq

  err_sq = np.maximum(
      0.0, bi_sq * delta_i_sq + bj_sq * delta_j_sq + 2.0 * g_out_ij * delta_ij
  )
  raw_costs = np.sqrt(err_sq)
  norm_costs = raw_costs / (norm + _NORM_GUARD)

  for k in range(len(vi)):
    cost_manager.push_action(
        float(norm_costs[k]),
        "MERGE",
        conv_a_name,
        (int(vi[k]), int(vj[k])),
        fan_in,
    )


def _push_ws_cosine_single_neuron_updates(
    conv_a_name: str,
    survivor_idx: int,
    fan_in: int,
    feats: Dict[str, np.ndarray],
    norm: float,
    cost_manager: BaselineCostManager,
) -> None:
  """Pushes updated `PRUNE` and pairwise `MERGE` candidates for `survivor_idx` after a merge."""
  a_vec = feats["a_vec"]
  b_vec = feats["b_vec"]
  e_vec = feats["e_vec"]
  u_in = feats["U_in"]
  w_out = feats["W_out"]

  i = survivor_idx
  prune_cost = float(e_vec[i] / (norm + _NORM_GUARD))
  cost_manager.push_action(prune_cost, "PRUNE", conv_a_name, (i,), fan_in)

  if e_vec[i] <= _UNIT_VEC_GUARD:
    return

  other_active = sorted(
      k for k in cost_manager.active_neurons[conv_a_name] if k != i
  )
  if not other_active:
    return

  k_arr = np.asarray(other_active, dtype=np.int64)
  rho_ik = np.clip(u_in[k_arr] @ u_in[i], -1.0, 1.0)
  ek = e_vec[k_arr]
  valid = (rho_ik > 0.0) & (ek > _UNIT_VEC_GUARD)
  if not np.any(valid):
    return

  vk = k_arr[valid]
  rho = rho_ik[valid]
  ai = a_vec[i]
  ak = a_vec[vk]
  bi_sq = b_vec[i] ** 2
  bk_sq = b_vec[vk] ** 2
  ei = e_vec[i]
  ek_v = e_vec[vk]
  g_out_ik = w_out[vk] @ w_out[i]

  denom_e = ei + ek_v + _UNIT_VEC_GUARD
  omega_i = ei / denom_e
  omega_k = ek_v / denom_e

  alpha_i = omega_i * ai
  alpha_k = omega_k * ak
  ap_sq = np.maximum(
      0.0, alpha_i**2 + alpha_k**2 + 2.0 * alpha_i * alpha_k * rho
  )
  ap = np.sqrt(ap_sq)
  ap_denom_sq = (ap + _UNIT_VEC_GUARD) ** 2

  dot_ip = ai * (alpha_i + alpha_k * rho)
  dot_kp = ak * (alpha_i * rho + alpha_k)
  ri = dot_ip / ap_denom_sq
  rk = dot_kp / ap_denom_sq

  delta_i_sq = np.maximum(0.0, ai**2 - 2.0 * ri * dot_ip + (ri**2) * ap_sq)
  delta_k_sq = np.maximum(0.0, ak**2 - 2.0 * rk * dot_kp + (rk**2) * ap_sq)
  delta_ik = ai * ak * rho - ri * dot_kp - rk * dot_ip + ri * rk * ap_sq

  err_sq = np.maximum(
      0.0, bi_sq * delta_i_sq + bk_sq * delta_k_sq + 2.0 * g_out_ik * delta_ik
  )
  norm_costs = np.sqrt(err_sq) / (norm + _NORM_GUARD)

  for idx_pos, k_val in enumerate(vk):
    kv = int(k_val)
    loc = (i, kv) if i < kv else (kv, i)
    cost_manager.push_action(
        float(norm_costs[idx_pos]), "MERGE", conv_a_name, loc, fan_in
    )


# =====================================================================================================================
# 3. Action execution helpers (PRUNE and weight-space MERGE)
# =====================================================================================================================


def _write_density_accuracy_file(
    filepath: str, results: Sequence[Tuple[float, float]]
) -> None:
  """Writes `(density, accuracy)` pairs to `filepath` in two-column CSV format."""
  with open(filepath, "w") as f_out:
    f_out.write("density,accuracy\n")
    for d_val, a_val in results:
      f_out.write(f"{float(d_val):.6f},{float(a_val):.6f}\n")


def _execute_baseline_prune(
    current_weights: Dict[str, np.ndarray],
    quadruplet: hope.Quadruplet,
    i: int,
    fan_in: int,
    benchmark_method: str,
    cost_manager: BaselineCostManager,
    curriculum: List[hope.InstructionPacket],
) -> int:
  """Executes a single-neuron `PRUNE` action and records its packet.

  Args:
    current_weights: Mutable dictionary of live network weights.
    quadruplet: Bottleneck quadruplet `(conv_a, bn_a, conv_b, bn_b)`.
    i: Index of the pruned neuron in `conv_a`.
    fan_in: Length of the BN-folded incoming filter slice.
    benchmark_method: Active baseline method identifier.
    cost_manager: Global min-heap cost manager.
    curriculum: List to which the `PRUNE` `InstructionPacket` is appended.

  Returns:
    The net change in total nonzero parameters (`<= 0`).
  """
  conv_a, bn_a, conv_b, bn_b = quadruplet
  k_a = current_weights[f"{conv_a}/kernel"]
  k_b = current_weights[f"{conv_b}/kernel"]
  # The consumer kernel may be 2D (Dense consumer, as in hope.py's mixed
  # Conv/Dense quadruplets); index it through a 4D view that shares storage
  # with the stored tensor, matching hope.read_quadruplet's convention.
  k_b4 = (
      k_b
      if k_b.ndim == 4
      else k_b.reshape(1, 1, k_b.shape[0], k_b.shape[1])
  )

  gamma_i = current_weights[f"{bn_a}/gamma"][i]
  beta_i = current_weights[f"{bn_a}/beta"][i]
  mean_i = current_weights[f"{bn_a}/moving_mean"][i]
  var_i = current_weights[f"{bn_a}/moving_variance"][i]

  inv_std_i = np.float32(1.0) / np.sqrt(
      np.maximum(var_i, np.float32(0.0)) + np.float32(hope.BN_EPSILON)
  )
  scale_i = gamma_i * inv_std_i
  w_n_i = (k_a[..., i] * scale_i).reshape(-1)
  w_c_i = k_b4[:, :, i, :].reshape(-1)
  victim_row_before = np.concatenate([w_n_i, w_c_i])

  param_delta = 0
  consumer_mean_before: Optional[np.ndarray] = None
  # The consumer BN compensation only applies when the consumer actually
  # has a BatchNorm (quadruplets with `bn_b=None`, e.g. a Dense consumer,
  # have no running mean to shift) - the same `has_consumer_bn` guard that
  # hope.py's own prune uses.
  has_consumer_bn = (
      bn_b is not None and f"{bn_b}/moving_mean" in current_weights
  )
  if benchmark_method == "L1_JOINT_BIASCORR" and has_consumer_bn:
    consumer_mean_before = current_weights[f"{bn_b}/moving_mean"].copy()
    m_i = hope.relu_mean_scalar(abs(float(gamma_i)), float(beta_i))
    chi_2d = hope.consumer_tap_fractions(k_b, conv_b)
    w_chi_i = np.sum(chi_2d * k_b4[:, :, i, :], axis=(0, 1))
    delta_mu_b = -(m_i * w_chi_i)
    if conv_b in cost_manager.active_neurons:
      active_b = cost_manager.active_neurons[conv_b]
      dead_b = [k for k in range(len(delta_mu_b)) if k not in active_b]
      if dead_b:
        delta_mu_b[dead_b] = 0.0
    mu_b_before = current_weights[f"{bn_b}/moving_mean"]
    nz_mu_before = int(np.count_nonzero(mu_b_before))
    current_weights[f"{bn_b}/moving_mean"] = mu_b_before + delta_mu_b.astype(
        mu_b_before.dtype
    )
    nz_mu_after = int(np.count_nonzero(current_weights[f"{bn_b}/moving_mean"]))
    param_delta += nz_mu_after - nz_mu_before

  curriculum.append(
      hope.InstructionPacket(
          action="PRUNE",
          indices=(-1, i, conv_a),
          quadruplet=quadruplet,
          fan_in=fan_in,
          producer_bn_before=(gamma_i, beta_i, mean_i, var_i),
          consumer_mean_before=consumer_mean_before,
          victim_row_before=victim_row_before,
      )
  )

  removed_nz = int(np.count_nonzero(k_a[..., i])) + int(
      np.count_nonzero(k_b4[:, :, i, :])
  )
  k_a[..., i] = 0.0
  k_b4[:, :, i, :] = 0.0
  for param in ("gamma", "beta", "moving_mean", "moving_variance"):
    arr = current_weights[f"{bn_a}/{param}"]
    if arr[i] != 0.0:
      removed_nz += 1
    arr[i] = 0.0

  cost_manager.remove_neuron_idx(conv_a, i)
  return param_delta - removed_nz


def _execute_baseline_merge(
    current_weights: Dict[str, np.ndarray],
    quadruplet: hope.Quadruplet,
    i: int,
    j: int,
    fan_in: int,
    ws_layer_cache: Dict[str, Dict[str, np.ndarray]],
    cost_manager: BaselineCostManager,
    curriculum: List[hope.InstructionPacket],
) -> int:
  """Executes a weight-space cosine `MERGE` `(i, j) -> i` and updates caches.

  Args:
    current_weights: Mutable dictionary of live network weights.
    quadruplet: Bottleneck quadruplet `(conv_a, bn_a, conv_b, bn_b)`.
    i: Index of the surviving merged neuron in `conv_a`.
    j: Index of the absorbed neuron in `conv_a`.
    fan_in: Length of the BN-folded incoming filter slice.
    ws_layer_cache: Per-layer cached weight-space feature arrays.
    cost_manager: Global min-heap cost manager.
    curriculum: List to which the `MERGE` `InstructionPacket` is appended.

  Returns:
    The net change in total nonzero parameters (`<= 0`).
  """
  conv_a, bn_a, conv_b, _ = quadruplet
  k_a = current_weights[f"{conv_a}/kernel"]
  k_b = current_weights[f"{conv_b}/kernel"]
  # See _execute_baseline_prune: index a 2D (Dense) consumer kernel through
  # a storage-sharing 4D view.
  k_b4 = (
      k_b
      if k_b.ndim == 4
      else k_b.reshape(1, 1, k_b.shape[0], k_b.shape[1])
  )

  gamma_i = current_weights[f"{bn_a}/gamma"][i]
  beta_i = current_weights[f"{bn_a}/beta"][i]
  mean_i = current_weights[f"{bn_a}/moving_mean"][i]
  var_i = current_weights[f"{bn_a}/moving_variance"][i]

  gamma_j = current_weights[f"{bn_a}/gamma"][j]
  beta_j = current_weights[f"{bn_a}/beta"][j]
  mean_j = current_weights[f"{bn_a}/moving_mean"][j]
  var_j = current_weights[f"{bn_a}/moving_variance"][j]

  inv_std_i = np.float32(1.0) / np.sqrt(
      np.maximum(var_i, np.float32(0.0)) + np.float32(hope.BN_EPSILON)
  )
  scale_i = gamma_i * inv_std_i
  w_n_i = (k_a[..., i] * scale_i).reshape(-1)
  w_c_i = k_b4[:, :, i, :].reshape(-1)
  row_i_before = np.concatenate([w_n_i, w_c_i])

  inv_std_j = np.float32(1.0) / np.sqrt(
      np.maximum(var_j, np.float32(0.0)) + np.float32(hope.BN_EPSILON)
  )
  scale_j = gamma_j * inv_std_j
  w_n_j = (k_a[..., j] * scale_j).reshape(-1)
  w_c_j = k_b4[:, :, j, :].reshape(-1)
  row_j_before = np.concatenate([w_n_j, w_c_j])

  # Synthesize merged parent in Euclidean weight space from live weights of i
  # and j.
  w_in_i_f64 = w_n_i.astype(np.float64)
  w_in_j_f64 = w_n_j.astype(np.float64)
  w_out_i_f64 = w_c_i.astype(np.float64)
  w_out_j_f64 = w_c_j.astype(np.float64)

  b_eff_i = float(beta_i) - float(scale_i) * float(mean_i)
  b_eff_j = float(beta_j) - float(scale_j) * float(mean_j)
  w_in_aug_i = np.append(w_in_i_f64, b_eff_i)
  w_in_aug_j = np.append(w_in_j_f64, b_eff_j)

  a_i = float(np.linalg.norm(w_in_aug_i))
  a_j = float(np.linalg.norm(w_in_aug_j))
  b_i_norm = float(np.linalg.norm(w_out_i_f64))
  b_j_norm = float(np.linalg.norm(w_out_j_f64))
  e_i = a_i * b_i_norm
  e_j = a_j * b_j_norm

  u_i = w_in_aug_i / (a_i + _UNIT_VEC_GUARD)
  u_j = w_in_aug_j / (a_j + _UNIT_VEC_GUARD)
  rho_ws = float(np.clip(np.dot(u_i, u_j), -1.0, 1.0))

  denom_e = e_i + e_j + _UNIT_VEC_GUARD
  omega_i = e_i / denom_e
  omega_j = e_j / denom_e

  w_in_aug_p = omega_i * w_in_aug_i + omega_j * w_in_aug_j
  a_p = float(np.linalg.norm(w_in_aug_p))
  u_p = w_in_aug_p / (a_p + _UNIT_VEC_GUARD)

  r_i = float(np.dot(w_in_aug_i, u_p)) / (a_p + _UNIT_VEC_GUARD)
  r_j = float(np.dot(w_in_aug_j, u_p)) / (a_p + _UNIT_VEC_GUARD)

  w_out_p = r_i * w_out_i_f64 + r_j * w_out_j_f64
  eff_n_p = w_in_aug_p[:-1]
  bias_eff_p = float(w_in_aug_p[-1])

  gamma_i_f, beta_i_f = float(gamma_i), float(beta_i)
  gamma_j_f, beta_j_f = float(gamma_j), float(beta_j)

  gamma_new = math.sqrt(
      max(
          0.0,
          (omega_i * gamma_i_f) ** 2
          + (omega_j * gamma_j_f) ** 2
          + 2.0 * omega_i * omega_j * rho_ws * abs(gamma_i_f) * abs(gamma_j_f),
      )
  )
  beta_new = omega_i * beta_i_f + omega_j * beta_j_f
  var_new = max(0.0, gamma_new**2 - hope.BN_EPSILON)
  scale_new = gamma_new / math.sqrt(var_new + hope.BN_EPSILON)
  mu_new_scaled = (beta_new - bias_eff_p) / (scale_new + _UNIT_VEC_GUARD)
  bare_k_n = (eff_n_p / (scale_new + hope.RAW_WEIGHT_DIV_GUARD)).astype(
      k_a.dtype
  )
  new_kb_slice_i = w_out_p.astype(k_b.dtype).reshape(k_b4.shape[0:2] + (-1,))

  curriculum.append(
      hope.InstructionPacket(
          action="MERGE",
          indices=(i, j, conv_a),
          quadruplet=quadruplet,
          fan_in=fan_in,
          producer_bn_before=(
              gamma_i,
              beta_i,
              mean_i,
              var_i,
              gamma_j,
              beta_j,
              mean_j,
              var_j,
          ),
          consumer_mean_before=None,
          joint_rows_before=(row_i_before, row_j_before),
      )
  )

  nz_before = (
      int(np.count_nonzero(k_a[..., i]))
      + int(np.count_nonzero(k_a[..., j]))
      + int(np.count_nonzero(k_b4[:, :, i, :]))
      + int(np.count_nonzero(k_b4[:, :, j, :]))
  )
  for param in ("gamma", "beta", "moving_mean", "moving_variance"):
    arr = current_weights[f"{bn_a}/{param}"]
    if arr[i] != 0.0:
      nz_before += 1
    if arr[j] != 0.0:
      nz_before += 1

  k_a[..., i] = bare_k_n.reshape(k_a.shape[:-1])
  k_a[..., j] = 0.0
  k_b4[:, :, i, :] = new_kb_slice_i
  k_b4[:, :, j, :] = 0.0

  current_weights[f"{bn_a}/gamma"][i] = np.float32(gamma_new)
  current_weights[f"{bn_a}/beta"][i] = np.float32(beta_new)
  current_weights[f"{bn_a}/moving_mean"][i] = np.float32(mu_new_scaled)
  current_weights[f"{bn_a}/moving_variance"][i] = np.float32(var_new)
  for param in ("gamma", "beta", "moving_mean", "moving_variance"):
    current_weights[f"{bn_a}/{param}"][j] = 0.0

  nz_after = int(np.count_nonzero(k_a[..., i])) + int(
      np.count_nonzero(k_b4[:, :, i, :])
  )
  for param in ("gamma", "beta", "moving_mean", "moving_variance"):
    if current_weights[f"{bn_a}/{param}"][i] != 0.0:
      nz_after += 1

  cost_manager.remove_neuron_idx(conv_a, j)
  cost_manager.invalidate_neuron(conv_a, i)

  # Refresh cached weight-space features for survivor slot i and push new
  # PRUNE/MERGE candidates.
  feats = ws_layer_cache[conv_a]
  w_eff_i_live = (
      (
          k_a[..., i]
          * (
              current_weights[f"{bn_a}/gamma"][i]
              / np.sqrt(
                  np.maximum(
                      current_weights[f"{bn_a}/moving_variance"][i],
                      np.float32(0.0),
                  )
                  + np.float32(hope.BN_EPSILON)
              )
          )
      )
      .reshape(-1)
      .astype(np.float64)
  )
  scale_i_live = float(current_weights[f"{bn_a}/gamma"][i]) / math.sqrt(
      max(float(current_weights[f"{bn_a}/moving_variance"][i]), 0.0)
      + hope.BN_EPSILON
  )
  b_eff_i_live = float(
      current_weights[f"{bn_a}/beta"][i]
  ) - scale_i_live * float(current_weights[f"{bn_a}/moving_mean"][i])
  w_in_aug_i_live = np.append(w_eff_i_live, b_eff_i_live)
  w_out_i_live = k_b4[:, :, i, :].reshape(-1).astype(np.float64)

  a_i_live = float(np.linalg.norm(w_in_aug_i_live))
  b_i_live = float(np.linalg.norm(w_out_i_live))
  feats["W_in_aug"][i] = w_in_aug_i_live
  feats["W_out"][i] = w_out_i_live
  feats["a_vec"][i] = a_i_live
  feats["b_vec"][i] = b_i_live
  feats["e_vec"][i] = a_i_live * b_i_live
  feats["U_in"][i] = w_in_aug_i_live / (a_i_live + _UNIT_VEC_GUARD)

  norm_ws = cost_manager.baseline_norms[conv_a]["WS_COSINE_MERGE"]
  _push_ws_cosine_single_neuron_updates(
      conv_a, i, fan_in, feats, norm_ws, cost_manager
  )

  return nz_after - nz_before


# =====================================================================================================================
# 4. Baseline encoder public API
# =====================================================================================================================


def baseline_encode(
    weights: Dict[str, np.ndarray],
    quadruplets: Sequence[Tuple[str, ...]],
    benchmark_method: str = "L1_IN",
    use_normalized_baselines: bool = True,
    eval_fn: Optional[Callable[[Dict[str, np.ndarray]], float]] = None,
    eval_stride: int = 10,
    partial_encoding_steps: Optional[int] = None,
    output_file: Optional[str] = None,
) -> Tuple[
    Dict[str, np.ndarray],
    List[hope.InstructionPacket],
    List[Tuple[float, float]],
]:
  """Runs progressive structured neuron compression on ResNet-50 for one of the seven baselines.

  Supported `benchmark_method` values:
    - `"L1_IN"`:             L1-Norm Input Pruning                (default file:
    `results_L1_IN.csv`)
    - `"L1_JOINT"`:          L1-Norm Joint Pruning                (default file:
    `results_L1_JOINT.csv`)
    - `"BN_SCALE"`:          BatchNorm Scale Pruning              (default file:
    `results_BN_SCALE.csv`)
    - `"FPGM"`:              Filter Pruning via Geometric Median  (default file:
    `results_FPGM.csv`)
    - `"L1_JOINT_BIASCORR"`: Bias-Corrected Joint L1 Pruning      (default file:
    `results_L1_JOINT_BIASCORR.csv`)
    - `"WS_COSINE_MERGE"`:   Weight-Space Cosine Neuron Merging   (default file:
    `results_WS_COSINE_MERGE.csv`)
    - `"SYNFLOW"`:           Data-Free Synaptic Flow Pruning      (default file:
    `results_SYNFLOW.csv`)

  Args:
    weights: Dictionary mapping Keras weight paths to NumPy arrays. Not
      modified.
    quadruplets: Bottleneck quadruplets `(conv_a, bn_a, conv_b, bn_b)` (e.g.
      from `find_bottleneck_quadruplets(weights)`), or 3-tuples `(conv_a, bn_a,
      conv_b)`.
    benchmark_method: One of `BASELINE_METHODS` (or alias `"L1_JOINT_P1A"`).
    use_normalized_baselines: If True (default), normalizes each neuron's score
      by the layer's initial total score so initial layer scores sum to 1. If
      False, uses unnormalized raw scores.
    eval_fn: Optional callback `eval_fn(current_weights) -> float` invoked at
      `step < 5`, every `eval_stride` steps, and on the final compressed network
      to record `(density, accuracy)` pairs.
    eval_stride: Step interval between `eval_fn` evaluations (default: 10; 0
      disables evaluation).
    partial_encoding_steps: Optional cap on the number of compression steps.
    output_file: Path to save `(density, accuracy)` pairs when `eval_stride > 0`
      and `eval_fn is not None`. If `None` (default), uses
      `results_<benchmark_method>.csv`. Pass `""` to disable file output.

  Returns:
    `(compressed_weights, curriculum, results)` matching `hope.hope_encode`,
    where
    `curriculum` is in decoding order and can be losslessly decoded via
    `hope.hope_decode`.
  """
  if benchmark_method == "L1_JOINT_P1A":
    benchmark_method = "L1_JOINT_BIASCORR"
  if benchmark_method not in BASELINE_METHODS:
    raise ValueError(
        f"Unknown benchmark_method {benchmark_method!r}. Expected one of"
        f" {BASELINE_METHODS}."
    )

  if output_file is None:
    resolved_output_file = (
        DEFAULT_BASELINE_FILES[benchmark_method]
        if (eval_stride > 0 and eval_fn is not None)
        else None
    )
  elif not output_file:
    resolved_output_file = None
  else:
    resolved_output_file = output_file

  t_start = time.time()
  current_weights = {k: v.copy() for k, v in weights.items()}
  curriculum: List[hope.InstructionPacket] = []
  cost_manager = BaselineCostManager()

  normalized_quadruplets: List[hope.Quadruplet] = []
  for t in quadruplets:
    if len(t) == 3:
      c_a, bn_a, c_b = t
      bn_b = c_b.replace("_conv", "_bn")
      normalized_quadruplets.append((c_a, bn_a, c_b, bn_b))
    else:
      normalized_quadruplets.append((t[0], t[1], t[2], t[3]))
  quadruplet_of = {q[0]: q for q in normalized_quadruplets}

  total_initial_neurons = 0
  total_params_initial = sum(int(np.count_nonzero(v)) for v in weights.values())
  current_params = total_params_initial

  ws_layer_cache: Dict[str, Dict[str, np.ndarray]] = {}

  for quadruplet in normalized_quadruplets:
    conv_a_name = quadruplet[0]
    if f"{conv_a_name}/kernel" not in weights:
      continue

    joint_weights, fan_in, producer_bn = hope.read_quadruplet(
        current_weights, quadruplet
    )
    e_weights = float(np.sum(joint_weights**2))
    e_aux = float(
        np.sum(producer_bn["gamma"] ** 2) + np.sum(producer_bn["beta"] ** 2)
    )
    initial_layer_energy = e_weights + e_aux

    fpgm_scores: Optional[np.ndarray] = None
    ws_feats: Optional[Dict[str, np.ndarray]] = None

    if benchmark_method == "FPGM":
      w_in_f64 = joint_weights[:, :fan_in].astype(np.float64)
      fpgm_scores = np.sum(
          distance.cdist(w_in_f64, w_in_f64, metric="euclidean"), axis=1
      )
    elif benchmark_method == "WS_COSINE_MERGE":
      ws_feats = _compute_ws_cosine_layer_features(
          joint_weights, fan_in, producer_bn
      )
      ws_layer_cache[conv_a_name] = ws_feats

    baseline_stats: Dict[str, float] = {}
    if use_normalized_baselines:
      if benchmark_method == "L1_IN":
        baseline_stats["L1_IN"] = float(
            np.sum(np.abs(joint_weights[:, :fan_in]))
        )
      elif benchmark_method == "L1_JOINT":
        baseline_stats["L1_JOINT"] = float(np.sum(np.abs(joint_weights)))
      elif benchmark_method == "BN_SCALE":
        baseline_stats["BN_SCALE"] = float(np.sum(np.abs(producer_bn["gamma"])))
      elif benchmark_method == "FPGM":
        assert fpgm_scores is not None
        baseline_stats["FPGM"] = float(np.sum(fpgm_scores))
      elif benchmark_method == "L1_JOINT_BIASCORR":
        baseline_stats["L1_JOINT_BIASCORR"] = float(
            np.sum(np.abs(joint_weights))
        )
      elif benchmark_method == "WS_COSINE_MERGE":
        assert ws_feats is not None
        baseline_stats["WS_COSINE_MERGE"] = float(np.sum(ws_feats["e_vec"]))
      elif benchmark_method == "SYNFLOW":
        l1_in_vec = np.sum(
            np.abs(joint_weights[:, :fan_in]), axis=1, dtype=np.float64
        )
        l1_out_vec = np.sum(
            np.abs(joint_weights[:, fan_in:]), axis=1, dtype=np.float64
        )
        baseline_stats["SYNFLOW"] = float(np.sum(l1_in_vec * l1_out_vec))
    else:
      for m_name in BASELINE_METHODS:
        baseline_stats[m_name] = 1.0

    num_neurons = joint_weights.shape[0]
    total_initial_neurons += num_neurons
    cost_manager.register_layer(
        conv_a_name, num_neurons, initial_layer_energy, baseline_stats
    )

    active_indices = list(cost_manager.active_neurons[conv_a_name])
    for i in active_indices:
      cost = compute_baseline_prune_cost(
          benchmark_method,
          i,
          joint_weights,
          fan_in,
          producer_bn,
          baseline_norms=baseline_stats,
          fpgm_scores=fpgm_scores,
      )
      cost_manager.push_action(cost, "PRUNE", conv_a_name, (i,), fan_in)

    if benchmark_method == "WS_COSINE_MERGE":
      assert ws_feats is not None
      _push_ws_cosine_all_pairs(
          conv_a_name,
          fan_in,
          ws_feats,
          baseline_stats["WS_COSINE_MERGE"],
          cost_manager,
      )

  logger.info(
      "[%s] Baseline initialization complete in %.2fs (%d neurons,"
      " normalized=%s).",
      benchmark_method,
      time.time() - t_start,
      total_initial_neurons,
      use_normalized_baselines,
  )

  step_counter = 0
  results: List[Tuple[float, float]] = []

  while True:
    if (
        partial_encoding_steps is not None
        and step_counter >= partial_encoding_steps
    ):
      break

    if step_counter > 0 and step_counter % 1000 == 0:
      cost_manager.clean_heap()

    best_entry = cost_manager.pop_best_valid_action()
    if best_entry is None:
      break

    if eval_stride > 0 and (
        step_counter % eval_stride == 0 or step_counter < 5
    ):
      # An all-zero input network has total_params_initial == 0; report
      # density 0.0 for it instead of dividing by zero.
      density = (
          float(current_params) / float(total_params_initial)
          if total_params_initial > 0
          else 0.0
      )
      acc = float(eval_fn(current_weights)) if eval_fn is not None else 0.0
      results.append((np.float64(density), np.float64(acc)))
      if resolved_output_file is not None and eval_fn is not None:
        _write_density_accuracy_file(resolved_output_file, results)

    quadruplet = quadruplet_of[best_entry.layer_name]
    fan_in = best_entry.fan_in

    if best_entry.action_type == "PRUNE":
      i = best_entry.location[0]
      current_params += _execute_baseline_prune(
          current_weights,
          quadruplet,
          i,
          fan_in,
          benchmark_method,
          cost_manager,
          curriculum,
      )
    elif best_entry.action_type == "MERGE":
      i, j = best_entry.location
      current_params += _execute_baseline_merge(
          current_weights,
          quadruplet,
          i,
          j,
          fan_in,
          ws_layer_cache,
          cost_manager,
          curriculum,
      )

    step_counter += 1

  if eval_stride > 0 and step_counter > 0:
    final_density = (
        float(current_params) / float(total_params_initial)
        if total_params_initial > 0
        else 0.0
    )
    final_acc = float(eval_fn(current_weights)) if eval_fn is not None else 0.0
    results.append((np.float64(final_density), np.float64(final_acc)))

  if (
      resolved_output_file is not None
      and eval_fn is not None
      and len(results) > 0
  ):
    _write_density_accuracy_file(resolved_output_file, results)

  curriculum.reverse()
  return current_weights, curriculum, results


# Backward-compatible alias used in scripts and notebooks.
HOPE_Baseline_Encoder = baseline_encode  # pylint: disable=invalid-name


def run_all_baselines(
    weights: Dict[str, np.ndarray],
    quadruplets: Optional[Sequence[Tuple[str, ...]]] = None,
    eval_fn: Optional[Callable[[Dict[str, np.ndarray]], float]] = None,
    eval_stride: int = 10,
    use_normalized_baselines: bool = True,
    output_dir: str = ".",
    methods: Optional[Sequence[str]] = None,
) -> Dict[str, List[Tuple[float, float]]]:
  """Runs the specified baseline methods (default: all 7 in `BASELINE_METHODS`) and saves their CSVs.

  Args:
    weights: Dictionary mapping Keras weight paths to NumPy arrays.
    quadruplets: Optional sequence of bottleneck quadruplets; discovered
      automatically via `find_bottleneck_quadruplets(weights)` when omitted.
    eval_fn: Optional evaluation callback `eval_fn(current_weights) -> float`.
    eval_stride: Step interval between evaluations (default: 10).
    use_normalized_baselines: Whether to normalize initial layer scores to sum
      to 1 (default: True).
    output_dir: Directory in which to write `results_<method>.csv` files.
    methods: Optional subset of `BASELINE_METHODS` to run.

  Returns:
    Dictionary mapping each method name to its `(density, accuracy)` trajectory
    list.
  """
  os.makedirs(output_dir, exist_ok=True)
  if quadruplets is None:
    quadruplets = hope.find_bottleneck_quadruplets(weights)
  selected_methods = tuple(methods) if methods is not None else BASELINE_METHODS
  all_results: Dict[str, List[Tuple[float, float]]] = {}
  for method in selected_methods:
    out_path = (
        os.path.join(output_dir, DEFAULT_BASELINE_FILES[method])
        if (eval_stride > 0 and eval_fn is not None)
        else None
    )
    _, _, res = baseline_encode(
        weights,
        quadruplets,
        benchmark_method=method,
        use_normalized_baselines=use_normalized_baselines,
        eval_fn=eval_fn,
        eval_stride=eval_stride,
        output_file=out_path,
    )
    all_results[method] = res
  return all_results
