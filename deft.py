# -*- coding: utf-8 -*-
# Copyright 2026 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Licensed under the Apache License, Version 2.0 (the "License");
# http://www.apache.org/licenses/LICENSE-2.0
"""DEFT (Dispersed Elastic Fine-Tuning): cross-domain transfer learning (CIFAR-100 -> SVHN) instantiation.

Data-free structural partitioning of a pretrained neural network into a frozen
Universal Core and a plastic Slack via
HOPE's progressive encoding, coupled with a structural initialization mask and
binary gradient elasticity for
parameter-efficient cross-domain transfer learning without catastrophic
forgetting. Includes the full CIFAR-100 -> SVHN
cross-domain benchmark and four transfer learning baselines (Head-Only FT,
Standard Full FT, EWC, and PEFT / BN-Tuning).

Method
------
DEFT resolves the stability-plasticity dilemma in transfer learning by
converting HOPE's data-free progressive
compression curriculum into a hard neuron partition and an exact subspace
projection (Section 10.2, Appendix I, and
Algorithm 3 of the HOPE paper, https://arxiv.org/abs/2607.21366):

  1. Progressive encoding and pruning-cost pool (Section 10.2.2, Algorithm 3
  Phase 1):
     Run HOPE's data-free progressive encoder (`hope.hope_encode`) on the
     pretrained
     source parameters theta_src until
     every hidden neuron has been removed by a PRUNE or MERGE action. Let M
     denote the set of all neurons that ever
     participate in a MERGE step (either as the surviving merge root i or as the
     absorbed neuron j). Because DEFT
     fine-tunes the original source weights rather than deploying HOPE's
     synthesized merged parents, both members of
     every merged pair are placed in the frozen core (M subseteq F_core). For
     every unmerged neuron i not in M, HOPE
     records its mean-centred pruning cost upon removal:
       J_prune^(i) = N^(i) ||f_i||_o / E_b^(i),
     where N^(i) is the active layer width (including i), ||f_i||_o =
     ||w_out,i||_2 sqrt(v_i) is the mean-centred
     Hilbert-Schmidt capacity evaluated on the original source parameters, and
     E_b^(i) = E_a^(i) - ||f_i||_o is the
     remaining layer capacity. Excluding extinction artifacts (the last neuron
     of a layer, where E_b^(i) <= eps with
     eps = 1e-5), the global pruning-cost pool is:
       C = { J_prune^(i) | i not in M, E_b^(i) > eps }   (eps = 1e-5).

  2. Binary elasticity and Core/Slack partition (Eq. deft_elasticity, Definition
  I.1):
     Given a target percentile P in (0, 100], let J_P = Percentile(C, P) (with
     J_P = 0 if C is empty). Each backbone
     neuron i is assigned the binary elasticity:
       E_i = 1   if i not in M, E_b^(i) > eps, and J_prune^(i) < J_P,
       E_i = 0   otherwise.
     This induces a hard partition of every hidden layer into a frozen Universal
     Core F_core = {i : E_i = 0}
     (high-cost unmerged neurons, extinction-guarded last neurons, and all merge
     participants M) and a plastic Slack
     F_slack = {j : E_j = 1} (low-cost unmerged neurons).

  3. Consistency at initialization via the structural mask (Axiom I.1, Theorem
  I.1, Algorithm 3 Phase 2):
     At initialization t = 0, DEFT severs all weight projections from upstream
     slack neurons (E_in,j > 0) to
     downstream core neurons (E_out,u = 0):
       M_{u,j} = 0   if E_in,j > 0 and E_out,u = 0,
       M_{u,j} = 1   otherwise,
     setting W_0 = M (o) W_source for all hidden layers (the first layer, whose
     inputs are image channels, is unmasked,
     and the target classification head is freshly initialized with E = 1). The
     same structural mask is applied to the
     original source classification head (keeping its bias), which therefore
     reads only core features of the penultimate
     layer. By Theorem I.1 (Static Initialization Shock Bound), under HOPE's
     Gaussian surrogate the perturbation
     injected into the downstream core of layer l + 1 is bounded by sum_{j in
     F_slack^(l)} ||f_j||_H <= |F_slack^(l)| tau^(l).

  4. Binary gradient elasticity and exact source preservation (Axioms I.2-I.3,
  Theorem I.2, Corollary I.1, Algorithm 3 Phase 3):
     At every target training step t, the gradient of the regularized target
     loss L_target (which includes the 5e-4 L2
     penalty on tensors of rank > 1) with respect to every trainable parameter
     vector theta_k of destination neuron k
     (its incoming kernel and its BN scale gamma_k and shift beta_k) is
     multiplied by E_k before the optimizer update:
       g_tilde_k^(t) = E_k * grad_{theta_k} L_target(theta^(t)).
     Because E_u = 0 on F_core and the optimizer (SGD with momentum 0.9) starts
     from a zero velocity state, all core
     parameters and all severed slack-to-core weights remain identically equal
     to their t = 0 values at every step t >= 0.
     When evaluated with the masked source head and the original source BN
     running statistics (Axiom I.3), the source
     logits are invariant across all training steps (Theorem I.2: Dynamic
     Decoupling), so source retention is fixed at
     initialization by the partition alone (Corollary I.1).

Code map
--------
  1. Runtime environment and dataset loaders   load_and_partition_cifar100,
  load_and_partition_svhn,
                                               generate_multiclass_domain_tracks,
                                               get_multiclass_cifar_task,
                                               get_10class_svhn_task
  2. Architecture and quadruplet topology      create_cnn (8-layer BN-VGG with
  GAP), find_prunable_blocks
  3. HOPE progressive encoder wrapper          CNN8_FEATURE_MAP_SIZE,
                                               encode_source_curriculum
                                               (delegates to hope.hope_encode)
  4. DEFT partition, structural mask, and step generate_elasticity_map,
  generate_target_elasticity_list,
                                               apply_null_space_initialization,
                                               precompute_masked_source_head,
                                               get_source_eval_weights_fast,
                                               train_step_deft_jit
  5. Transfer baselines (Full, Head, EWC, PEFT) init_target_task_weights,
  generate_head_only_mask,
                                               generate_peft_mask,
                                               generate_ewc_mask,
                                               compute_diagonal_fim,
                                               train_step_ewc_jit,
                                               get_source_eval_weights
  6. Hyperparameter pre-sweeps and benchmark   find_optimal_deft_hyperparams,
  find_optimal_finetune_lr,
                                               find_optimal_head_only_lr,
                                               find_optimal_ewc_hyperparams,
                                               find_optimal_peft_lr,
                                               parse_args, run_deft_benchmark

Reference benchmark results (20 cross-domain scenarios: CIFAR-100 20-class ->
SVHN 10-class)
--------------------------------------------------------------------------------------------
Evaluated across 4 independent trials x 5 tasks = 20 cross-domain scenarios on
the 8-layer BN-VGG architecture
(`CNN_8Layer`, Appendix H.2), reporting test-set accuracy at the peak target
validation epoch (mean +/- standard
deviation across scenarios; Table 2 in Section 10.2.4, with P* = 30 and eta* =
0.005 selected by the pre-sweep):

  Method              Target Acc (SVHN)    Source Retention (CIFAR-100)
  H-Score
  DEFT (Ours)          89.88 +/- 0.59%            61.18 +/- 4.05%          72.72
  +/- 2.77%
  Head-Only FT         36.01 +/- 3.06%            62.74 +/- 3.84%          45.63
  +/- 2.47%
  Standard Full FT     92.66 +/- 0.27%            12.61 +/- 2.78%          22.08
  +/- 4.32%
  EWC*                 84.38 +/- 19.37%           41.98 +/- 11.49%         55.81
  +/- 13.95%
  PEFT (BN-Tuning)     78.99 +/- 0.98%             6.80 +/- 1.70%          12.46
  +/- 2.84%

  * EWC (lambda* = 1e7, eta* = 1e-4) diverged to NaN at epoch 1 in 1 of 20
    scenarios (counted as 0; over the 19 non-diverged scenarios it reaches
    88.82 +/- 0.66% target, 44.19 +/- 6.42% source, and 58.75 +/- 5.68%
    H-Score).

Companion code to the paper "Hilbert Operator for Progressive Encoding (HOPE): A
Mathematical Framework for
Deconstructing Learned Representations in Deep Networks" by H. Mobahi and P. L.
Bartlett.

Example
-------
  # 1. Quick end-to-end verification (~2 minutes on CPU, skips pre-sweeps):
  python deft.py --smoke_test

  # 2. Run the complete 20-scenario cross-domain benchmark (pre-sweep + 4 trials
  x 5 tasks):
  python deft.py

  # 3. Programmatic DEFT partitioning and fine-tuning on a trained source_model
  and target_model:
  paired_layers = find_prunable_blocks()
  source_weights = {w.path: w.numpy() for w in source_model.trainable_weights +
  source_model.non_trainable_weights}
  _, curriculum = encode_source_curriculum(source_weights, paired_layers)

  elasticity_map = generate_elasticity_map(curriculum, source_model,
  percentile_threshold=30.0, paired_layers=paired_layers)
  deft_grad_mask = generate_target_elasticity_list(elasticity_map, source_model,
  target_model, paired_layers)
  t_weights = apply_null_space_initialization(orig_source_t_weights,
  elasticity_map, source_model, target_model, paired_layers)
  masked_src_head = precompute_masked_source_head(orig_source_t_weights,
  source_model, elasticity_map, paired_layers)

Requires numpy >= 2.0, scipy >= 1.13, jax, optax, keras >= 3.5, tensorflow >=
2.18, and tensorflow-datasets >= 4.9.
"""

import argparse
import functools
import os
import tempfile
import time
import types
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

# ==============================================================================
# 0. AUTOMATIC HARDWARE & RUNTIME CONFIGURATION
# ==============================================================================


def _host_has_nvidia_gpu() -> bool:
  """Probes whether an NVIDIA GPU device is exposed to the current host or container."""
  if os.environ.get("DEFT_FORCE_CPU", "0") == "1":
    return False
  return os.path.exists("/dev/nvidia0") or os.path.exists(
      "/proc/driver/nvidia/version"
  )


def _configure_runtime_environment() -> None:
  """Configures environment variables for JAX, Keras, and XLA before backend initialization."""
  os.environ.setdefault("KERAS_BACKEND", "jax")
  os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
  os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

  if "KERAS_HOME" not in os.environ:
    default_keras_home = os.path.expanduser("~/.keras")
    try:
      os.makedirs(default_keras_home, exist_ok=True)
      test_path = os.path.join(default_keras_home, ".write_test")
      with open(test_path, "w") as f:
        f.write("ok")
      os.remove(test_path)
    except Exception:  # pylint: disable=broad-exception-caught
      os.environ["KERAS_HOME"] = tempfile.mkdtemp(prefix="keras_cache_")

  if _host_has_nvidia_gpu():
    xla_flags = os.environ.get("XLA_FLAGS", "")
    if "--xla_gpu_enable_command_buffer=" not in xla_flags:
      os.environ["XLA_FLAGS"] = (
          f"{xla_flags} --xla_gpu_enable_command_buffer=".strip()
      )
  else:
    cpu_threads = os.environ.get("OMP_NUM_THREADS", "8")
    os.environ.setdefault("OMP_NUM_THREADS", cpu_threads)
    os.environ.setdefault("MKL_NUM_THREADS", cpu_threads)
    os.environ.setdefault("OPENBLAS_NUM_THREADS", cpu_threads)
    if "XLA_FLAGS" not in os.environ:
      os.environ["XLA_FLAGS"] = (
          "--xla_cpu_multi_thread_eigen=true"
          f" intra_op_parallelism_threads={cpu_threads}"
      )


_configure_runtime_environment()

# pylint: disable=g-import-not-at-top
import hope
import jax
import jax.numpy as jnp
import keras
from keras import regularizers
import numpy as np
import optax
import tensorflow as tf
import tensorflow_datasets as tfds
# pylint: enable=g-import-not-at-top


try:
  tf.config.set_visible_devices([], "GPU")
except Exception:  # pylint: disable=broad-exception-caught
  pass


def detect_hardware_backend(verbose: bool = True) -> Dict[str, Any]:
  """Automatically detects CPU vs. GPU hardware and returns backend metadata."""
  try:
    devices = jax.devices()
  except RuntimeError:
    devices = jax.devices("cpu")
  platform = devices[0].platform if devices else "cpu"
  is_gpu = any(d.platform == "gpu" for d in devices)
  device_kind = (
      getattr(devices[0], "device_kind", str(devices[0])) if devices else "CPU"
  )
  info = {
      "backend": platform.upper(),
      "is_gpu": is_gpu,
      "num_devices": len(devices),
      "device_kind": device_kind,
      "devices": devices,
  }
  if verbose:
    print(
        f"[DEFT Hardware Auto-Detect] Backend: {info['backend']} "
        f"({info['num_devices']}x {info['device_kind']})",
        flush=True,
    )
  return info


HARDWARE_INFO = detect_hardware_backend(verbose=False)
HARDWARE_BACKEND = HARDWARE_INFO["backend"]
IS_GPU = HARDWARE_INFO["is_gpu"]
ACTIVE_DEVICES = HARDWARE_INFO["devices"]


# ==============================================================================
# 1. DATASET LOADERS & CROSS-DOMAIN TASK GENERATION
# ==============================================================================


def _get_tfds_data_dir() -> str:
  """Returns the local TensorFlow Datasets (TFDS) directory."""
  env_dir = os.environ.get("TFDS_DATA_DIR", "").strip()
  if env_dir:
    return env_dir
  candidates = [
      os.path.expanduser("~/tensorflow_datasets"),
      "/root/tensorflow_datasets",
      "/tmp/tensorflow_datasets",
  ]
  for cand in candidates:
    try:
      cifar_dir = os.path.join(cand, "cifar100")
      svhn_dir = os.path.join(cand, "svhn_cropped")
      if (
          os.path.isdir(cand)
          and os.access(cand, os.R_OK)
          and os.path.isdir(cifar_dir)
          and os.listdir(cifar_dir)
          and os.path.isdir(svhn_dir)
          and os.listdir(svhn_dir)
      ):
        return cand
    except OSError:
      continue
  for cand in candidates:
    try:
      if (
          os.path.isdir(cand)
          and os.access(cand, os.R_OK | os.W_OK)
          and os.listdir(cand)
      ):
        return cand
    except OSError:
      continue
  return os.path.expanduser("~/tensorflow_datasets")


def load_and_partition_svhn(max_tfds_examples: Optional[int] = None):
  """Loads cropped SVHN (32x32x3 in [0, 1]) and creates an 80/20 train/validation split (Appendix H.1)."""
  tfds_dir = _get_tfds_data_dir()
  print(f"Loading SVHN from: {tfds_dir}")

  train_split = (
      "train" if max_tfds_examples is None else f"train[:{max_tfds_examples}]"
  )
  test_split = (
      "test"
      if max_tfds_examples is None
      else f"test[:{max(512, max_tfds_examples // 4)}]"
  )
  ds_train = tfds.load(
      "svhn_cropped", split=train_split, data_dir=tfds_dir, download=True
  )
  ds_test = tfds.load(
      "svhn_cropped", split=test_split, data_dir=tfds_dir, download=True
  )

  def process_data(ex):
    return tf.cast(ex["image"], tf.float32) / 255.0, ex["label"]

  ds_train = ds_train.map(process_data, num_parallel_calls=tf.data.AUTOTUNE)
  ds_test = ds_test.map(process_data, num_parallel_calls=tf.data.AUTOTUNE)

  train_data = list(tfds.as_numpy(ds_train))
  test_data = list(tfds.as_numpy(ds_test))

  x_train_all, y_train_all = zip(*train_data)
  x_test, y_test = zip(*test_data)

  x_train_all = np.array(x_train_all)
  y_train_all = np.array(y_train_all).astype(np.int32)

  rng = np.random.default_rng(42)
  indices = rng.permutation(len(x_train_all))
  split_idx = int(0.8 * len(indices))
  train_idx, val_idx = indices[:split_idx], indices[split_idx:]

  return (
      (x_train_all[train_idx], y_train_all[train_idx]),
      (x_train_all[val_idx], y_train_all[val_idx]),
      (np.array(x_test), np.array(y_test).astype(np.int32)),
  )


def generate_multiclass_domain_tracks(
    coarse_to_fine, num_classes=10, tasks_per_track=5, seed=42
):
  """Samples multi-class CIFAR-100 source tasks (from random superclasses) paired with 10-class SVHN (Appendix H.1)."""
  rng = np.random.default_rng(seed)
  superclasses = list(coarse_to_fine.keys())
  cross_tasks = []

  for _ in range(tasks_per_track):
    # Each CIFAR-100 superclass contains 5 fine classes
    num_superclasses = num_classes // 5
    sc_src_list = rng.choice(superclasses, size=num_superclasses, replace=False)

    src_classes = []
    for sc in sc_src_list:
      src_classes.extend(coarse_to_fine[sc])

    target_task = "All 10 SVHN Digits"
    cross_tasks.append((tuple(src_classes), target_task))

  return cross_tasks


def get_10class_svhn_task(
    x_train_full,
    y_train_full,
    x_val_full,
    y_val_full,
    x_test_full,
    y_test_full,
    max_train_samples=None,
):
  """Returns the 10-class SVHN splits, optionally capping the training set size during hyperparameter pre-sweeps."""
  if max_train_samples is not None:
    return (
        (x_train_full[:max_train_samples], y_train_full[:max_train_samples]),
        (x_val_full, y_val_full),
        (x_test_full, y_test_full),
    )
  return (
      (x_train_full, y_train_full),
      (x_val_full, y_val_full),
      (x_test_full, y_test_full),
  )


def load_and_partition_cifar100(max_tfds_examples: Optional[int] = None):
  """Loads CIFAR-100 (32x32x3 in [0, 1]), extracts the superclass taxonomy, and creates an 80/20 train/val split."""
  tfds_dir = _get_tfds_data_dir()
  print(f"Loading CIFAR-100 from: {tfds_dir}")

  train_split = (
      "train" if max_tfds_examples is None else f"train[:{max_tfds_examples}]"
  )
  test_split = (
      "test"
      if max_tfds_examples is None
      else f"test[:{max(512, max_tfds_examples // 4)}]"
  )
  ds_train = tfds.load(
      "cifar100", split=train_split, data_dir=tfds_dir, download=True
  )
  ds_test = tfds.load(
      "cifar100", split=test_split, data_dir=tfds_dir, download=True
  )

  def process_data(ex):
    return (
        tf.cast(ex["image"], tf.float32) / 255.0,
        ex["label"],
        ex["coarse_label"],
    )

  ds_train = ds_train.map(process_data, num_parallel_calls=tf.data.AUTOTUNE)
  ds_test = ds_test.map(process_data, num_parallel_calls=tf.data.AUTOTUNE)

  train_data = list(tfds.as_numpy(ds_train))
  test_data = list(tfds.as_numpy(ds_test))

  x_train_all, y_train_all, coarse_train_all = zip(*train_data)
  x_test, y_test, coarse_test_all = zip(*test_data)

  x_train_all = np.array(x_train_all)
  y_train_all = np.array(y_train_all).astype(np.int32)
  coarse_train_all = np.array(coarse_train_all).astype(np.int32)

  # Build the superclass-to-fine-class taxonomy map:
  # {superclass_id: [fine_class_1, ..., fine_class_5]}
  coarse_to_fine = {c: [] for c in range(20)}
  for fine, coarse in zip(y_train_all, coarse_train_all):
    if fine not in coarse_to_fine[coarse]:
      coarse_to_fine[coarse].append(int(fine))
  if max_tfds_examples is not None:
    for fine, coarse in zip(y_test, coarse_test_all):
      c_idx = int(coarse)
      f_idx = int(fine)
      if f_idx not in coarse_to_fine[c_idx]:
        coarse_to_fine[c_idx].append(f_idx)

  for c in coarse_to_fine:
    coarse_to_fine[c].sort()

  rng = np.random.default_rng(42)
  indices = rng.permutation(len(x_train_all))
  split_idx = int(0.8 * len(indices))
  train_idx, val_idx = indices[:split_idx], indices[split_idx:]

  return (
      (x_train_all[train_idx], y_train_all[train_idx]),
      (x_train_all[val_idx], y_train_all[val_idx]),
      (np.array(x_test), np.array(y_test).astype(np.int32)),
      coarse_to_fine,
  )


def get_multiclass_cifar_task(
    class_tuple,
    x_train_full,
    y_train_full,
    x_val_full,
    y_val_full,
    x_test_full,
    y_test_full,
):
  """Filters CIFAR-100 splits for the selected fine classes and remaps labels to 0...N-1."""
  class_array = np.array(class_tuple)

  train_mask = np.isin(y_train_full, class_array)
  val_mask = np.isin(y_val_full, class_array)
  test_mask = np.isin(y_test_full, class_array)

  x_train, y_train = x_train_full[train_mask], y_train_full[train_mask]
  x_val, y_val = x_val_full[val_mask], y_val_full[val_mask]
  x_test, y_test = x_test_full[test_mask], y_test_full[test_mask]

  y_train_mapped = np.zeros_like(y_train)
  y_val_mapped = np.zeros_like(y_val)
  y_test_mapped = np.zeros_like(y_test)

  for new_idx, old_val in enumerate(class_tuple):
    y_train_mapped[y_train == old_val] = new_idx
    y_val_mapped[y_val == old_val] = new_idx
    y_test_mapped[y_test == old_val] = new_idx

  return (
      (x_train, y_train_mapped),
      (x_val, y_val_mapped),
      (x_test, y_test_mapped),
  )


# ==============================================================================
# 2. ARCHITECTURE & QUADRUPLET TOPOLOGY (APPENDIX H.2)
# ==============================================================================


def create_cnn(input_shape=(32, 32, 3), num_classes=100):
  """Constructs the 8-layer BN-VGG architecture (`CNN_8Layer`) with Global Average Pooling (Appendix H.2)."""
  inputs = keras.Input(shape=input_shape)

  # Standard 5e-4 L2 weight decay regularization (Appendix H.3)
  l2_reg = regularizers.l2(5e-4)

  # --- Block 1 (32x32 -> 16x16) ---
  x = keras.layers.Conv2D(
      128,
      (3, 3),
      padding="same",
      use_bias=False,
      kernel_regularizer=l2_reg,
      name="conv1",
  )(inputs)
  x = keras.layers.BatchNormalization(gamma_regularizer=l2_reg, name="bn1")(x)
  x = keras.layers.Activation("relu", name="relu1")(x)
  x = keras.layers.MaxPooling2D(pool_size=(2, 2), name="pool1")(x)

  # --- Block 2 (16x16 -> 8x8) ---
  x = keras.layers.Conv2D(
      256,
      (3, 3),
      padding="same",
      use_bias=False,
      kernel_regularizer=l2_reg,
      name="conv2",
  )(x)
  x = keras.layers.BatchNormalization(gamma_regularizer=l2_reg, name="bn2")(x)
  x = keras.layers.Activation("relu", name="relu2")(x)
  x = keras.layers.MaxPooling2D(pool_size=(2, 2), name="pool2")(x)

  # --- Block 3 (8x8 -> 4x4) ---
  x = keras.layers.Conv2D(
      512,
      (3, 3),
      padding="same",
      use_bias=False,
      kernel_regularizer=l2_reg,
      name="conv3a",
  )(x)
  x = keras.layers.BatchNormalization(gamma_regularizer=l2_reg, name="bn3a")(x)
  x = keras.layers.Activation("relu", name="relu3a")(x)

  x = keras.layers.Conv2D(
      512,
      (3, 3),
      padding="same",
      use_bias=False,
      kernel_regularizer=l2_reg,
      name="conv3b",
  )(x)
  x = keras.layers.BatchNormalization(gamma_regularizer=l2_reg, name="bn3b")(x)
  x = keras.layers.Activation("relu", name="relu3b")(x)
  x = keras.layers.MaxPooling2D(pool_size=(2, 2), name="pool3")(x)

  # --- Block 4 (4x4 -> 2x2) ---
  x = keras.layers.Conv2D(
      512,
      (3, 3),
      padding="same",
      use_bias=False,
      kernel_regularizer=l2_reg,
      name="conv4a",
  )(x)
  x = keras.layers.BatchNormalization(gamma_regularizer=l2_reg, name="bn4a")(x)
  x = keras.layers.Activation("relu", name="relu4a")(x)

  x = keras.layers.Conv2D(
      512,
      (3, 3),
      padding="same",
      use_bias=False,
      kernel_regularizer=l2_reg,
      name="conv4b",
  )(x)
  x = keras.layers.BatchNormalization(gamma_regularizer=l2_reg, name="bn4b")(x)
  x = keras.layers.Activation("relu", name="relu4b")(x)
  x = keras.layers.MaxPooling2D(pool_size=(2, 2), name="pool4")(x)

  # --- Global Average Pooling Transition ---
  x = keras.layers.GlobalAveragePooling2D(name="gap")(x)

  # --- Bottleneck Representation Layer ---
  x = keras.layers.Dense(
      512, use_bias=False, kernel_regularizer=l2_reg, name="bottleneck"
  )(x)
  x = keras.layers.BatchNormalization(gamma_regularizer=l2_reg, name="bn5")(x)
  x = keras.layers.Activation("relu", name="relu5")(x)

  # --- Terminal Classification Head ---
  outputs = keras.layers.Dense(
      num_classes, kernel_regularizer=l2_reg, name="output"
  )(x)

  return keras.Model(inputs=inputs, outputs=outputs, name="CNN_8Layer")


def find_prunable_blocks(verbose: bool = True):
  """Returns the 7 sequential `(producer_layer, producer_bn, consumer_layer, consumer_bn)` quadruplets of `CNN_8Layer`.

  Args:
    verbose: If True, prints the number of configured layer quadruplets.
  """
  prunable_blocks = [
      ("conv1", "bn1", "conv2", "bn2"),
      ("conv2", "bn2", "conv3a", "bn3a"),
      ("conv3a", "bn3a", "conv3b", "bn3b"),
      ("conv3b", "bn3b", "conv4a", "bn4a"),
      ("conv4a", "bn4a", "conv4b", "bn4b"),
      ("conv4b", "bn4b", "bottleneck", "bn5"),  # GAP transition
      ("bottleneck", "bn5", "output", None),
  ]
  if verbose:
    print(
        f"Configured {len(prunable_blocks)} prunable layer quadruplets for"
        " CNN_8Layer."
    )

  return prunable_blocks


# ==============================================================================
# 3. HILBERT OPERATOR FOR PROGRESSIVE ENCODING (HOPE) WRAPPER
# ==============================================================================

# Spatial feature-map resolutions H of `CNN_8Layer` consumer convolutions on
# 32x32 inputs (Appendix H.2), passed to `hope.hope_encode` for 3x3 zero-padding
# tap fractions chi = r r^T with r = ((H - 1)/H, 1, (H - 1)/H).
CNN8_FEATURE_MAP_SIZE = types.MappingProxyType({
    "conv1": 32.0,
    "conv2": 16.0,
    "conv3a": 8.0,
    "conv3b": 8.0,
    "conv4a": 4.0,
    "conv4b": 4.0,
})


def encode_source_curriculum(
    trained_weights_dict: Dict[str, np.ndarray],
    paired_layers: Optional[Sequence[hope.Quadruplet]] = None,
) -> Tuple[Dict[str, np.ndarray], List[hope.InstructionPacket]]:
  """Runs the canonical data-free HOPE progressive encoder (`hope.hope_encode`) on `CNN_8Layer` weights.

  Delegates progressive pruning and 2->1 Hilbert neuron merging to `hope.py`
  with `CNN8_FEATURE_MAP_SIZE` tap fractions and `eval_stride=0` (strictly
  data-free, no intermediate validation forward passes).

  Args:
    trained_weights_dict: Mapping from Keras weight path (e.g. `"conv1/kernel"`,
      `"bn1/gamma"`) to NumPy parameter arrays of the pretrained source model.
      Not modified.
    paired_layers: Optional sequence of `(producer, producer_bn, consumer,
      consumer_bn)` quadruplets; defaults to `PAIRED_LAYERS`.

  Returns:
    Tuple `(compressed_weights, curriculum)`, where `curriculum` is the list of
    `hope.InstructionPacket` objects in decoding order (last encoder action
    first), ready for `generate_elasticity_map` or `hope.hope_decode`.
  """
  t_start = time.time()
  active_pairs = PAIRED_LAYERS if paired_layers is None else paired_layers
  compressed_weights, curriculum, _ = hope.hope_encode(
      trained_weights_dict,
      active_pairs,
      eval_stride=0,
      feature_map_sizes=CNN8_FEATURE_MAP_SIZE,
  )
  print(
      f"HOPE Encoder finished in {time.time() - t_start:.2f}s"
      f" ({len(curriculum)} steps)."
  )
  return compressed_weights, curriculum


# ==============================================================================
# 4. DEFT PARTITION, STRUCTURAL MASK, & TARGET TRAINING STEP
# ==============================================================================

BATCH_SIZE = 16
NUM_SOURCE_CLASSES = 20
HYPERPARAM_PHASE_TRAIN_SAMPLE_CAP = 15000
FINETUNE_EPOCHS = 30
NUM_TRIALS = 4
TASKS_PER_TRACK = 5

PAIRED_LAYERS = find_prunable_blocks(verbose=False)


def apply_null_space_initialization(
    orig_t_weights,
    elasticity_dict_source,
    source_model,
    target_model,
    paired_layers,
):
  """Applies the DEFT structural mask M at t = 0 to sever upstream-slack-to-downstream-core connections (Axiom I.1).

  Implements Phase 2 of Algorithm 3 (`alg:deft_pipeline`, lines 10-14) and Eq.
  1235 (Section 10.2.2, Figure 2):
  for every weight tensor connecting an upstream hidden layer (with elasticities
  E_in) to a downstream hidden layer
  (with elasticities E_out), the structural mask M sets
    M_{u,j} = 0   if E_{in,j} > 0 (upstream Slack) and E_{out,u} = 0 (downstream
    Core),
    M_{u,j} = 1   otherwise,
  yielding W_0 = M (o) W_source. The first convolutional layer (whose inputs are
  image channels) is unmasked, and the
  target classification head ('output') is freshly initialized from
  `target_model`. By Theorem I.1, the static
  perturbation injected into the downstream core of layer l + 1 is bounded by
  sum_{j in F_slack^(l)} ||f_j||_H.

  Args:
    orig_t_weights: List of pretrained source trainable weight arrays
      (`W_source`).
    elasticity_dict_source: Dict mapping source weight paths to 1D per-channel
      elasticity arrays.
    source_model: Pretrained source Keras model.
    target_model: Target Keras model providing the freshly initialized target
      head weights.
    paired_layers: Sequence of `(conv_a, bn_a, conv_b, bn_b)` layer quadruplets
      defining upstream/downstream pairs.

  Returns:
    List of initialized float32 JAX trainable weight arrays `W_0` for target
    adaptation.
  """
  upstream_map = {
      quad[2]: quad[0] for quad in paired_layers if quad[2] is not None
  }
  init_weights = []

  for w_src, w_tgt, w_src_ref in zip(
      orig_t_weights,
      target_model.trainable_weights,
      source_model.trainable_weights,
  ):
    if "output" in w_src_ref.path:
      init_weights.append(
          jnp.array(w_tgt.numpy())
      )  # Freshly initialized target head (Algorithm 3, line 10)
      continue

    w_init = jnp.array(w_src)

    if len(w_init.shape) >= 2:
      layer_name = w_src_ref.path.split("/")[0]
      upstream_layer = upstream_map.get(layer_name)
      upstream_path = f"{upstream_layer}/kernel" if upstream_layer else None

      if upstream_path and upstream_path in elasticity_dict_source:
        in_elasticity = elasticity_dict_source[upstream_path]
        out_elasticity = elasticity_dict_source[w_src_ref.path]

        # Structural mask M_{u,j} = 0 iff E_{in,j} > 0 (Slack) and
        # E_{out,u} = 0 (Core) (Eq. 1235, Axiom I.1).
        is_upstream_slack = (in_elasticity > 0.0).astype(jnp.float32)
        is_downstream_core = (out_elasticity == 0.0).astype(jnp.float32)

        if len(w_init.shape) == 4:  # Conv2D kernel: (H, W, C_in, C_out)
          slack_to_core_cut = is_upstream_slack.reshape(
              1, 1, -1, 1
          ) * is_downstream_core.reshape(1, 1, 1, -1)
        else:  # Dense kernel: (C_in, C_out)
          slack_to_core_cut = is_upstream_slack.reshape(
              -1, 1
          ) * is_downstream_core.reshape(1, -1)

        w_init = w_init * (1.0 - slack_to_core_cut)

    init_weights.append(w_init)
  return init_weights


# ==============================================================================
# 5. TRANSFER BASELINES (EWC, PEFT, HEAD-ONLY FT, FULL FT) & EVALUATION
# ==============================================================================


def generate_ewc_mask(source_model, target_model):
  """Returns 1.0 for shared backbone parameters (penalized by EWC) and 0.0 for the target head (Appendix H.4)."""
  ewc_mask = []
  for w_src_ref, w_tgt in zip(
      source_model.trainable_weights, target_model.trainable_weights
  ):
    if "output" in w_src_ref.path:
      ewc_mask.append(jnp.zeros(w_tgt.shape, dtype=jnp.float32))
    else:
      ewc_mask.append(jnp.ones(w_tgt.shape, dtype=jnp.float32))
  return ewc_mask


@functools.partial(jax.jit, static_argnames=["apply_fn", "num_classes"])
def _batch_fisher_jit(t_weights, nt_weights, apply_fn, x_b, y_b, num_classes):
  """JIT-compiled per-example diagonal Fisher computation via vmap (Appendix H.4)."""

  def single_example_fisher(x_i, y_i):
    def loss_fn(w):
      logits, _ = apply_fn(
          w, nt_weights, jnp.expand_dims(x_i, 0), training=False
      )
      logits_single = logits[0]
      y_i_squeezed = jnp.squeeze(y_i)

      if num_classes > 1:
        one_hot_y = jax.nn.one_hot(
            y_i_squeezed, num_classes=logits_single.shape[-1]
        )
        return optax.softmax_cross_entropy(logits_single, one_hot_y).mean()
      else:
        return optax.sigmoid_binary_cross_entropy(
            logits_single.squeeze(), y_i_squeezed
        ).mean()

    grads = jax.grad(loss_fn)(t_weights)
    return jax.tree_util.tree_map(lambda g: g**2, grads)

  batch_sq_grads = jax.vmap(single_example_fisher)(x_b, y_b)
  return jax.tree_util.tree_map(lambda g: jnp.mean(g, axis=0), batch_sq_grads)


def compute_diagonal_fim(
    trainable_weights,
    non_trainable_weights,
    apply_fn,
    x_data,
    y_data,
    batch_size=16,
    num_classes=1,
):
  """Computes the exact per-example diagonal Fisher Information Matrix over the source training set (Appendix H.4)."""
  print("  -> Computing FIM for EWC...")
  fim_accumulators = [jnp.zeros_like(w) for w in trainable_weights]
  num_batches = 0

  for i in range(0, len(x_data), batch_size):
    x_b, y_b = x_data[i : i + batch_size], y_data[i : i + batch_size]
    if len(x_b) < batch_size:
      continue

    batch_fim = _batch_fisher_jit(
        trainable_weights,
        non_trainable_weights,
        apply_fn,
        x_b,
        y_b,
        num_classes,
    )
    fim_accumulators = [acc + f for acc, f in zip(fim_accumulators, batch_fim)]
    num_batches += 1

  return [acc / max(num_batches, 1) for acc in fim_accumulators]


@functools.partial(
    jax.jit, static_argnames=["model_apply_fn", "optimizer", "num_classes"]
)
def train_step_ewc_jit(
    trainable_weights,
    non_trainable_weights,
    opt_state,
    model_apply_fn,
    optimizer,
    x_batch,
    y_batch,
    star_weights,
    fim_diags,
    ewc_lambda,
    ewc_mask_jax,
    num_classes,
):
  """Executes one JIT-compiled target optimization step with EWC quadratic Fisher regularization (Appendix H.4)."""

  def loss_fn(t_weights):
    logits, new_nt = model_apply_fn(
        t_weights, non_trainable_weights, x_batch, training=True
    )
    y_squeezed = jnp.squeeze(y_batch)
    if num_classes > 1:
      one_hot_y = jax.nn.one_hot(y_squeezed, num_classes=logits.shape[-1])
      ce_loss = optax.softmax_cross_entropy(logits, one_hot_y).mean()
    else:
      ce_loss = optax.sigmoid_binary_cross_entropy(
          logits.squeeze(), y_squeezed
      ).mean()

    # Apply quadratic Fisher penalty over backbone parameters via ewc_mask_jax
    penalty = 0.0
    for w, w_star, fim, mask in zip(
        t_weights, star_weights, fim_diags, ewc_mask_jax
    ):
      penalty += jnp.sum(mask * fim * (w - w_star) ** 2)

    total_loss = ce_loss + (ewc_lambda / 2.0) * penalty
    l2_penalty = sum([jnp.sum(w**2) for w in t_weights if len(w.shape) > 1])
    return total_loss + (5e-4 * l2_penalty), new_nt

  (loss, new_nt), grads = jax.value_and_grad(loss_fn, has_aux=True)(
      trainable_weights
  )
  updates, new_opt_state = optimizer.update(grads, opt_state, trainable_weights)
  new_trainable_weights = optax.apply_updates(trainable_weights, updates)
  return new_trainable_weights, new_nt, new_opt_state, loss


def generate_peft_mask(source_model, target_model):
  """Returns 1.0 for 1D BatchNorm affine parameters (gamma, beta) and the target head, and 0.0 elsewhere (Appendix H.4)."""
  peft_mask = []
  for w_src_ref, w_tgt in zip(
      source_model.trainable_weights, target_model.trainable_weights
  ):
    if "output" in w_src_ref.path:
      peft_mask.append(jnp.ones(w_tgt.shape, dtype=jnp.float32))
    elif len(w_tgt.shape) == 1:
      peft_mask.append(jnp.ones(w_tgt.shape, dtype=jnp.float32))
    else:
      peft_mask.append(jnp.zeros(w_tgt.shape, dtype=jnp.float32))
  return peft_mask


# ==============================================================================
# 6. HYPERPARAMETER PRE-SWEEPS (APPENDIX H.5)
# ==============================================================================


def find_optimal_deft_hyperparams(
    source_model,
    target_model,
    pristine_weights,
    paired_layers,
    x_train_cifar,
    y_train_cifar,
    x_val_cifar,
    y_val_cifar,
    x_test_cifar,
    y_test_cifar,
    x_train_svhn,
    y_train_svhn,
    x_val_svhn,
    y_val_svhn,
    x_test_svhn,
    y_test_svhn,
    coarse_to_fine,
    pretrain_epochs=100,
    finetune_epochs=FINETUNE_EPOCHS,
    batch_size=BATCH_SIZE,
):
  """Runs the 2D DEFT hyperparameter pre-sweep over (P, eta) (Appendix H.5).

  Draws a single cross-domain tuning task with `seed=999` (disjoint from the 20
  evaluation scenarios generated with `seed in {0, 1, 2, 3}`), trains the source
  expert for `pretrain_epochs` epochs under the cosine-decay schedule of
  Appendix H.3, runs
  `encode_source_curriculum` (`hope.hope_encode`) once, and evaluates the grid:
    P   in {99, 95, 90, 80, 70, 60, 50, 40, 30, 20},
    eta in {0.04, 0.02, 0.01, 0.005, 0.001},
  for `finetune_epochs` fine-tuning epochs on SVHN (capped at
  `HYPERPARAM_PHASE_TRAIN_SAMPLE_CAP = 15,000` training examples). Selects the
  pair (P*, eta*) that maximizes the peak validation H-Score (Eq. 2695):
    H-Score = (2 * Acc_val^target * Acc_val^source) /
              (Acc_val^target + Acc_val^source).
  By Corollary I.1, `Acc_val^source` is fixed at t = 0 by the percentile P
  alone.

  Args:
    source_model: Template 20-class CIFAR-100 Keras model.
    target_model: Template 10-class SVHN Keras model.
    pristine_weights: Initial untrained weights for resetting `source_model`.
    paired_layers: Sequence of `(conv_a, bn_a, conv_b, bn_b)` quadruplets.
    x_train_cifar: Full CIFAR-100 training images.
    y_train_cifar: Full CIFAR-100 training fine labels.
    x_val_cifar: Full CIFAR-100 validation images.
    y_val_cifar: Full CIFAR-100 validation fine labels.
    x_test_cifar: Full CIFAR-100 test images.
    y_test_cifar: Full CIFAR-100 test fine labels.
    x_train_svhn: Full SVHN training images.
    y_train_svhn: Full SVHN training digit labels.
    x_val_svhn: Full SVHN validation images.
    y_val_svhn: Full SVHN validation digit labels.
    x_test_svhn: Full SVHN test images.
    y_test_svhn: Full SVHN test digit labels.
    coarse_to_fine: Mapping from CIFAR-100 superclass ID to fine class IDs.

  Returns:
    Tuple `(best_percentile, best_lr)` locked in for all 20 benchmark scenarios
    in `run_deft_benchmark`.
  """
  print(
      "\n"
      + "=" * 40
      + "\n EXECUTING PRE-SWEEP: DEFT HYPERPARAMETERS\n"
      + "=" * 40
  )
  percentile_grid = [99.0, 95.0, 90.0, 80.0, 70.0, 60.0, 50.0, 40.0, 30.0, 20.0]
  lr_grid = [0.04, 0.02, 0.01, 0.005, 0.001]

  best_params = {}
  cross_tasks = generate_multiclass_domain_tracks(
      coarse_to_fine,
      num_classes=NUM_SOURCE_CLASSES,
      tasks_per_track=1,
      seed=999,
  )
  sample_tasks = {"Track 1 (Cross-Domain)": cross_tasks[0]}

  # Pre-instantiate optimizers per learning rate to reuse compiled JAX
  # functions across runs
  optimizers_cache = {
      lr: optax.sgd(learning_rate=lr, momentum=0.9) for lr in lr_grid
  }
  target_apply_fn = target_model.stateless_call
  source_apply_fn = source_model.stateless_call

  for track_name, (source_pair, _) in sample_tasks.items():
    print(f"\n>>> Tuning {track_name} <<<")

    (x_train_s, y_train_s), (x_val_s, y_val_s), _ = get_multiclass_cifar_task(
        source_pair,
        x_train_cifar,
        y_train_cifar,
        x_val_cifar,
        y_val_cifar,
        x_test_cifar,
        y_test_cifar,
    )

    source_model.set_weights(pristine_weights)

    steps_per_epoch = max(1, len(x_train_s) // batch_size)
    lr_schedule = keras.optimizers.schedules.CosineDecay(
        initial_learning_rate=0.05,
        decay_steps=pretrain_epochs * steps_per_epoch,
        alpha=0.001,
    )

    source_model.compile(
        optimizer=keras.optimizers.SGD(learning_rate=lr_schedule, momentum=0.9),
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"],
    )

    # Train source expert for the full 100-epoch cosine-decay schedule with L2
    # weight decay (Appendix H.3)
    source_model.fit(
        x_train_s,
        y_train_s,
        epochs=pretrain_epochs,
        batch_size=batch_size,
        validation_data=(x_val_s, y_val_s),
        verbose=0,
    )

    source_weights_dict = {
        w.path: w.numpy()
        for w in (
            source_model.trainable_weights + source_model.non_trainable_weights
        )
    }
    orig_source_t_weights = [
        jnp.array(w.numpy()) for w in source_model.trainable_weights
    ]
    orig_source_nt_weights = [
        jnp.array(w.numpy()) for w in source_model.non_trainable_weights
    ]

    print("  -> Executing HOPE Encoder...")
    _, curriculum = encode_source_curriculum(source_weights_dict, paired_layers)

    (x_train_d, y_train_d), (x_val_d, y_val_d), _ = get_10class_svhn_task(
        x_train_svhn,
        y_train_svhn,
        x_val_svhn,
        y_val_svhn,
        x_test_svhn,
        y_test_svhn,
        max_train_samples=HYPERPARAM_PHASE_TRAIN_SAMPLE_CAP,
    )

    best_overall_h_score, best_combo = -1.0, (percentile_grid[2], lr_grid[0])
    is_first_eval = True

    for pct in percentile_grid:
      print(f"  -> Testing Threshold: J_{pct} ...")

      elasticity_dict_source = generate_elasticity_map(
          curriculum,
          source_model,
          percentile_threshold=pct,
          verbose=is_first_eval,
          paired_layers=paired_layers,
      )
      is_first_eval = False

      deft_map_jax = generate_target_elasticity_list(
          elasticity_dict_source, source_model, target_model, paired_layers
      )
      t_weights_init = apply_null_space_initialization(
          orig_source_t_weights,
          elasticity_dict_source,
          source_model,
          target_model,
          paired_layers,
      )
      precomputed_head = precompute_masked_source_head(
          orig_source_t_weights,
          source_model,
          elasticity_dict_source,
          paired_layers,
      )

      for lr in lr_grid:
        t_weights = list(t_weights_init)
        nt_weights = list(orig_source_nt_weights)

        ft_opt = optimizers_cache[lr]
        opt_state = ft_opt.init(t_weights)
        peak_h_this_combo = -1.0

        for epoch in range(finetune_epochs):
          perms = jax.random.permutation(
              jax.random.PRNGKey(epoch), len(x_train_d)
          )
          for batch_idx in range(0, len(x_train_d), batch_size):
            batch_indices = perms[batch_idx : batch_idx + batch_size]
            if len(batch_indices) < batch_size:
              continue
            t_weights, nt_weights, opt_state, _ = train_step_deft_jit(
                t_weights,
                nt_weights,
                opt_state,
                target_apply_fn,
                ft_opt,
                x_train_d[batch_indices],
                y_train_d[batch_indices],
                deft_map_jax,
                num_classes=10,
            )

          ep_val_tgt = evaluate_model(
              t_weights,
              nt_weights,
              target_apply_fn,
              x_val_d,
              y_val_d,
              batch_size * 2,
              num_classes=10,
          )
          t_eval_s, nt_eval_s = get_source_eval_weights_fast(
              t_weights, source_model, precomputed_head, orig_source_nt_weights
          )
          ep_val_src = evaluate_model(
              t_eval_s,
              nt_eval_s,
              source_apply_fn,
              x_val_s,
              y_val_s,
              batch_size * 2,
              num_classes=NUM_SOURCE_CLASSES,
          )

          h_score = (
              (2 * ep_val_tgt * ep_val_src) / (ep_val_tgt + ep_val_src)
              if (ep_val_tgt + ep_val_src) > 0
              else 0.0
          )
          if h_score > peak_h_this_combo:
            peak_h_this_combo = h_score

        print(f"     [LR: {lr}] Peak H-Score: {peak_h_this_combo:.4f}")
        if peak_h_this_combo > best_overall_h_score:
          best_overall_h_score, best_combo = peak_h_this_combo, (pct, lr)

    print(
        f"  => Optimal DEFT Params for {track_name}: J_{best_combo[0]},"
        f" LR={best_combo[1]}"
    )
    best_params[track_name] = best_combo

  return best_params["Track 1 (Cross-Domain)"]


def find_optimal_finetune_lr(
    source_model,
    target_model,
    pristine_weights,
    x_train_cifar,
    y_train_cifar,
    x_val_cifar,
    y_val_cifar,
    x_test_cifar,
    y_test_cifar,
    x_train_svhn,
    y_train_svhn,
    x_val_svhn,
    y_val_svhn,
    x_test_svhn,
    y_test_svhn,
    coarse_to_fine,
    pretrain_epochs=100,
    finetune_epochs=FINETUNE_EPOCHS,
    batch_size=BATCH_SIZE,
):
  """Tunes the learning rate for Standard Full Fine-Tuning on the seed-999 tuning task (Appendix H.5)."""
  print(
      "\n"
      + "=" * 40
      + "\n EXECUTING PRE-SWEEP: FULL FINE-TUNING LR TUNING\n"
      + "=" * 40
  )
  lr_grid = [0.02, 0.01, 0.005, 0.001, 0.0005, 0.0001]
  best_lrs = {}

  cross_tasks = generate_multiclass_domain_tracks(
      coarse_to_fine,
      num_classes=NUM_SOURCE_CLASSES,
      tasks_per_track=1,
      seed=999,
  )
  sample_tasks = {"Track 1 (Cross-Domain)": cross_tasks[0]}

  full_mask_jax = [
      jnp.ones(w.shape, dtype=jnp.float32)
      for w in target_model.trainable_weights
  ]
  optimizers_cache = {
      lr: optax.sgd(learning_rate=lr, momentum=0.9) for lr in lr_grid
  }

  for track_name, (source_pair, _) in sample_tasks.items():
    print(f"\n>>> Tuning Full FT LR for {track_name} <<<")

    (x_train_s, y_train_s), (x_val_s, y_val_s), _ = get_multiclass_cifar_task(
        source_pair,
        x_train_cifar,
        y_train_cifar,
        x_val_cifar,
        y_val_cifar,
        x_test_cifar,
        y_test_cifar,
    )

    # Train source model with cosine-decay SGD (Appendix H.3)
    source_model.set_weights(pristine_weights)
    steps_per_epoch = max(1, len(x_train_s) // batch_size)
    lr_schedule = keras.optimizers.schedules.CosineDecay(
        initial_learning_rate=0.05,
        decay_steps=pretrain_epochs * steps_per_epoch,
        alpha=0.001,
    )
    source_model.compile(
        optimizer=keras.optimizers.SGD(learning_rate=lr_schedule, momentum=0.9),
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"],
    )
    source_model.fit(
        x_train_s,
        y_train_s,
        epochs=pretrain_epochs,
        batch_size=batch_size,
        validation_data=(x_val_s, y_val_s),
        verbose=0,
    )

    orig_source_t_weights = [
        jnp.array(w.numpy()) for w in source_model.trainable_weights
    ]
    orig_source_nt_weights = [
        jnp.array(w.numpy()) for w in source_model.non_trainable_weights
    ]

    (x_train_d, y_train_d), (x_val_d, y_val_d), _ = get_10class_svhn_task(
        x_train_svhn,
        y_train_svhn,
        x_val_svhn,
        y_val_svhn,
        x_test_svhn,
        y_test_svhn,
        max_train_samples=HYPERPARAM_PHASE_TRAIN_SAMPLE_CAP,
    )

    best_overall_h_score, best_lr = -1.0, lr_grid[0]

    for lr in lr_grid:
      print(f"  -> Testing Learning Rate: {lr}")

      # Clone initial weight list for this learning-rate trial
      t_weights = list(
          init_target_task_weights(
              orig_source_t_weights, target_model, source_model
          )
      )
      nt_weights = list(orig_source_nt_weights)

      ft_opt = optimizers_cache[lr]
      opt_state = ft_opt.init(t_weights)
      peak_h_this_lr = -1.0

      for epoch in range(finetune_epochs):
        perms = jax.random.permutation(
            jax.random.PRNGKey(epoch), len(x_train_d)
        )
        for batch_idx in range(0, len(x_train_d), batch_size):
          batch_indices = perms[batch_idx : batch_idx + batch_size]
          if len(batch_indices) < batch_size:
            continue

          t_weights, nt_weights, opt_state, _ = train_step_deft_jit(
              t_weights,
              nt_weights,
              opt_state,
              target_model.stateless_call,
              ft_opt,
              x_train_d[batch_indices],
              y_train_d[batch_indices],
              full_mask_jax,
              num_classes=10,
          )

        ep_val_tgt = evaluate_model(
            t_weights,
            nt_weights,
            target_model.stateless_call,
            x_val_d,
            y_val_d,
            batch_size * 2,
            num_classes=10,
        )
        t_eval_s, nt_eval_s = get_source_eval_weights(
            t_weights,
            orig_source_t_weights,
            orig_source_nt_weights,
            source_model,
        )
        ep_val_src = evaluate_model(
            t_eval_s,
            nt_eval_s,
            source_model.stateless_call,
            x_val_s,
            y_val_s,
            batch_size * 2,
            num_classes=NUM_SOURCE_CLASSES,
        )

        h_score = (
            (2 * ep_val_tgt * ep_val_src) / (ep_val_tgt + ep_val_src)
            if (ep_val_tgt + ep_val_src) > 0
            else 0.0
        )

        if h_score > peak_h_this_lr:
          peak_h_this_lr = h_score

      print(f"     [LR: {lr}] Peak H-Score: {peak_h_this_lr:.4f}")
      if peak_h_this_lr > best_overall_h_score:
        best_overall_h_score, best_lr = peak_h_this_lr, lr

    print(f"  => Optimal Full FT LR for {track_name}: {best_lr}")
    best_lrs[track_name] = best_lr

  return best_lrs["Track 1 (Cross-Domain)"]


def find_optimal_head_only_lr(
    source_model,
    target_model,
    pristine_weights,
    x_train_cifar,
    y_train_cifar,
    x_val_cifar,
    y_val_cifar,
    x_test_cifar,
    y_test_cifar,
    x_train_svhn,
    y_train_svhn,
    x_val_svhn,
    y_val_svhn,
    x_test_svhn,
    y_test_svhn,
    coarse_to_fine,
    pretrain_epochs=100,
    finetune_epochs=FINETUNE_EPOCHS,
    batch_size=BATCH_SIZE,
):
  """Tunes the learning rate for Head-Only Fine-Tuning on the seed-999 tuning task (Appendix H.5)."""
  print(
      "\n"
      + "=" * 40
      + "\n EXECUTING PRE-SWEEP: HEAD-ONLY FINE-TUNING LR TUNING\n"
      + "=" * 40
  )
  lr_grid = [0.04, 0.02, 0.01, 0.005, 0.001, 0.0005]
  best_lrs = {}

  cross_tasks = generate_multiclass_domain_tracks(
      coarse_to_fine,
      num_classes=NUM_SOURCE_CLASSES,
      tasks_per_track=1,
      seed=999,
  )
  sample_tasks = {"Track 1 (Cross-Domain)": cross_tasks[0]}

  head_only_mask_jax = generate_head_only_mask(source_model, target_model)
  optimizers_cache = {
      lr: optax.sgd(learning_rate=lr, momentum=0.9) for lr in lr_grid
  }

  for track_name, (source_pair, _) in sample_tasks.items():
    print(f"\n>>> Tuning Head-Only FT LR for {track_name} <<<")

    (x_train_s, y_train_s), (x_val_s, y_val_s), _ = get_multiclass_cifar_task(
        source_pair,
        x_train_cifar,
        y_train_cifar,
        x_val_cifar,
        y_val_cifar,
        x_test_cifar,
        y_test_cifar,
    )

    # Train source model with cosine-decay SGD (Appendix H.3)
    source_model.set_weights(pristine_weights)
    steps_per_epoch = max(1, len(x_train_s) // batch_size)
    lr_schedule = keras.optimizers.schedules.CosineDecay(
        initial_learning_rate=0.05,
        decay_steps=pretrain_epochs * steps_per_epoch,
        alpha=0.001,
    )
    source_model.compile(
        optimizer=keras.optimizers.SGD(learning_rate=lr_schedule, momentum=0.9),
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"],
    )
    source_model.fit(
        x_train_s,
        y_train_s,
        epochs=pretrain_epochs,
        batch_size=batch_size,
        validation_data=(x_val_s, y_val_s),
        verbose=0,
    )

    orig_source_t_weights = [
        jnp.array(w.numpy()) for w in source_model.trainable_weights
    ]
    orig_source_nt_weights = [
        jnp.array(w.numpy()) for w in source_model.non_trainable_weights
    ]

    (x_train_d, y_train_d), (x_val_d, y_val_d), _ = get_10class_svhn_task(
        x_train_svhn,
        y_train_svhn,
        x_val_svhn,
        y_val_svhn,
        x_test_svhn,
        y_test_svhn,
        max_train_samples=HYPERPARAM_PHASE_TRAIN_SAMPLE_CAP,
    )

    best_overall_h_score, best_lr = -1.0, lr_grid[0]

    for lr in lr_grid:
      print(f"  -> Testing Learning Rate: {lr}")

      # Clone initial weight list for this learning-rate trial
      t_weights = list(
          init_target_task_weights(
              orig_source_t_weights, target_model, source_model
          )
      )
      nt_weights = list(orig_source_nt_weights)

      ft_opt = optimizers_cache[lr]
      opt_state = ft_opt.init(t_weights)
      peak_h_this_lr = -1.0

      for epoch in range(finetune_epochs):
        perms = jax.random.permutation(
            jax.random.PRNGKey(epoch), len(x_train_d)
        )
        for batch_idx in range(0, len(x_train_d), batch_size):
          batch_indices = perms[batch_idx : batch_idx + batch_size]
          if len(batch_indices) < batch_size:
            continue

          t_weights, nt_weights, opt_state, _ = train_step_deft_jit(
              t_weights,
              nt_weights,
              opt_state,
              target_model.stateless_call,
              ft_opt,
              x_train_d[batch_indices],
              y_train_d[batch_indices],
              head_only_mask_jax,
              num_classes=10,
          )

        ep_val_tgt = evaluate_model(
            t_weights,
            nt_weights,
            target_model.stateless_call,
            x_val_d,
            y_val_d,
            batch_size * 2,
            num_classes=10,
        )
        t_eval_s, nt_eval_s = get_source_eval_weights(
            t_weights,
            orig_source_t_weights,
            orig_source_nt_weights,
            source_model,
        )
        ep_val_src = evaluate_model(
            t_eval_s,
            nt_eval_s,
            source_model.stateless_call,
            x_val_s,
            y_val_s,
            batch_size * 2,
            num_classes=NUM_SOURCE_CLASSES,
        )

        h_score = (
            (2 * ep_val_tgt * ep_val_src) / (ep_val_tgt + ep_val_src)
            if (ep_val_tgt + ep_val_src) > 0
            else 0.0
        )

        if h_score > peak_h_this_lr:
          peak_h_this_lr = h_score

      print(f"     [LR: {lr}] Peak H-Score: {peak_h_this_lr:.4f}")
      if peak_h_this_lr > best_overall_h_score:
        best_overall_h_score, best_lr = peak_h_this_lr, lr

    print(f"  => Optimal Head-Only FT LR for {track_name}: {best_lr}")
    best_lrs[track_name] = best_lr

  return best_lrs["Track 1 (Cross-Domain)"]


# Alias for backward compatibility
find_optimal_linear_probe_lr = find_optimal_head_only_lr


def find_optimal_ewc_hyperparams(
    source_model,
    target_model,
    pristine_weights,
    x_train_cifar,
    y_train_cifar,
    x_val_cifar,
    y_val_cifar,
    x_test_cifar,
    y_test_cifar,
    x_train_svhn,
    y_train_svhn,
    x_val_svhn,
    y_val_svhn,
    x_test_svhn,
    y_test_svhn,
    coarse_to_fine,
    pretrain_epochs=100,
    finetune_epochs=FINETUNE_EPOCHS,
    batch_size=BATCH_SIZE,
):
  """Tunes the 2D hyperparameter grid (lambda, eta) for EWC on the seed-999 tuning task (Appendix H.5)."""
  print(
      "\n"
      + "=" * 40
      + "\n EXECUTING PRE-SWEEP: EWC 2D HYPERPARAM TUNING\n"
      + "=" * 40
  )

  lambda_grid = [
      0.1,
      0.5,
      1.0,
      5.0,
      10.0,
      20.0,
      50.0,
      1e2,
      1e3,
      1e4,
      1e5,
      5e5,
      1e6,
      2e6,
      5e6,
      1e7,
      5e7,
      1e8,
  ]
  ewc_lr_grid = [0.01, 0.005, 0.001, 0.0005, 0.0001]
  best_lambdas = {}

  cross_tasks = generate_multiclass_domain_tracks(
      coarse_to_fine,
      num_classes=NUM_SOURCE_CLASSES,
      tasks_per_track=1,
      seed=999,
  )
  sample_tasks = {"Track 1 (Cross-Domain)": cross_tasks[0]}

  for track_name, (source_pair, _) in sample_tasks.items():
    print(f"\n>>> Tuning EWC for {track_name} <<<")

    (x_train_s, y_train_s), (x_val_s, y_val_s), _ = get_multiclass_cifar_task(
        source_pair,
        x_train_cifar,
        y_train_cifar,
        x_val_cifar,
        y_val_cifar,
        x_test_cifar,
        y_test_cifar,
    )

    # Train source model with cosine-decay SGD (Appendix H.3)
    source_model.set_weights(pristine_weights)
    steps_per_epoch = max(1, len(x_train_s) // batch_size)
    lr_schedule = keras.optimizers.schedules.CosineDecay(
        initial_learning_rate=0.05,
        decay_steps=pretrain_epochs * steps_per_epoch,
        alpha=0.001,
    )
    source_model.compile(
        optimizer=keras.optimizers.SGD(learning_rate=lr_schedule, momentum=0.9),
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"],
    )
    source_model.fit(
        x_train_s,
        y_train_s,
        epochs=pretrain_epochs,
        batch_size=batch_size,
        validation_data=(x_val_s, y_val_s),
        verbose=0,
    )

    orig_source_t_weights = [
        jnp.array(w.numpy()) for w in source_model.trainable_weights
    ]
    nt_weights_base = [
        jnp.array(w.numpy()) for w in source_model.non_trainable_weights
    ]

    fim_diags = compute_diagonal_fim(
        orig_source_t_weights,
        nt_weights_base,
        source_model.stateless_call,
        x_train_s,
        y_train_s,
        batch_size,
        num_classes=NUM_SOURCE_CLASSES,
    )

    (x_train_d, y_train_d), (x_val_d, y_val_d), _ = get_10class_svhn_task(
        x_train_svhn,
        y_train_svhn,
        x_val_svhn,
        y_val_svhn,
        x_test_svhn,
        y_test_svhn,
        max_train_samples=HYPERPARAM_PHASE_TRAIN_SAMPLE_CAP,
    )

    best_overall_h_score = -1.0
    best_lambda = lambda_grid[0]
    best_lr = ewc_lr_grid[0]

    ewc_mask_jax = jax.tree_util.tree_map(
        jnp.array, generate_ewc_mask(source_model, target_model)
    )
    fim_diags_aligned = align_ewc_tensors_to_target(
        fim_diags, target_model, source_model
    )
    star_weights_aligned = align_ewc_tensors_to_target(
        orig_source_t_weights, target_model, source_model
    )

    for lr in ewc_lr_grid:
      print(f"\n  --- Testing Fine-Tuning LR: {lr} ---")
      ft_opt = optax.sgd(learning_rate=lr, momentum=0.9)

      for lmbda in lambda_grid:
        t_weights = init_target_task_weights(
            orig_source_t_weights, target_model, source_model
        )
        nt_weights = nt_weights_base
        opt_state = ft_opt.init(t_weights)

        peak_h_this_lambda = -1.0
        diverged = False

        for epoch in range(finetune_epochs):
          perms = jax.random.permutation(
              jax.random.PRNGKey(epoch), len(x_train_d)
          )
          for batch_idx in range(0, len(x_train_d), batch_size):
            batch_indices = perms[batch_idx : batch_idx + batch_size]
            if len(batch_indices) < batch_size:
              continue

            t_weights, nt_weights, opt_state, step_loss = train_step_ewc_jit(
                t_weights,
                nt_weights,
                opt_state,
                target_model.stateless_call,
                ft_opt,
                x_train_d[batch_indices],
                y_train_d[batch_indices],
                star_weights_aligned,
                fim_diags_aligned,
                jnp.array(lmbda, dtype=jnp.float32),
                ewc_mask_jax,
                num_classes=10,
            )
            if not bool(jnp.isfinite(step_loss)):
              diverged = True
              break

          if diverged:
            break

          ep_val_tgt = float(
              evaluate_model(
                  t_weights,
                  nt_weights,
                  target_model.stateless_call,
                  x_val_d,
                  y_val_d,
                  batch_size * 2,
                  num_classes=10,
              )
          )
          t_eval_s, nt_eval_s = get_source_eval_weights(
              t_weights, orig_source_t_weights, nt_weights_base, source_model
          )
          ep_val_src = float(
              evaluate_model(
                  t_eval_s,
                  nt_eval_s,
                  source_model.stateless_call,
                  x_val_s,
                  y_val_s,
                  batch_size * 2,
                  num_classes=NUM_SOURCE_CLASSES,
              )
          )

          h_score = (
              (2 * ep_val_tgt * ep_val_src) / (ep_val_tgt + ep_val_src)
              if (ep_val_tgt + ep_val_src) > 0
              else 0.0
          )

          if np.isfinite(h_score) and h_score > peak_h_this_lambda:
            peak_h_this_lambda = h_score

        print(
            f"      [LR: {lr} | Lambda: {lmbda}] Peak H-Score:"
            f" {peak_h_this_lambda:.4f}"
        )

        if peak_h_this_lambda > best_overall_h_score:
          best_overall_h_score = peak_h_this_lambda
          best_lambda = lmbda
          best_lr = lr

    print("\n" + "=" * 50)
    print(f"=> OPTIMAL CONFIGURATION FOR {track_name}:")
    print(f"   -> Found LR:     {best_lr}")
    print(f"   -> Found Lambda: {best_lambda}")
    print(f"   -> Maximum Achieved H-Score: {best_overall_h_score:.4f}")
    print("=" * 50 + "\n")

    best_lambdas[track_name] = {"lambda": best_lambda, "lr": best_lr}

  return best_lambdas["Track 1 (Cross-Domain)"]


# Alias for backward compatibility
find_optimal_ewc_lambdas = find_optimal_ewc_hyperparams


def find_optimal_peft_lr(
    source_model,
    target_model,
    pristine_weights,
    x_train_cifar,
    y_train_cifar,
    x_val_cifar,
    y_val_cifar,
    x_test_cifar,
    y_test_cifar,
    x_train_svhn,
    y_train_svhn,
    x_val_svhn,
    y_val_svhn,
    x_test_svhn,
    y_test_svhn,
    coarse_to_fine,
    pretrain_epochs=100,
    finetune_epochs=FINETUNE_EPOCHS,
    batch_size=BATCH_SIZE,
):
  """Tunes the learning rate for PEFT (BatchNorm + Head Tuning) on the seed-999 tuning task (Appendix H.5)."""
  print(
      "\n"
      + "=" * 40
      + "\n EXECUTING PRE-SWEEP: PEFT LEARNING RATE\n"
      + "=" * 40
  )
  lr_grid = [0.01, 0.005, 0.0025, 0.001, 0.0005, 0.0001]
  best_lrs = {}

  cross_tasks = generate_multiclass_domain_tracks(
      coarse_to_fine,
      num_classes=NUM_SOURCE_CLASSES,
      tasks_per_track=1,
      seed=999,
  )
  sample_tasks = {"Track 1 (Cross-Domain)": cross_tasks[0]}

  peft_mask_jax = jax.tree_util.tree_map(
      jnp.array, generate_peft_mask(source_model, target_model)
  )
  optimizers_cache = {
      lr: optax.sgd(learning_rate=lr, momentum=0.9) for lr in lr_grid
  }

  for track_name, (source_pair, _) in sample_tasks.items():
    print(f"\n>>> Tuning PEFT LR for {track_name} <<<")

    (x_train_s, y_train_s), (x_val_s, y_val_s), _ = get_multiclass_cifar_task(
        source_pair,
        x_train_cifar,
        y_train_cifar,
        x_val_cifar,
        y_val_cifar,
        x_test_cifar,
        y_test_cifar,
    )

    # Train source model with cosine-decay SGD (Appendix H.3)
    source_model.set_weights(pristine_weights)
    steps_per_epoch = max(1, len(x_train_s) // batch_size)
    lr_schedule = keras.optimizers.schedules.CosineDecay(
        initial_learning_rate=0.05,
        decay_steps=pretrain_epochs * steps_per_epoch,
        alpha=0.001,
    )
    source_model.compile(
        optimizer=keras.optimizers.SGD(learning_rate=lr_schedule, momentum=0.9),
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"],
    )
    source_model.fit(
        x_train_s,
        y_train_s,
        epochs=pretrain_epochs,
        batch_size=batch_size,
        validation_data=(x_val_s, y_val_s),
        verbose=0,
    )

    orig_source_t_weights = [
        jnp.array(w.numpy()) for w in source_model.trainable_weights
    ]
    orig_source_nt_weights = [
        jnp.array(w.numpy()) for w in source_model.non_trainable_weights
    ]

    (x_train_d, y_train_d), (x_val_d, y_val_d), _ = get_10class_svhn_task(
        x_train_svhn,
        y_train_svhn,
        x_val_svhn,
        y_val_svhn,
        x_test_svhn,
        y_test_svhn,
        max_train_samples=HYPERPARAM_PHASE_TRAIN_SAMPLE_CAP,
    )

    best_overall_h_score, best_lr = -1.0, lr_grid[0]

    for lr in lr_grid:
      print(f"  -> Testing Learning Rate: {lr}")
      t_weights = init_target_task_weights(
          orig_source_t_weights, target_model, source_model
      )
      nt_weights = orig_source_nt_weights

      ft_opt = optimizers_cache[lr]
      opt_state = ft_opt.init(t_weights)
      peak_h_this_lr = -1.0

      for epoch in range(finetune_epochs):
        perms = jax.random.permutation(
            jax.random.PRNGKey(epoch), len(x_train_d)
        )
        for batch_idx in range(0, len(x_train_d), batch_size):
          batch_indices = perms[batch_idx : batch_idx + batch_size]
          if len(batch_indices) < batch_size:
            continue

          t_weights, nt_weights, opt_state, _ = train_step_deft_jit(
              t_weights,
              nt_weights,
              opt_state,
              target_model.stateless_call,
              ft_opt,
              x_train_d[batch_indices],
              y_train_d[batch_indices],
              peft_mask_jax,
              num_classes=10,
          )

        ep_val_tgt = evaluate_model(
            t_weights,
            nt_weights,
            target_model.stateless_call,
            x_val_d,
            y_val_d,
            batch_size * 2,
            num_classes=10,
        )
        t_eval_s, nt_eval_s = get_source_eval_weights(
            t_weights,
            orig_source_t_weights,
            orig_source_nt_weights,
            source_model,
        )
        ep_val_src = evaluate_model(
            t_eval_s,
            nt_eval_s,
            source_model.stateless_call,
            x_val_s,
            y_val_s,
            batch_size * 2,
            num_classes=NUM_SOURCE_CLASSES,
        )

        h_score = (
            (2 * ep_val_tgt * ep_val_src) / (ep_val_tgt + ep_val_src)
            if (ep_val_tgt + ep_val_src) > 0
            else 0.0
        )
        if h_score > peak_h_this_lr:
          peak_h_this_lr = h_score

      print(f"     Peak H-Score: {peak_h_this_lr:.4f}")
      if peak_h_this_lr > best_overall_h_score:
        best_overall_h_score, best_lr = peak_h_this_lr, lr

    print(f"  => Optimal LR for {track_name}: {best_lr}")
    best_lrs[track_name] = best_lr
  return best_lrs["Track 1 (Cross-Domain)"]


@functools.partial(jax.jit, static_argnames=["model_apply_fn", "num_classes"])
def evaluate_step(
    trainable_weights,
    non_trainable_weights,
    model_apply_fn,
    x_batch,
    y_batch,
    num_classes,
):
  """Counts correct top-1 predictions on a single evaluation batch."""
  logits, _ = model_apply_fn(
      trainable_weights, non_trainable_weights, x_batch, training=False
  )
  y_squeezed = jnp.squeeze(y_batch)
  if num_classes > 1:
    return (logits.argmax(axis=-1) == y_squeezed).sum()
  else:
    return ((logits.squeeze() > 0) == y_squeezed).sum()


def evaluate_model(
    trainable_weights,
    non_trainable_weights=None,
    model_apply_fn=None,
    x_test=None,
    y_test=None,
    batch_size=16,
    num_classes=1,
):
  """Evaluates top-1 accuracy on a dataset, supporting both explicit JAX weight lists and (keras.Model, dataset) telemetry calls."""
  if hasattr(trainable_weights, "stateless_call") and isinstance(
      non_trainable_weights, tuple
  ):
    expert_model = trainable_weights
    ds_test = non_trainable_weights
    x_t, y_t = ds_test
    t_w = [w.numpy() for w in expert_model.trainable_weights]
    nt_w = [w.numpy() for w in expert_model.non_trainable_weights]
    inferred_num_classes = t_w[-1].shape[-1]
    return evaluate_model(
        t_w,
        nt_w,
        expert_model.stateless_call,
        x_t,
        y_t,
        32,
        inferred_num_classes,
    )

  trainable_jax = jax.tree_util.tree_map(jnp.asarray, trainable_weights)
  non_trainable_jax = jax.tree_util.tree_map(jnp.asarray, non_trainable_weights)

  if np.asarray(y_test).shape[0] == 0:
    return 0.0

  num_correct = 0
  for i in range(0, len(x_test), batch_size):
    x_batch = x_test[i : i + batch_size]
    y_batch = y_test[i : i + batch_size]
    num_correct += evaluate_step(
        trainable_jax,
        non_trainable_jax,
        model_apply_fn,
        x_batch,
        y_batch,
        num_classes,
    )

  return num_correct / len(y_test)


# ==============================================================================
# 7. DEFT ELASTICITY MAP, GRADIENT MASKING, & BENCHMARK RUNNER
# ==============================================================================


def _unpack_curriculum_entry(
    packet: Union[hope.InstructionPacket, Mapping[str, Any]],
) -> Tuple[str, str, Tuple[int, ...], float, float, int]:
  """Extracts `(action, layer_name, neuron_indices, norm, e_b, n_active)` from an `InstructionPacket` or dict."""
  if isinstance(packet, hope.InstructionPacket):
    action = packet.action
    layer_name = packet.quadruplet[0]
    if action == "PRUNE":
      indices = (int(packet.indices[1]),)
    elif action == "MERGE":
      indices = (int(packet.indices[0]), int(packet.indices[1]))
    else:
      indices = ()
    return (
        action,
        layer_name,
        indices,
        float(packet.norm),
        float(packet.e_b),
        int(packet.n_active),
    )
  return (
      str(packet["action"]),
      str(packet["layer_name"]),
      tuple(int(idx) for idx in packet["indices"]),
      float(packet["norm"]),
      float(packet["E_b"]),
      int(packet["N_active"]),
  )


def generate_elasticity_map(
    curriculum: Sequence[Union[hope.InstructionPacket, Mapping[str, Any]]],
    model,
    percentile_threshold: float = 90.0,
    verbose: bool = False,
    paired_layers: Optional[Sequence[hope.Quadruplet]] = None,
):
  """Computes the binary neuron elasticity map E_i in {0, 1} from HOPE's encoding curriculum (Algorithm 3, Phase 1).

  Partitions each hidden backbone layer into a frozen Universal Core F_core = {i
  : E_i = 0} and a plastic Slack
  F_slack = {i : E_i = 1} according to Section 10.2.2 (Eq. deft_elasticity) and
  Definition I.1:
    1. Merge-pair core placement (Section 10.2.2 & Appendix I.3): All neurons M
    that ever participate in a MERGE
       action (both the surviving merge root i and the absorbed child j) are
       placed in the frozen core (E_i = E_j = 0)
       and excluded from the threshold pool C, because DEFT fine-tunes the
       unmerged source weights W_source.
    2. Extinction-guarded pruning-cost pool (Eq. 1189): Over unmerged PRUNE
    steps (i not in M) with remaining
       mean-centred layer capacity E_b^(i) > eps (eps = 1e-5), collect the
       mean-centred pruning costs
         C = { J_prune^(i) = N^(i) ||f_i||_o / E_b^(i) | i not in M, E_b^(i) >
         1e-5 }.
    3. Global percentile thresholding (Eq. deft_elasticity): With J_P =
    Percentile(C, P) (0.0 if C is empty),
         E_i = 1   if i not in M, E_b^(i) > 1e-5, and J_prune^(i) < J_P,
         E_i = 0   otherwise.
       The classification head ('output') is assigned elasticity 1.

  Args:
    curriculum: Sequence of `hope.InstructionPacket` objects (or legacy action
      dicts) recorded by `hope.hope_encode` / `encode_source_curriculum` in
      decoding order.
    model: Source Keras model whose `trainable_weights` paths define the output
      dictionary keys.
    percentile_threshold: Target percentile P in (0, 100] used to compute J_P =
      Percentile(C, P).
    verbose: If True, prints the computed global locking threshold J_P.
    paired_layers: Optional sequence of `(conv_a, bn_a, conv_b, bn_b)` layer
      quadruplets; defaults to `PAIRED_LAYERS` when `None`.

  Returns:
    Dict mapping each trainable weight path (`w.path`) in `model` to a 1D
    float32 NumPy array of per-channel
    elasticities E_i in {0.0, 1.0}.
  """
  elasticity_map = {}

  for w in model.trainable_weights:
    c_out = w.shape[-1]
    if "output" in w.path:
      elasticity_map[w.path] = np.ones(c_out, dtype=np.float32)
    else:
      elasticity_map[w.path] = np.zeros(c_out, dtype=np.float32)

  active_pairs = PAIRED_LAYERS if paired_layers is None else paired_layers
  conv_to_bn = {
      t[0]: t[1] for t in active_pairs if len(t) >= 2 and t[1] is not None
  }

  entries = [_unpack_curriculum_entry(p) for p in curriculum]

  # Phase 1a (Section 10.2.2 & Appendix I.3): Identify all merge participants M
  # per layer. Both the surviving root i and absorbed neuron j of every MERGE
  # step belong to M subseteq F_core.
  merged_neurons_by_layer = {}
  for action, layer, indices, _, _, _ in entries:
    if action == "MERGE":
      if layer not in merged_neurons_by_layer:
        merged_neurons_by_layer[layer] = set()
      merged_neurons_by_layer[layer].update(indices)

  # Phase 1b (Eq. 1189): Collect pruning-cost pool
  # C = { J_prune^(i) | i not in M, E_b^(i) > eps } (eps = 1e-5), excluding
  # extinction artifacts where the last neuron of a layer has vanishing
  # remaining capacity E_b^(i).
  prune_cost_pool = [
      (n_active * norm) / e_b
      for action, layer, indices, norm, e_b, n_active in entries
      if action == "PRUNE"
      and indices[0] not in merged_neurons_by_layer.get(layer, set())
      and e_b > 1e-5
  ]
  prune_cost_pool_arr = np.array(prune_cost_pool)

  # Phase 1c (Eq. deft_elasticity, Algorithm 3 line 2): Global locking
  # threshold J_P = Percentile(C, P).
  if prune_cost_pool_arr.size > 0:
    j_p = float(np.percentile(prune_cost_pool_arr, percentile_threshold))
  else:
    j_p = 0.0

  if verbose:
    print(
        "\n[Diagnostics] Global Locking Threshold (J_P) computed at"
        f" {percentile_threshold}th percentile: {j_p:.4e}"
    )

  layer_packets = {}
  for entry in reversed(entries):
    layer = entry[1]
    if layer not in layer_packets:
      layer_packets[layer] = []
    layer_packets[layer].append(entry)

  for layer_name, packets in layer_packets.items():
    bn_name = conv_to_bn.get(layer_name)

    target_layers = (
        (layer_name, bn_name) if bn_name is not None else (layer_name,)
    )
    layer_weight_paths = [
        w.path
        for w in model.trainable_weights
        if w.path.rsplit("/", 1)[0] in target_layers
    ]
    if not layer_weight_paths:
      continue

    c_out = elasticity_map[layer_weight_paths[0]].shape[0]
    layer_elasticity = np.zeros(c_out, dtype=np.float32)
    merged_in_layer = merged_neurons_by_layer.get(layer_name, set())

    # Assign binary elasticity E_i in {0, 1} (Eq. deft_elasticity, Algorithm 3
    # lines 3-9).
    for action, _, indices, norm, e_b, n_active in packets:
      if action == "PRUNE":
        idx = indices[0]
        if idx in merged_in_layer or e_b <= 1e-5:
          layer_elasticity[idx] = 0.0
        else:
          j_prune = (n_active * norm) / e_b
          layer_elasticity[idx] = 1.0 if j_prune < j_p else 0.0
      elif action == "MERGE":
        for m_idx in indices:
          layer_elasticity[m_idx] = 0.0

    for path in layer_weight_paths:
      elasticity_map[path] = layer_elasticity.copy()

  return elasticity_map


@functools.partial(
    jax.jit, static_argnames=["model_apply_fn", "optimizer", "num_classes"]
)
def train_step_deft_jit(
    trainable_weights,
    non_trainable_weights,
    opt_state,
    model_apply_fn,
    optimizer,
    x_batch,
    y_batch,
    elasticity_map_jax,
    num_classes,
):
  """Executes one JIT-compiled target optimization step with pre-optimizer gradient masking (Axiom I.2, Algorithm 3).

  Evaluates the regularized target objective (Appendix H.3):
    L_target(W) = L_CE(W; x_batch, y_batch) + 5e-4 * sum_{rank(W) > 1}
    ||W||_F^2,
  where 1D BatchNorm scale/shift parameters are excluded from L2 regularization,
  and multiplies the full gradient
  element-wise by `elasticity_map_jax` BEFORE passing it to `optimizer.update`
  (Eq. 1245, Axiom I.2, Algorithm 3
  lines 17-19):
    g_mod = E (o) grad_W L_target(W_t).
  Because L2 regularization is inside `L_target` and `optimizer` (SGD with
  momentum 0.9) is initialized from a zero
  state, parameters with E = 0 receive identically zero velocity and zero
  parameter updates at every step (Theorem I.2).
  Also serves Head-Only FT (`generate_head_only_mask`), PEFT
  (`generate_peft_mask`), and Standard Full FT (all-ones mask).
  """

  def loss_fn(t_weights):
    logits, new_nt = model_apply_fn(
        t_weights, non_trainable_weights, x_batch, training=True
    )
    y_squeezed = jnp.squeeze(y_batch)
    if num_classes > 1:
      one_hot_y = jax.nn.one_hot(y_squeezed, num_classes=logits.shape[-1])
      loss = optax.softmax_cross_entropy(logits, one_hot_y).mean()
    else:
      loss = optax.sigmoid_binary_cross_entropy(
          logits.squeeze(), y_squeezed
      ).mean()
    l2_penalty = sum([jnp.sum(w**2) for w in t_weights if len(w.shape) > 1])
    return loss + (5e-4 * l2_penalty), new_nt

  (loss, new_nt), grads = jax.value_and_grad(loss_fn, has_aux=True)(
      trainable_weights
  )
  modulated_grads = jax.tree_util.tree_map(
      lambda g, c: g * c, grads, elasticity_map_jax
  )
  updates, new_opt_state = optimizer.update(
      modulated_grads, opt_state, trainable_weights
  )
  new_trainable_weights = optax.apply_updates(trainable_weights, updates)
  return new_trainable_weights, new_nt, new_opt_state, loss


def init_target_task_weights(source_t_weights_list, target_model, source_model):
  """Transfers the pretrained source backbone weights and injects a freshly initialized target head."""
  t_weights = []
  for w_src, w_tgt, w_src_ref in zip(
      source_t_weights_list,
      target_model.trainable_weights,
      source_model.trainable_weights,
  ):
    if "output" in w_src_ref.path:
      t_weights.append(jnp.array(w_tgt.numpy()))  # Fresh target head
    else:
      t_weights.append(jnp.array(w_src))  # Pretrained source backbone
  return t_weights


def generate_target_elasticity_list(
    elasticity_dict_source, source_model, target_model, paired_layers
):
  """Broadcasts per-neuron elasticities E_out into full-tensor gradient masks for target fine-tuning (Axiom I.2).

  Implements the gradient elasticity projector of Section 10.2.2 (Eq. 1245) and
  Axiom I.2 (Algorithm 3, line 18):
  each trainable parameter tensor of a destination layer (4D Conv2D kernel, 2D
  Dense kernel, or 1D BN gamma/beta)
  is modulated along its output-channel dimension by the downstream neuron's
  binary elasticity E_{out,k} in {0, 1},
  uniformly across all input channels and spatial kernel taps. The freshly
  initialized target classification head
  ('output') receives full plasticity (E = 1).

  Args:
    elasticity_dict_source: Dict mapping source weight paths to 1D per-channel
      elasticity arrays from `generate_elasticity_map`.
    source_model: Pretrained source Keras model providing canonical weight path
      names.
    target_model: Target Keras model whose trainable weight shapes are matched.
    paired_layers: Sequence of `(conv_a, bn_a, conv_b, bn_b)` layer quadruplets.

  Returns:
    List of float32 JAX arrays matching `target_model.trainable_weights` for
    element-wise gradient multiplication in
    `train_step_deft_jit`.
  """
  del paired_layers
  target_elasticity_list = []
  for w_src_ref, w_tgt in zip(
      source_model.trainable_weights, target_model.trainable_weights
  ):
    if "output" in w_src_ref.path:
      target_elasticity_list.append(jnp.ones(w_tgt.shape, dtype=jnp.float32))
      continue

    downstream_elasticity = elasticity_dict_source[w_src_ref.path]
    mask = jnp.ones(w_tgt.shape, dtype=jnp.float32)

    # Broadcast downstream neuron elasticity E_out across input channels and
    # spatial taps (Eq. 1245, Axiom I.2).
    if len(w_tgt.shape) == 4:
      mask = mask * downstream_elasticity.reshape(1, 1, 1, -1)
    elif len(w_tgt.shape) == 2:
      mask = mask * downstream_elasticity.reshape(1, -1)
    else:
      mask = mask * downstream_elasticity

    target_elasticity_list.append(jnp.array(mask))

  return target_elasticity_list


def generate_head_only_mask(source_model, target_model):
  """Returns 1.0 for the target classification head ('output') and 0.0 for all backbone layers (Appendix H.4)."""
  head_mask = []
  for w_src_ref, w_tgt in zip(
      source_model.trainable_weights, target_model.trainable_weights
  ):
    if "output" in w_src_ref.path:
      head_mask.append(jnp.ones(w_tgt.shape, dtype=jnp.float32))
    else:
      head_mask.append(jnp.zeros(w_tgt.shape, dtype=jnp.float32))
  return head_mask


def get_source_eval_weights(
    current_target_t_weights,
    original_source_t_weights_list,
    pristine_nt_weights,
    source_model,
):
  """Restores the original source classification head and source BatchNorm running statistics for baseline evaluation (Appendix H.5)."""
  t_eval = []
  for w_cur, w_src, w_src_ref in zip(
      current_target_t_weights,
      original_source_t_weights_list,
      source_model.trainable_weights,
  ):
    if "output" in w_src_ref.path:
      t_eval.append(w_src)
    else:
      t_eval.append(w_cur)

  return t_eval, pristine_nt_weights


def precompute_masked_source_head(
    original_source_t_weights,
    source_model,
    elasticity_dict_source,
    paired_layers,
):
  """Applies the structural mask to the pretrained source classification head once at t = 0 (Axioms I.1 & I.3).

  Treats the units of the source classification head as core units of the output
  layer L + 1 (Section 10.2.2,
  Appendix H.5, and Corollary I.2), severing all incoming connections from slack
  neurons of the penultimate layer
  (E_{in,j} > 0) while preserving connections from core features (E_{in,j} = 0)
  and keeping the 1D source head bias
  intact. Because the masked source head is invariant throughout target
  training, precomputing it once enables O(1)
  source view assembly in `get_source_eval_weights_fast`.

  Args:
    original_source_t_weights: List of pretrained source trainable weight
      arrays.
    source_model: Pretrained source Keras model.
    elasticity_dict_source: Dict mapping source weight paths to 1D per-channel
      elasticity arrays.
    paired_layers: Sequence of `(conv_a, bn_a, conv_b, bn_b)` layer quadruplets.

  Returns:
    List of JAX arrays `[masked_kernel, bias]` for the source classification
    head.
  """
  upstream_map = {
      quad[2]: quad[0] for quad in paired_layers if quad[2] is not None
  }

  precomputed_head = []
  for w_src, w_src_ref in zip(
      original_source_t_weights, source_model.trainable_weights
  ):
    if "output" in w_src_ref.path:
      w_out_src = jnp.array(w_src)
      # Apply the structural mask M to the source head kernel (1D bias vector
      # is kept unchanged).
      if len(w_out_src.shape) >= 2:
        layer_name = w_src_ref.path.split("/")[0]
        upstream_layer = upstream_map.get(layer_name)
        upstream_path = f"{upstream_layer}/kernel" if upstream_layer else None

        if upstream_path and upstream_path in elasticity_dict_source:
          in_elasticity = elasticity_dict_source[upstream_path]
          core_feature_mask = (in_elasticity == 0.0).astype(jnp.float32)
          if len(w_out_src.shape) == 4:
            w_out_src = w_out_src * core_feature_mask.reshape(1, 1, -1, 1)
          else:
            w_out_src = w_out_src * core_feature_mask.reshape(-1, 1)
      precomputed_head.append(w_out_src)

  return precomputed_head


def get_source_eval_weights_fast(
    current_target_t_weights,
    source_model,
    precomputed_head_list,
    pristine_nt_weights,
):
  """Assembles the source evaluation view `SourceView(theta_t, theta_src, E)` in O(1) without device allocation (Axiom I.3).

  Implements the source evaluation protocol of Appendix H.5, Axiom I.3, and
  Algorithm 3 (line 21): grafts the
  precomputed masked source classification head (`precomputed_head_list`) onto
  the current backbone weights and
  restores the original source BatchNorm running statistics
  (`pristine_nt_weights`). By Theorem I.2 (Dynamic
  Decoupling) and Corollary I.1, evaluating `source_model` with these weights
  produces source logits identical to
  those at epoch 0 across all target fine-tuning steps.

  Args:
    current_target_t_weights: Current target trainable weight list at step t.
    source_model: Pretrained source Keras model.
    precomputed_head_list: Masked source head tensors returned by
      `precompute_masked_source_head`.
    pristine_nt_weights: Original source non-trainable weights (BN moving_mean
      and moving_variance).

  Returns:
    Tuple `(t_eval, pristine_nt_weights)` ready for `evaluate_model` on the
    source dataset.
  """
  t_eval = []
  head_idx = 0
  for w_cur, w_src_ref in zip(
      current_target_t_weights, source_model.trainable_weights
  ):
    if "output" in w_src_ref.path:
      t_eval.append(precomputed_head_list[head_idx])
      head_idx += 1
    else:
      t_eval.append(w_cur)
  return t_eval, pristine_nt_weights


def align_ewc_tensors_to_target(
    source_tensors_list, target_model, source_model
):
  """Aligns source backbone tensors with target head shapes by zero-padding the terminal head entry."""
  aligned_tensors = []
  for src_tensor, w_tgt, w_src_ref in zip(
      source_tensors_list,
      target_model.trainable_weights,
      source_model.trainable_weights,
  ):
    if "output" in w_src_ref.path:
      aligned_tensors.append(jnp.zeros(w_tgt.shape, dtype=jnp.float32))
    else:
      aligned_tensors.append(jnp.array(src_tensor))
  return aligned_tensors


def _positive_int(value: str) -> int:
  """Argparse type validator requiring a positive integer (>= 1)."""
  parsed = int(value)
  if parsed < 1:
    raise argparse.ArgumentTypeError(
        f"Expected a positive integer, got {value}"
    )
  return parsed


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
  """Parses command-line flags for `deft.py`.

  Args:
    argv: Optional sequence of CLI arguments (defaults to `sys.argv[1:]`).

  Returns:
    Parsed `argparse.Namespace` with resolved benchmark or smoke-test settings.
  """
  parser = argparse.ArgumentParser(
      description=(
          "DEFT: Dispersed Elastic Fine-Tuning cross-domain benchmark "
          "(20-class CIFAR-100 -> 10-class SVHN on CNN_8Layer)."
      )
  )
  parser.add_argument(
      "--smoke_test",
      action="store_true",
      help=(
          "Run a fast end-to-end verification pass (~2 minutes on CPU): skips "
          "hyperparameter pre-sweeps and executes 1 trial x 1 scenario with "
          "2 source epochs, 2 fine-tuning epochs, and capped dataset splits."
      ),
  )
  parser.add_argument(
      "--skip_presweep",
      action="store_true",
      help=(
          "Skip the 5 seed-999 hyperparameter pre-sweeps and use the locked "
          "paper hyperparameters (or explicit CLI overrides)."
      ),
  )
  parser.add_argument(
      "--num_trials",
      type=_positive_int,
      default=None,
      help=(
          "Number of independent random-seed trials (default: 4; 1 in"
          " --smoke_test)."
      ),
  )
  parser.add_argument(
      "--tasks_per_trial",
      type=_positive_int,
      default=None,
      help=(
          "Number of cross-domain tasks per trial (default: 5; 1 in"
          " --smoke_test)."
      ),
  )
  parser.add_argument(
      "--pretrain_epochs",
      type=_positive_int,
      default=None,
      help=(
          "Source CIFAR-100 pretraining epochs (default: 100; 2 in"
          " --smoke_test)."
      ),
  )
  parser.add_argument(
      "--finetune_epochs",
      type=_positive_int,
      default=None,
      help=(
          "Target SVHN fine-tuning epochs (default: 30; 2 in --smoke_test)."
      ),
  )
  parser.add_argument(
      "--batch_size",
      type=_positive_int,
      default=BATCH_SIZE,
      help=f"Training mini-batch size (default: {BATCH_SIZE}).",
  )
  parser.add_argument(
      "--max_source_samples",
      type=_positive_int,
      default=None,
      help=(
          "Optional cap on CIFAR-100 source training samples per scenario (512"
          " in --smoke_test)."
      ),
  )
  parser.add_argument(
      "--max_target_samples",
      type=_positive_int,
      default=None,
      help=(
          "Optional cap on SVHN target training samples per scenario (512 in"
          " --smoke_test)."
      ),
  )
  parser.add_argument(
      "--max_eval_samples",
      type=_positive_int,
      default=None,
      help=(
          "Optional cap on validation and test split sizes (256 in"
          " --smoke_test)."
      ),
  )
  parser.add_argument(
      "--max_tfds_examples",
      type=_positive_int,
      default=None,
      help=(
          "Optional cap on raw TFDS train split examples loaded from disk (8192"
          " in --smoke_test)."
      ),
  )
  parser.add_argument(
      "--deft_percentile",
      type=float,
      default=30.0,
      help=(
          "Locked DEFT percentile threshold P* when --skip_presweep is active"
          " (default: 30.0)."
      ),
  )
  parser.add_argument(
      "--deft_lr",
      type=float,
      default=0.005,
      help=(
          "Locked DEFT learning rate when --skip_presweep is active (default:"
          " 0.005)."
      ),
  )
  parser.add_argument(
      "--head_only_lr",
      type=float,
      default=0.005,
      help=(
          "Locked Head-Only FT learning rate when --skip_presweep is active"
          " (default: 0.005)."
      ),
  )
  parser.add_argument(
      "--finetune_lr",
      type=float,
      default=1e-4,
      help=(
          "Locked Standard Full FT learning rate when --skip_presweep is active"
          " (default: 1e-4)."
      ),
  )
  parser.add_argument(
      "--ewc_lambda",
      type=float,
      default=1e7,
      help=(
          "Locked EWC Fisher penalty lambda* when --skip_presweep is active"
          " (default: 1e7)."
      ),
  )
  parser.add_argument(
      "--ewc_lr",
      type=float,
      default=1e-4,
      help=(
          "Locked EWC learning rate when --skip_presweep is active (default:"
          " 1e-4)."
      ),
  )
  parser.add_argument(
      "--peft_lr",
      type=float,
      default=1e-4,
      help=(
          "Locked PEFT (BN-Tuning) learning rate when --skip_presweep is active"
          " (default: 1e-4)."
      ),
  )
  args = parser.parse_args(argv)
  if args.smoke_test:
    args.skip_presweep = True
    if args.num_trials is None:
      args.num_trials = 1
    if args.tasks_per_trial is None:
      args.tasks_per_trial = 1
    if args.pretrain_epochs is None:
      args.pretrain_epochs = 2
    if args.finetune_epochs is None:
      args.finetune_epochs = 2
    if args.max_source_samples is None:
      args.max_source_samples = 512
    if args.max_target_samples is None:
      args.max_target_samples = 512
    if args.max_eval_samples is None:
      args.max_eval_samples = 256
    if args.max_tfds_examples is None:
      args.max_tfds_examples = 8192
  else:
    if args.num_trials is None:
      args.num_trials = NUM_TRIALS
    if args.tasks_per_trial is None:
      args.tasks_per_trial = TASKS_PER_TRACK
    if args.pretrain_epochs is None:
      args.pretrain_epochs = 100
    if args.finetune_epochs is None:
      args.finetune_epochs = FINETUNE_EPOCHS
  return args


def run_deft_benchmark(args: Optional[argparse.Namespace] = None):
  """Executes the full 20-scenario CIFAR-100 -> SVHN cross-domain transfer benchmark (Section 10.2.4 & Appendix H).

  Implements the end-to-end experimental protocol of Appendix H:
    1. Dataset preparation (Appendix H.1): Loads CIFAR-100 and cropped SVHN
    (32x32x3 in [0, 1]), partitioning each
       into an 80% training split, a 20% validation split (`rng =
       np.random.default_rng(42)`), and the standard test
       split.
    2. Hyperparameter pre-sweeps (Appendix H.5): Tunes each method on an
    independent tuning task (`seed=999`,
       15,000 SVHN training sample cap) by maximizing peak validation H-Score
       across 30 epochs:
         - EWC:          (lambda, eta) in {0.1, 0.5, 1, 5, 10, 20, 50, 1e2,
         1e3, 1e4, 1e5, 5e5, 1e6, 2e6, 5e6, 1e7, 5e7, 1e8} x {0.01, 0.005,
         0.001, 0.0005, 0.0001}
         - PEFT:         eta in {0.01, 0.005, 0.0025, 0.001, 0.0005, 0.0001}
         - Full FT:      eta in {0.02, 0.01, 0.005, 0.001, 0.0005, 0.0001}
         - Head-Only FT: eta in {0.04, 0.02, 0.01, 0.005, 0.001, 0.0005}
         - DEFT:         (P, eta) in {99, 95, 90, 80, 70, 60, 50, 40, 30, 20} x
         {0.04, 0.02, 0.01, 0.005, 0.001}
    3. Multi-trial evaluation (4 trials x 5 tasks = 20 scenarios, Appendix H.1 &
    H.3):
       For each scenario, samples 4 random CIFAR-100 superclasses (20 fine
       classes), trains `CNN_8Layer` from scratch
       for 100 epochs (SGD with momentum 0.9, batch size 16, cosine decay from
       eta = 0.05 to alpha*eta = 5e-5, L2
       weight decay 5e-4 on kernels and BN gamma), runs
       `encode_source_curriculum` (`hope.hope_encode`), and
       adapts to the full 10-class SVHN training
       set for 30 epochs under all five methods.
    4. Dual-domain test reporting (Appendix H.5, Table 2): For each method and
    scenario, selects the epoch with peak
       SVHN validation accuracy (including epoch 0 for DEFT, Head-Only FT, and
       Full FT), evaluates SVHN test accuracy
       and CIFAR-100 test retention (with source BN running statistics restored
       per Axiom I.3), and prints the mean
       +/- standard deviation of Target Accuracy, Source Retention, and
       per-scenario H-Score across all 20 scenarios.

  Args:
    args: Optional parsed CLI `argparse.Namespace` from `parse_args()`. When
      `None`, runs the complete 20-scenario paper benchmark with pre-sweeps.
  """
  if args is None:
    args = parse_args([])

  num_trials = args.num_trials
  tasks_per_trial = args.tasks_per_trial
  pretrain_epochs = args.pretrain_epochs
  finetune_epochs = args.finetune_epochs
  batch_size = args.batch_size
  eval_batch_size = batch_size * 2

  print(
      f"[DEFT Hardware Auto-Detect] Backend: {HARDWARE_INFO['backend']} "
      f"({HARDWARE_INFO['num_devices']}x {HARDWARE_INFO['device_kind']})",
      flush=True,
  )
  print(
      f"Configured {len(PAIRED_LAYERS)} prunable layer quadruplets for"
      " CNN_8Layer."
  )
  if args.smoke_test:
    print(
        "[DEFT Smoke Test] Running fast end-to-end verification"
        f" ({num_trials} trial x {tasks_per_trial} task,"
        f" pretrain_epochs={pretrain_epochs},"
        f" finetune_epochs={finetune_epochs}).",
        flush=True,
    )

  # 1. Load source (CIFAR-100) and target (SVHN) datasets
  (
      (x_train_cifar, y_train_cifar),
      (x_val_cifar, y_val_cifar),
      (x_test_cifar, y_test_cifar),
      coarse_to_fine,
  ) = load_and_partition_cifar100(max_tfds_examples=args.max_tfds_examples)
  (
      (x_train_svhn, y_train_svhn),
      (x_val_svhn, y_val_svhn),
      (x_test_svhn, y_test_svhn),
  ) = load_and_partition_svhn(max_tfds_examples=args.max_tfds_examples)

  # 2. Setup the Cross-Domain results arrays
  cross_results = {
      "DEFT_tgt": np.full((num_trials, tasks_per_trial), np.nan),
      "DEFT_src": np.full((num_trials, tasks_per_trial), np.nan),
      "STD_tgt": np.full((num_trials, tasks_per_trial), np.nan),
      "STD_src": np.full((num_trials, tasks_per_trial), np.nan),
      "FULL_tgt": np.full((num_trials, tasks_per_trial), np.nan),
      "FULL_src": np.full((num_trials, tasks_per_trial), np.nan),
      "EWC_tgt": np.full((num_trials, tasks_per_trial), np.nan),
      "EWC_src": np.full((num_trials, tasks_per_trial), np.nan),
      "PEFT_tgt": np.full((num_trials, tasks_per_trial), np.nan),
      "PEFT_src": np.full((num_trials, tasks_per_trial), np.nan),
  }

  tf.keras.backend.clear_session()

  # Instantiate source (20-class CIFAR-100) and target (10-class SVHN) models
  source_model = create_cnn(num_classes=NUM_SOURCE_CLASSES)
  target_model = create_cnn(num_classes=10)

  source_model.compile(
      optimizer=keras.optimizers.SGD(learning_rate=0.01, momentum=0.9),
      loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
      metrics=["accuracy"],
  )
  pristine_initial_weights = source_model.get_weights()

  # 3. Execute hyperparameter pre-sweeps on the independent tuning task
  # (seed=999), unless --skip_presweep or --smoke_test is enabled.
  if args.skip_presweep:
    optimal_deft = (args.deft_percentile, args.deft_lr)
    optimal_head_only_lr = args.head_only_lr
    optimal_finetune_lr = args.finetune_lr
    optimal_lambda = args.ewc_lambda
    optimal_ewc_lr = args.ewc_lr
    optimal_peft_lr = args.peft_lr
    print(
        "\nSkipping hyperparameter pre-sweeps; using locked hyperparameters:"
        f" DEFT=(P={optimal_deft[0]}, lr={optimal_deft[1]}),"
        f" Head-Only lr={optimal_head_only_lr},"
        f" Full FT lr={optimal_finetune_lr},"
        f" EWC=(lambda={optimal_lambda}, lr={optimal_ewc_lr}),"
        f" PEFT lr={optimal_peft_lr}"
    )
  else:
    ewc_configs = find_optimal_ewc_hyperparams(
        source_model,
        target_model,
        pristine_initial_weights,
        x_train_cifar,
        y_train_cifar,
        x_val_cifar,
        y_val_cifar,
        x_test_cifar,
        y_test_cifar,
        x_train_svhn,
        y_train_svhn,
        x_val_svhn,
        y_val_svhn,
        x_test_svhn,
        y_test_svhn,
        coarse_to_fine,
        pretrain_epochs=pretrain_epochs,
        finetune_epochs=finetune_epochs,
        batch_size=batch_size,
    )
    optimal_lambda = ewc_configs["lambda"]
    optimal_ewc_lr = ewc_configs["lr"]

    optimal_peft_lr = find_optimal_peft_lr(
        source_model,
        target_model,
        pristine_initial_weights,
        x_train_cifar,
        y_train_cifar,
        x_val_cifar,
        y_val_cifar,
        x_test_cifar,
        y_test_cifar,
        x_train_svhn,
        y_train_svhn,
        x_val_svhn,
        y_val_svhn,
        x_test_svhn,
        y_test_svhn,
        coarse_to_fine,
        pretrain_epochs=pretrain_epochs,
        finetune_epochs=finetune_epochs,
        batch_size=batch_size,
    )

    optimal_finetune_lr = find_optimal_finetune_lr(
        source_model,
        target_model,
        pristine_initial_weights,
        x_train_cifar,
        y_train_cifar,
        x_val_cifar,
        y_val_cifar,
        x_test_cifar,
        y_test_cifar,
        x_train_svhn,
        y_train_svhn,
        x_val_svhn,
        y_val_svhn,
        x_test_svhn,
        y_test_svhn,
        coarse_to_fine,
        pretrain_epochs=pretrain_epochs,
        finetune_epochs=finetune_epochs,
        batch_size=batch_size,
    )

    optimal_head_only_lr = find_optimal_head_only_lr(
        source_model,
        target_model,
        pristine_initial_weights,
        x_train_cifar,
        y_train_cifar,
        x_val_cifar,
        y_val_cifar,
        x_test_cifar,
        y_test_cifar,
        x_train_svhn,
        y_train_svhn,
        x_val_svhn,
        y_val_svhn,
        x_test_svhn,
        y_test_svhn,
        coarse_to_fine,
        pretrain_epochs=pretrain_epochs,
        finetune_epochs=finetune_epochs,
        batch_size=batch_size,
    )

    optimal_deft = find_optimal_deft_hyperparams(
        source_model,
        target_model,
        pristine_initial_weights,
        PAIRED_LAYERS,
        x_train_cifar,
        y_train_cifar,
        x_val_cifar,
        y_val_cifar,
        x_test_cifar,
        y_test_cifar,
        x_train_svhn,
        y_train_svhn,
        x_val_svhn,
        y_val_svhn,
        x_test_svhn,
        y_test_svhn,
        coarse_to_fine,
        pretrain_epochs=pretrain_epochs,
        finetune_epochs=finetune_epochs,
        batch_size=batch_size,
    )

  # Pre-instantiate optimizers outside the trial loop so JAX reuses compiled
  # step functions
  current_pct, current_deft_lr = optimal_deft

  ft_optimizer_deft = optax.sgd(learning_rate=current_deft_lr, momentum=0.9)
  ft_optimizer_peft = optax.sgd(learning_rate=optimal_peft_lr, momentum=0.9)
  ft_optimizer_ewc = optax.sgd(learning_rate=optimal_ewc_lr, momentum=0.9)

  ft_optimizer_std = optax.sgd(learning_rate=optimal_head_only_lr, momentum=0.9)
  ft_optimizer_full = optax.sgd(learning_rate=optimal_finetune_lr, momentum=0.9)

  for trial in range(num_trials):
    print(f"\n{'='*20}\n STARTING TRIAL {trial+1}/{num_trials}\n{'='*20}")
    cross_tasks = generate_multiclass_domain_tracks(
        coarse_to_fine,
        num_classes=NUM_SOURCE_CLASSES,
        tasks_per_track=tasks_per_trial,
        seed=trial,
    )

    tracks = [(
        "TRACK 1: CROSS-DOMAIN (CIFAR-100 -> SVHN 10-Class)",
        cross_tasks,
        cross_results,
    )]

    for track_idx, (track_name, task_list, result_dict) in enumerate(tracks):
      print(f"\n>>> EXECUTING {track_name} <<<")

      for task_idx, (source_pair, target_pair) in enumerate(task_list):
        print(f"\nTask {task_idx+1}/{tasks_per_trial} | Target {target_pair}")

        # 1. Train Source Expert on 20-class CIFAR-100
        (x_train_s, y_train_s), (x_val_s, y_val_s), (x_test_s, y_test_s) = (
            get_multiclass_cifar_task(
                source_pair,
                x_train_cifar,
                y_train_cifar,
                x_val_cifar,
                y_val_cifar,
                x_test_cifar,
                y_test_cifar,
            )
        )
        if args.max_source_samples is not None:
          x_train_s = x_train_s[: args.max_source_samples]
          y_train_s = y_train_s[: args.max_source_samples]
        if args.max_eval_samples is not None:
          x_val_s = x_val_s[: args.max_eval_samples]
          y_val_s = y_val_s[: args.max_eval_samples]
          x_test_s = x_test_s[: args.max_eval_samples]
          y_test_s = y_test_s[: args.max_eval_samples]

        # Seed RNG per (trial, task) and initialize fresh model weights
        keras.utils.set_random_seed(trial * 1000 + task_idx)
        fresh_source_weights = keras.models.clone_model(
            source_model
        ).get_weights()
        fresh_target_weights = keras.models.clone_model(
            target_model
        ).get_weights()
        source_model.set_weights(fresh_source_weights)
        target_model.set_weights(fresh_target_weights)

        steps_per_epoch = max(1, len(x_train_s) // batch_size)
        lr_schedule = keras.optimizers.schedules.CosineDecay(
            initial_learning_rate=0.05,
            decay_steps=pretrain_epochs * steps_per_epoch,
            alpha=0.001,
        )

        source_model.compile(
            optimizer=keras.optimizers.SGD(
                learning_rate=lr_schedule, momentum=0.9
            ),
            loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=["accuracy"],
        )

        # Train source expert with cosine-decay schedule and L2 weight decay
        # (Appendix H.3; 100 epochs in the full benchmark)
        source_model.fit(
            x_train_s,
            y_train_s,
            epochs=pretrain_epochs,
            batch_size=batch_size,
            validation_data=(x_val_s, y_val_s),
            verbose=0,
        )

        source_weights_dict = {
            w.path: w.numpy()
            for w in (
                source_model.trainable_weights
                + source_model.non_trainable_weights
            )
        }
        orig_source_t_weights = [
            jnp.array(w.numpy()) for w in source_model.trainable_weights
        ]
        nt_weights_base = [
            jnp.array(w.numpy()) for w in source_model.non_trainable_weights
        ]

        source_acc = evaluate_model(
            orig_source_t_weights,
            nt_weights_base,
            source_model.stateless_call,
            x_test_s,
            y_test_s,
            eval_batch_size,
            num_classes=NUM_SOURCE_CLASSES,
        )
        print(
            "  => Source Expert Model Trained | Final Source Test Acc:"
            f" {source_acc:.4f}"
        )

        # 2. Run HOPE progressive encoder
        _, curriculum = encode_source_curriculum(
            source_weights_dict, PAIRED_LAYERS
        )

        # 3. Fine-tune on 10-class SVHN
        (x_train_d, y_train_d), (x_val_d, y_val_d), (x_test_d, y_test_d) = (
            get_10class_svhn_task(
                x_train_svhn,
                y_train_svhn,
                x_val_svhn,
                y_val_svhn,
                x_test_svhn,
                y_test_svhn,
                max_train_samples=args.max_target_samples,
            )
        )
        if args.max_eval_samples is not None:
          x_val_d = x_val_d[: args.max_eval_samples]
          y_val_d = y_val_d[: args.max_eval_samples]
          x_test_d = x_test_d[: args.max_eval_samples]
          y_test_d = y_test_d[: args.max_eval_samples]

        # ==========================================
        # Method A: DEFT Fine-Tuning
        # ==========================================
        print("  -> Training DEFT...")

        elasticity_dict_source = generate_elasticity_map(
            curriculum,
            source_model,
            percentile_threshold=current_pct,
            paired_layers=PAIRED_LAYERS,
        )
        deft_map_jax = generate_target_elasticity_list(
            elasticity_dict_source, source_model, target_model, PAIRED_LAYERS
        )
        t_weights_deft = apply_null_space_initialization(
            orig_source_t_weights,
            elasticity_dict_source,
            source_model,
            target_model,
            PAIRED_LAYERS,
        )
        nt_weights_deft = nt_weights_base

        opt_state_deft = ft_optimizer_deft.init(t_weights_deft)
        precomputed_head = precompute_masked_source_head(
            orig_source_t_weights,
            source_model,
            elasticity_dict_source,
            PAIRED_LAYERS,
        )

        ep0_val = evaluate_model(
            t_weights_deft,
            nt_weights_deft,
            target_model.stateless_call,
            x_val_d,
            y_val_d,
            eval_batch_size,
            num_classes=10,
        )
        ep0_test = evaluate_model(
            t_weights_deft,
            nt_weights_deft,
            target_model.stateless_call,
            x_test_d,
            y_test_d,
            eval_batch_size,
            num_classes=10,
        )
        t_eval_s, nt_eval_s = get_source_eval_weights_fast(
            t_weights_deft, source_model, precomputed_head, nt_weights_base
        )
        ep0_source = evaluate_model(
            t_eval_s,
            nt_eval_s,
            source_model.stateless_call,
            x_test_s,
            y_test_s,
            eval_batch_size,
            num_classes=NUM_SOURCE_CLASSES,
        )

        deft_val_traj, deft_test_traj, deft_source_traj = (
            [float(ep0_val)],
            [float(ep0_test)],
            [float(ep0_source)],
        )

        for epoch in range(finetune_epochs):
          seed = epoch + trial * 10000 + track_idx * 100 + task_idx * 10
          perms = jax.random.permutation(
              jax.random.PRNGKey(seed), len(x_train_d)
          )
          for batch_idx in range(0, len(x_train_d), batch_size):
            batch_indices = perms[batch_idx : batch_idx + batch_size]
            if len(batch_indices) < batch_size:
              continue
            t_weights_deft, nt_weights_deft, opt_state_deft, _ = (
                train_step_deft_jit(
                    t_weights_deft,
                    nt_weights_deft,
                    opt_state_deft,
                    target_model.stateless_call,
                    ft_optimizer_deft,
                    x_train_d[batch_indices],
                    y_train_d[batch_indices],
                    deft_map_jax,
                    num_classes=10,
                )
            )

          ep_val = evaluate_model(
              t_weights_deft,
              nt_weights_deft,
              target_model.stateless_call,
              x_val_d,
              y_val_d,
              eval_batch_size,
              num_classes=10,
          )
          ep_test = evaluate_model(
              t_weights_deft,
              nt_weights_deft,
              target_model.stateless_call,
              x_test_d,
              y_test_d,
              eval_batch_size,
              num_classes=10,
          )
          t_eval_s, nt_eval_s = get_source_eval_weights_fast(
              t_weights_deft, source_model, precomputed_head, nt_weights_base
          )
          ep_source = evaluate_model(
              t_eval_s,
              nt_eval_s,
              source_model.stateless_call,
              x_test_s,
              y_test_s,
              eval_batch_size,
              num_classes=NUM_SOURCE_CLASSES,
          )

          deft_val_traj.append(float(ep_val))
          deft_test_traj.append(float(ep_test))
          deft_source_traj.append(float(ep_source))

        best_ep = int(np.argmax(deft_val_traj))

        result_dict["DEFT_tgt"][trial, task_idx] = deft_test_traj[best_ep]
        result_dict["DEFT_src"][trial, task_idx] = deft_source_traj[best_ep]
        print(
            f"    DEFT Peak @ Ep {best_ep} | Target Acc:"
            f" {deft_test_traj[best_ep]:.4f} | Source Retention:"
            f" {deft_source_traj[best_ep]:.4f}"
        )

        # ==========================================
        # Method B: Head-Only FT
        # ==========================================
        print("  -> Training Head-Only...")
        t_weights_std = init_target_task_weights(
            orig_source_t_weights, target_model, source_model
        )
        nt_weights_std = nt_weights_base

        opt_state_std = ft_optimizer_std.init(t_weights_std)
        head_only_mask_jax = generate_head_only_mask(source_model, target_model)

        ep0_val = evaluate_model(
            t_weights_std,
            nt_weights_std,
            target_model.stateless_call,
            x_val_d,
            y_val_d,
            eval_batch_size,
            num_classes=10,
        )
        ep0_test = evaluate_model(
            t_weights_std,
            nt_weights_std,
            target_model.stateless_call,
            x_test_d,
            y_test_d,
            eval_batch_size,
            num_classes=10,
        )
        t_eval_s, nt_eval_s = get_source_eval_weights(
            t_weights_std, orig_source_t_weights, nt_weights_base, source_model
        )
        ep0_source = evaluate_model(
            t_eval_s,
            nt_eval_s,
            source_model.stateless_call,
            x_test_s,
            y_test_s,
            eval_batch_size,
            num_classes=NUM_SOURCE_CLASSES,
        )

        std_val_traj, std_test_traj, std_source_traj = (
            [float(ep0_val)],
            [float(ep0_test)],
            [float(ep0_source)],
        )

        for epoch in range(finetune_epochs):
          seed = epoch + trial * 10000 + track_idx * 100 + task_idx * 10
          perms = jax.random.permutation(
              jax.random.PRNGKey(seed), len(x_train_d)
          )
          for batch_idx in range(0, len(x_train_d), batch_size):
            batch_indices = perms[batch_idx : batch_idx + batch_size]
            if len(batch_indices) < batch_size:
              continue
            t_weights_std, nt_weights_std, opt_state_std, _ = (
                train_step_deft_jit(
                    t_weights_std,
                    nt_weights_std,
                    opt_state_std,
                    target_model.stateless_call,
                    ft_optimizer_std,
                    x_train_d[batch_indices],
                    y_train_d[batch_indices],
                    head_only_mask_jax,
                    num_classes=10,
                )
            )

          ep_val = evaluate_model(
              t_weights_std,
              nt_weights_std,
              target_model.stateless_call,
              x_val_d,
              y_val_d,
              eval_batch_size,
              num_classes=10,
          )
          ep_test = evaluate_model(
              t_weights_std,
              nt_weights_std,
              target_model.stateless_call,
              x_test_d,
              y_test_d,
              eval_batch_size,
              num_classes=10,
          )
          t_eval_s, nt_eval_s = get_source_eval_weights(
              t_weights_std,
              orig_source_t_weights,
              nt_weights_base,
              source_model,
          )
          ep_source = evaluate_model(
              t_eval_s,
              nt_eval_s,
              source_model.stateless_call,
              x_test_s,
              y_test_s,
              eval_batch_size,
              num_classes=NUM_SOURCE_CLASSES,
          )

          std_val_traj.append(float(ep_val))
          std_test_traj.append(float(ep_test))
          std_source_traj.append(float(ep_source))

        best_ep = int(np.argmax(std_val_traj))
        result_dict["STD_tgt"][trial, task_idx] = std_test_traj[best_ep]
        result_dict["STD_src"][trial, task_idx] = std_source_traj[best_ep]
        print(
            f"    Head-Only Peak @ Ep {best_ep} | Target Acc:"
            f" {std_test_traj[best_ep]:.4f} | Source Retention:"
            f" {std_source_traj[best_ep]:.4f}"
        )

        # ==========================================
        # Method C: Standard Full FT (Entire Network)
        # ==========================================
        print("  -> Training Full FT...")
        t_weights_full = init_target_task_weights(
            orig_source_t_weights, target_model, source_model
        )
        nt_weights_full = nt_weights_base

        opt_state_full = ft_optimizer_full.init(t_weights_full)
        full_mask_jax = [
            jnp.ones(w.shape, dtype=jnp.float32) for w in t_weights_full
        ]

        ep0_val = evaluate_model(
            t_weights_full,
            nt_weights_full,
            target_model.stateless_call,
            x_val_d,
            y_val_d,
            eval_batch_size,
            num_classes=10,
        )
        ep0_test = evaluate_model(
            t_weights_full,
            nt_weights_full,
            target_model.stateless_call,
            x_test_d,
            y_test_d,
            eval_batch_size,
            num_classes=10,
        )
        t_eval_s, nt_eval_s = get_source_eval_weights(
            t_weights_full, orig_source_t_weights, nt_weights_base, source_model
        )
        ep0_source = evaluate_model(
            t_eval_s,
            nt_eval_s,
            source_model.stateless_call,
            x_test_s,
            y_test_s,
            eval_batch_size,
            num_classes=NUM_SOURCE_CLASSES,
        )

        full_val_traj, full_test_traj, full_source_traj = (
            [float(ep0_val)],
            [float(ep0_test)],
            [float(ep0_source)],
        )

        for epoch in range(finetune_epochs):
          seed = epoch + trial * 10000 + track_idx * 100 + task_idx * 10
          perms = jax.random.permutation(
              jax.random.PRNGKey(seed), len(x_train_d)
          )
          for batch_idx in range(0, len(x_train_d), batch_size):
            batch_indices = perms[batch_idx : batch_idx + batch_size]
            if len(batch_indices) < batch_size:
              continue

            t_weights_full, nt_weights_full, opt_state_full, _ = (
                train_step_deft_jit(
                    t_weights_full,
                    nt_weights_full,
                    opt_state_full,
                    target_model.stateless_call,
                    ft_optimizer_full,
                    x_train_d[batch_indices],
                    y_train_d[batch_indices],
                    full_mask_jax,
                    num_classes=10,
                )
            )

          ep_val = evaluate_model(
              t_weights_full,
              nt_weights_full,
              target_model.stateless_call,
              x_val_d,
              y_val_d,
              eval_batch_size,
              num_classes=10,
          )
          ep_test = evaluate_model(
              t_weights_full,
              nt_weights_full,
              target_model.stateless_call,
              x_test_d,
              y_test_d,
              eval_batch_size,
              num_classes=10,
          )
          t_eval_s, nt_eval_s = get_source_eval_weights(
              t_weights_full,
              orig_source_t_weights,
              nt_weights_base,
              source_model,
          )
          ep_source = evaluate_model(
              t_eval_s,
              nt_eval_s,
              source_model.stateless_call,
              x_test_s,
              y_test_s,
              eval_batch_size,
              num_classes=NUM_SOURCE_CLASSES,
          )

          full_val_traj.append(float(ep_val))
          full_test_traj.append(float(ep_test))
          full_source_traj.append(float(ep_source))

        best_ep = int(np.argmax(full_val_traj))
        result_dict["FULL_tgt"][trial, task_idx] = full_test_traj[best_ep]
        result_dict["FULL_src"][trial, task_idx] = full_source_traj[best_ep]
        print(
            f"    Full FT Peak @ Ep {best_ep} | Target Acc:"
            f" {full_test_traj[best_ep]:.4f} | Source Retention:"
            f" {full_source_traj[best_ep]:.4f}"
        )

        # ==========================================
        # Method D: EWC (Elastic Weight Consolidation)
        # ==========================================
        print("  -> Training EWC...")
        current_ewc_lambda = optimal_lambda

        t_weights_ewc = init_target_task_weights(
            orig_source_t_weights, target_model, source_model
        )
        nt_weights_ewc = nt_weights_base

        fim_diags = compute_diagonal_fim(
            orig_source_t_weights,
            nt_weights_base,
            source_model.stateless_call,
            x_train_s,
            y_train_s,
            batch_size,
            num_classes=NUM_SOURCE_CLASSES,
        )

        opt_state_ewc = ft_optimizer_ewc.init(t_weights_ewc)
        ewc_val_traj, ewc_test_traj, ewc_source_traj = [], [], []

        fim_diags_aligned = align_ewc_tensors_to_target(
            fim_diags, target_model, source_model
        )
        star_weights_aligned = align_ewc_tensors_to_target(
            orig_source_t_weights, target_model, source_model
        )
        ewc_mask_jax = jax.tree_util.tree_map(
            jnp.array, generate_ewc_mask(source_model, target_model)
        )

        ewc_diverged = False
        for epoch in range(finetune_epochs):
          seed = epoch + trial * 10000 + track_idx * 100 + task_idx * 10
          perms = jax.random.permutation(
              jax.random.PRNGKey(seed), len(x_train_d)
          )
          for batch_idx in range(0, len(x_train_d), batch_size):
            batch_indices = perms[batch_idx : batch_idx + batch_size]
            if len(batch_indices) < batch_size:
              continue

            t_weights_ewc, nt_weights_ewc, opt_state_ewc, step_loss_ewc = (
                train_step_ewc_jit(
                    t_weights_ewc,
                    nt_weights_ewc,
                    opt_state_ewc,
                    target_model.stateless_call,
                    ft_optimizer_ewc,
                    x_train_d[batch_indices],
                    y_train_d[batch_indices],
                    star_weights_aligned,
                    fim_diags_aligned,
                    jnp.array(current_ewc_lambda, dtype=jnp.float32),
                    ewc_mask_jax,
                    num_classes=10,
                )
            )
            if not bool(jnp.isfinite(step_loss_ewc)):
              ewc_diverged = True
              break

          if ewc_diverged:
            ewc_val_traj.append(0.0)
            ewc_test_traj.append(0.0)
            ewc_source_traj.append(0.0)
            break

          ep_val = evaluate_model(
              t_weights_ewc,
              nt_weights_ewc,
              target_model.stateless_call,
              x_val_d,
              y_val_d,
              eval_batch_size,
              num_classes=10,
          )
          ep_test = evaluate_model(
              t_weights_ewc,
              nt_weights_ewc,
              target_model.stateless_call,
              x_test_d,
              y_test_d,
              eval_batch_size,
              num_classes=10,
          )
          t_eval_s, nt_eval_s = get_source_eval_weights(
              t_weights_ewc,
              orig_source_t_weights,
              nt_weights_base,
              source_model,
          )
          ep_source = evaluate_model(
              t_eval_s,
              nt_eval_s,
              source_model.stateless_call,
              x_test_s,
              y_test_s,
              eval_batch_size,
              num_classes=NUM_SOURCE_CLASSES,
          )

          ewc_val_traj.append(float(ep_val))
          ewc_test_traj.append(float(ep_test))
          ewc_source_traj.append(float(ep_source))

        best_ep = int(np.argmax(ewc_val_traj))
        result_dict["EWC_tgt"][trial, task_idx] = ewc_test_traj[best_ep]
        result_dict["EWC_src"][trial, task_idx] = ewc_source_traj[best_ep]
        print(
            f"    EWC Peak @ Ep {best_ep + 1} | Target Acc:"
            f" {ewc_test_traj[best_ep]:.4f} | Source Retention:"
            f" {ewc_source_traj[best_ep]:.4f}"
        )

        # ==========================================
        # Method E: PEFT (BN & Head Tuning)
        # ==========================================
        print("  -> Training PEFT (BN-Tuning)...")
        peft_mask_jax = jax.tree_util.tree_map(
            jnp.array, generate_peft_mask(source_model, target_model)
        )

        t_weights_peft = init_target_task_weights(
            orig_source_t_weights, target_model, source_model
        )
        nt_weights_peft = nt_weights_base

        opt_state_peft = ft_optimizer_peft.init(t_weights_peft)
        peft_val_traj, peft_test_traj, peft_source_traj = [], [], []

        for epoch in range(finetune_epochs):
          seed = epoch + trial * 10000 + track_idx * 100 + task_idx * 10
          perms = jax.random.permutation(
              jax.random.PRNGKey(seed), len(x_train_d)
          )
          for batch_idx in range(0, len(x_train_d), batch_size):
            batch_indices = perms[batch_idx : batch_idx + batch_size]
            if len(batch_indices) < batch_size:
              continue

            t_weights_peft, nt_weights_peft, opt_state_peft, _ = (
                train_step_deft_jit(
                    t_weights_peft,
                    nt_weights_peft,
                    opt_state_peft,
                    target_model.stateless_call,
                    ft_optimizer_peft,
                    x_train_d[batch_indices],
                    y_train_d[batch_indices],
                    peft_mask_jax,
                    num_classes=10,
                )
            )

          ep_val = evaluate_model(
              t_weights_peft,
              nt_weights_peft,
              target_model.stateless_call,
              x_val_d,
              y_val_d,
              eval_batch_size,
              num_classes=10,
          )
          ep_test = evaluate_model(
              t_weights_peft,
              nt_weights_peft,
              target_model.stateless_call,
              x_test_d,
              y_test_d,
              eval_batch_size,
              num_classes=10,
          )
          t_eval_s, nt_eval_s = get_source_eval_weights(
              t_weights_peft,
              orig_source_t_weights,
              nt_weights_base,
              source_model,
          )
          ep_source = evaluate_model(
              t_eval_s,
              nt_eval_s,
              source_model.stateless_call,
              x_test_s,
              y_test_s,
              eval_batch_size,
              num_classes=NUM_SOURCE_CLASSES,
          )

          peft_val_traj.append(float(ep_val))
          peft_test_traj.append(float(ep_test))
          peft_source_traj.append(float(ep_source))

        best_ep = int(np.argmax(peft_val_traj))
        result_dict["PEFT_tgt"][trial, task_idx] = peft_test_traj[best_ep]
        result_dict["PEFT_src"][trial, task_idx] = peft_source_traj[best_ep]
        print(
            f"    PEFT Peak @ Ep {best_ep + 1} | Target Acc:"
            f" {peft_test_traj[best_ep]:.4f} | Source Retention:"
            f" {peft_source_traj[best_ep]:.4f}"
        )

  # 4. Final Aggregated Summary Table
  print(
      "\n"
      + "#" * 30
      + f" FINAL CROSS-DOMAIN RESULTS ({num_trials} TRIALS) "
      + "#" * 30
  )
  for track_name, res_dict in [
      ("Cross-Domain (CIFAR-100 -> SVHN 10-Class)", cross_results)
  ]:
    print(f"\n{track_name}:")
    print(
        f"  {'Method'.ljust(25)} | {'Target Accuracy'.ljust(20)} |"
        f" {'Source Retention'.ljust(20)} | {'H-Score'.ljust(20)}"
    )
    print("  " + "-" * 94)

    for prefix, display_name in [
        ("DEFT", "DEFT (Ours)"),
        ("EWC", "EWC Baseline"),
        ("PEFT", "PEFT (BN-Tuning)"),
        ("STD", "Head-Only FT"),
        ("FULL", "Standard Full FT"),
    ]:
      tgt_arr = res_dict[f"{prefix}_tgt"]
      src_arr = res_dict[f"{prefix}_src"]
      denom = tgt_arr + src_arr
      h_arr = np.divide(
          2.0 * tgt_arr * src_arr,
          denom,
          out=np.zeros_like(denom, dtype=np.float64),
          where=denom > 0,
      )

      tgt_mean, tgt_std = np.nanmean(tgt_arr), np.nanstd(tgt_arr)
      src_mean, src_std = np.nanmean(src_arr), np.nanstd(src_arr)
      h_mean, h_std = np.nanmean(h_arr), np.nanstd(h_arr)

      print(
          f"  {display_name.ljust(25)} | "
          f"{tgt_mean:.3f} ± {tgt_std:.3f}      | "
          f"{src_mean:.3f} ± {src_std:.3f}      | "
          f"{h_mean:.3f} ± {h_std:.3f}"
      )


if __name__ == "__main__":
  run_deft_benchmark(parse_args())
