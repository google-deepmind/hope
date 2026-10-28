# -*- coding: utf-8 -*-
# Copyright 2026 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Licensed under the Apache License, Version 2.0 (the "License");
# http://www.apache.org/licenses/LICENSE-2.0
"""Regression tests for `deft.compute_diagonal_fim` batch handling.

`compute_diagonal_fim` is documented as the *exact* per-example diagonal
Fisher over the source training set. It accumulates `_batch_fisher_jit`
results, which are per-example means over each batch. These tests pin
three properties of that accumulation:

  1. A trailing partial batch counts (it used to be skipped silently).
  2. A dataset smaller than one batch still yields its Fisher (it used
     to return an all-zero Fisher, silently disabling the EWC penalty).
  3. When the dataset size is a multiple of the batch size the result is
     unchanged (the previous unweighted batch average).

The JIT-compiled `_batch_fisher_jit` is replaced with a deterministic
fake whose per-example Fisher value for example ``x`` is ``x`` itself,
so the exact dataset Fisher is simply ``mean(x)`` and needs no model,
no dataset download, and no accelerator. If the real JAX/Keras/TF
backends are not importable, lightweight stub modules are installed
before importing `deft` (only the attributes `deft` touches at import
time are provided); with real backends installed, no stub is used and
only the Fisher kernel is faked.

Run with:  python -m pytest test_deft_fim.py
       or:  python test_deft_fim.py
"""

import os
import sys
import types

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np  # noqa: E402


def _install_missing_backend_stubs():
  """Installs minimal stubs for any backend `deft` imports but is absent."""
  try:
    import jax  # noqa: F401
  except ImportError:
    jax = types.ModuleType("jax")
    jax.numpy = np

    def _jit(fn=None, **_kwargs):
      return (lambda f: f) if fn is None else fn

    jax.jit = _jit
    jax.grad = lambda f: f
    jax.vmap = lambda f: f
    jax.devices = lambda *a, **k: [
        types.SimpleNamespace(platform="cpu", device_kind="CPU")
    ]
    jax.tree_util = types.SimpleNamespace(
        tree_map=lambda f, t: [f(x) for x in t]
    )
    jax.nn = types.SimpleNamespace(
        one_hot=lambda y, num_classes: np.eye(num_classes)[y]
    )
    jax.random = types.SimpleNamespace(
        PRNGKey=lambda s: s, permutation=lambda k, n: np.arange(n)
    )
    sys.modules["jax"] = jax
    sys.modules["jax.numpy"] = np

  try:
    import keras  # noqa: F401
  except ImportError:
    keras = types.ModuleType("keras")
    keras.regularizers = types.SimpleNamespace()
    sys.modules["keras"] = keras

  try:
    import optax  # noqa: F401
  except ImportError:
    sys.modules["optax"] = types.ModuleType("optax")

  try:
    import tensorflow  # noqa: F401
  except ImportError:
    tf = types.ModuleType("tensorflow")
    tf.config = types.SimpleNamespace(
        set_visible_devices=lambda *a, **k: None
    )
    sys.modules["tensorflow"] = tf

  try:
    import tensorflow_datasets  # noqa: F401
  except ImportError:
    sys.modules["tensorflow_datasets"] = types.ModuleType(
        "tensorflow_datasets"
    )


_install_missing_backend_stubs()

import deft  # noqa: E402


def _fake_batch_fisher(
    t_weights, nt_weights, apply_fn, x_b, y_b, num_classes
):
  """Per-example Fisher value = the example value; returns the batch mean."""
  return [np.full_like(w, float(np.mean(x_b))) for w in t_weights]


_WEIGHTS = [np.zeros((1,), dtype=np.float64)]


def _fim(n, batch_size=16):
  x = np.arange(1, n + 1, dtype=np.float64)
  original = deft._batch_fisher_jit
  deft._batch_fisher_jit = _fake_batch_fisher
  try:
    out = deft.compute_diagonal_fim(
        _WEIGHTS, [], None, x, x, batch_size=batch_size
    )
  finally:
    deft._batch_fisher_jit = original
  return float(out[0][0])


def test_trailing_partial_batch_is_counted():
  # 40 = 16 + 16 + 8: the last 8 examples must count.
  assert _fim(40) == np.float64(np.mean(np.arange(1, 41, dtype=np.float64)))


def test_dataset_smaller_than_one_batch_is_not_zero():
  # 10 < 16: previously this returned an all-zero Fisher.
  assert _fim(10) == np.float64(np.mean(np.arange(1, 11, dtype=np.float64)))


def test_full_batches_match_previous_batch_average():
  # 32 = 16 + 16: weighted and unweighted averages agree exactly.
  batch_means = [np.mean(np.arange(1, 17)), np.mean(np.arange(17, 33))]
  assert _fim(32) == np.float64(np.mean(batch_means))


def _run_all() -> int:
  tests = [
      test_trailing_partial_batch_is_counted,
      test_dataset_smaller_than_one_batch_is_not_zero,
      test_full_batches_match_previous_batch_average,
  ]
  for test in tests:
    test()
    print(f"PASS {test.__name__}")
  print(f"{len(tests)} passed")
  return 0


if __name__ == "__main__":
  sys.exit(_run_all())
