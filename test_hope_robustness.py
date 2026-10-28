# Copyright 2026 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Licensed under the Apache License, Version 2.0 (the "License");
# http://www.apache.org/licenses/LICENSE-2.0
"""Regression tests for decoder/encoder robustness in `hope`.

Covers three items, each reproduced on the pre-fix code before fixing:

1. `hope_decode_step` on a hand-built EVICT packet whose
   `saved_block_tensors` is None (the dataclass default) raised
   `AttributeError: 'NoneType' object has no attribute 'items'`; it now
   raises a ValueError naming `saved_block_tensors`. Encoder-produced
   packets always carry the saved tensors, so this only affects the
   public decoder API used directly.
2. `hope_decode(num_steps=-1)` raised a raw `itertools.islice`
   ValueError whose message never mentioned `num_steps`; a negative
   `num_steps` now raises a ValueError naming the argument.
3. `consumer_tap_fractions` returned all-ones fractions for consumer
   kernels that are neither 1x1/Dense nor 3x3 (e.g. 5x5) without the
   docstring saying so; the behaviour (an approximation the encoder's
   cost model uses for such consumers) is now documented, and these
   tests pin it together with the documented 1x1 and 3x3 cases.

Needs only NumPy and SciPy:

    python -m pytest test_hope_robustness.py
"""

import sys
import time
import traceback

import hope
import numpy as np


def _assert_raises(exc_type, fn, *args, **kwargs):
  """Asserts `fn(*args, **kwargs)` raises `exc_type`; returns the exception."""
  try:
    fn(*args, **kwargs)
  except exc_type as e:
    return e
  raise AssertionError(f"expected {exc_type.__name__} from {fn}")


def _evict_packet(saved_block_tensors):
  return hope.InstructionPacket(
      action="EVICT",
      indices=("c1", "c2", "c3"),
      quadruplet=("a", "b", "c", "d"),
      fan_in=0,
      saved_block_tensors=saved_block_tensors,
  )


def test_decode_step_evict_without_saved_tensors_raises_clear_error():
  """EVICT + saved_block_tensors=None: clear ValueError, weights untouched."""
  weights = {"x/kernel": np.zeros((2, 2), dtype=np.float32)}
  packet = _evict_packet(None)
  assert packet.saved_block_tensors is None  # the dataclass default
  error = _assert_raises(ValueError, hope.hope_decode_step, weights, packet)
  assert "saved_block_tensors" in str(error)
  assert "EVICT" in str(error)
  # The failed step must not modify the weights in place.
  np.testing.assert_array_equal(
      weights["x/kernel"], np.zeros((2, 2), dtype=np.float32)
  )


def test_decode_step_evict_with_saved_tensors_still_restores():
  """EVICT + a saved-tensors dict: unchanged in-place restore behaviour."""
  weights = {
      "x/kernel": np.zeros((2, 2), dtype=np.float32),
      "y/kernel": np.ones(3, dtype=np.float32),
  }
  saved = {
      "x/kernel": np.full((2, 2), 7.0, dtype=np.float32),
      "y/kernel": np.full(3, -2.0, dtype=np.float32),
  }
  out = hope.hope_decode_step(weights, _evict_packet(saved))
  assert out is weights  # in place
  np.testing.assert_array_equal(weights["x/kernel"], saved["x/kernel"])
  np.testing.assert_array_equal(weights["y/kernel"], saved["y/kernel"])


def test_decode_negative_num_steps_raises_named_error():
  """Any negative num_steps raises a ValueError that names num_steps."""
  weights = {"x/kernel": np.zeros((2, 2), dtype=np.float32)}
  for bad in (-1, -100):
    error = _assert_raises(
        ValueError, hope.hope_decode, weights, [], num_steps=bad
    )
    assert "num_steps" in str(error), str(error)


def test_decode_valid_num_steps_unchanged():
  """None / 0 / over-long num_steps behave exactly as before."""
  weights = {"x/kernel": np.full((2, 2), 3.0, dtype=np.float32)}
  for num_steps in (None, 0, 10):
    decoded = hope.hope_decode(weights, [], num_steps=num_steps)
    assert decoded is not weights  # decode returns a copy
    np.testing.assert_array_equal(decoded["x/kernel"], weights["x/kernel"])
  np.testing.assert_array_equal(
      weights["x/kernel"], np.full((2, 2), 3.0, dtype=np.float32)
  )


def test_consumer_tap_fractions_documented_sizes():
  """The exact documented cases: 1x1 and Dense are ones; 3x3 tapers."""
  conv_b = "conv2_block1_2_conv"  # ResNet stage prefix conv2 -> H = 56.
  ones = hope.consumer_tap_fractions(np.zeros((1, 1, 4, 4)), conv_b)
  np.testing.assert_array_equal(ones, np.ones((1, 1, 1)))
  dense = hope.consumer_tap_fractions(np.zeros((4, 4)), conv_b)
  np.testing.assert_array_equal(dense, np.ones((1, 1, 1)))
  chi = hope.consumer_tap_fractions(np.zeros((3, 3, 4, 4)), conv_b)
  edge = (56.0 - 1.0) / 56.0
  r = np.array([edge, 1.0, edge])
  np.testing.assert_allclose(chi[:, :, 0], np.outer(r, r), rtol=0, atol=1e-12)


def test_consumer_tap_fractions_other_sizes_are_documented_ones():
  """Non-1x1/3x3 kernels return all ones (documented approximation)."""
  conv_b = "conv2_block1_2_conv"
  for h_b, w_b in ((5, 5), (7, 7), (3, 5), (5, 3), (1, 3)):
    chi = hope.consumer_tap_fractions(np.zeros((h_b, w_b, 4, 4)), conv_b)
    assert chi.shape == (h_b, w_b, 1), (h_b, w_b, chi.shape)
    np.testing.assert_array_equal(chi, np.ones((h_b, w_b, 1)))


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
