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


def test_load_eval_set_truncated_cache_raises_runtime_error():
  """A corrupt cache file must surface as the documented RuntimeError.

  compress_resnet50.load_eval_set promises 'Corrupt evaluation cache ...;
  delete it and rerun' as a RuntimeError (which main() turns into a clean
  exit code 2); a truncated .npy used to leak a raw ValueError from
  np.load instead. No Keras/TensorFlow is needed: the failure happens in
  np.load before any framework code runs.
  """
  import compress_resnet50  # pylint: disable=g-import-not-at-top

  with tempfile.TemporaryDirectory() as tmpdir:
    images_path, labels_path = compress_resnet50._cache_paths(
        tmpdir, "imagenet2012", 4
    )
    with open(images_path, "wb") as f:
      f.write(b"\x93NUMPY garbage")
    np.save(labels_path, np.zeros((4,), dtype=np.int64))
    try:
      compress_resnet50.load_eval_set(
          tmpdir, "imagenet2012", None, 4, "tensorflow"
      )
    except RuntimeError as e:
      assert "Corrupt evaluation cache" in str(e)
    else:
      raise AssertionError("expected RuntimeError for a truncated cache")


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
