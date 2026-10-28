# Copyright 2026 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Licensed under the Apache License, Version 2.0 (the "License");
# http://www.apache.org/licenses/LICENSE-2.0
"""Regression tests for the compress_resnet50 runner fixes.

Each test names the defect it guards:

* Final artifacts (compressed_weights.npz, curriculum.pkl,
  results_<METHOD>.csv, summary.json) were written directly to their final
  paths, so an interrupted save left a valid-looking but incomplete file
  behind - an interrupted np.savez produces an .npz that loads fine yet
  holds only a subset of the tensors. They now go through
  compress_resnet50._atomic_output (partial file + rename).
* main()'s KeyboardInterrupt message named density_accuracy.csv whenever
  that file existed in --output_dir, including --method ALL runs (which
  never write it) and stale leftovers from previous runs.
  _interrupted_note now names only files this run wrote.

The cache tests pin behaviour the audit verified as correct: an
interrupted cache build (.partial files, or labels renamed before the
images commit point) is never mistaken for a complete cache, and a larger
cache is sliced to the requested number of images.

Needs only NumPy/SciPy - no Keras/TensorFlow: the runner imports those
lazily. Run with: python -m pytest test_compress_resnet50_runner.py
"""

import argparse
import logging
import os
import sys
import tempfile
import time

import compress_resnet50 as cr
import numpy as np


# ----------------------------------------------------------------------------------------------------
# _atomic_output
# ----------------------------------------------------------------------------------------------------


def test_atomic_output_npz_round_trip():
  arrays = {"a": np.ones((3, 3), np.float32), "b": np.arange(5)}
  with tempfile.TemporaryDirectory() as d:
    path = os.path.join(d, "compressed_weights.npz")
    with cr._atomic_output(path) as f:
      np.savez(f, **arrays)
    with np.load(path) as z:
      assert sorted(z.keys()) == ["a", "b"]
      assert float(z["a"].sum()) == 9.0
      assert int(z["b"].sum()) == 10


def test_atomic_output_failure_keeps_previous_file():
  with tempfile.TemporaryDirectory() as d:
    path = os.path.join(d, "summary.json")
    with cr._atomic_output(path, "w") as f:
      f.write('{"complete": true}')
    try:
      with cr._atomic_output(path, "w") as f:
        f.write('{"complete": tr')
        raise KeyboardInterrupt
    except KeyboardInterrupt:
      pass
    with open(path) as f:
      assert f.read() == '{"complete": true}'
    assert not [n for n in os.listdir(d) if ".partial" in n]


def test_atomic_output_failure_leaves_no_file():
  with tempfile.TemporaryDirectory() as d:
    path = os.path.join(d, "curriculum.pkl")
    try:
      with cr._atomic_output(path) as f:
        f.write(b"half a pickle")
        raise RuntimeError("simulated failure")
    except RuntimeError:
      pass
    assert not os.path.exists(path)
    assert not os.listdir(d)


# ----------------------------------------------------------------------------------------------------
# _interrupted_note / KeyboardInterrupt message
# ----------------------------------------------------------------------------------------------------


def _args(method="HOPE", output_dir=".", eval_stride=10):
  return argparse.Namespace(
      method=method, output_dir=output_dir, eval_stride=eval_stride
  )


def _make_stale(path, text="step,density,top1_accuracy\n0,1.0,0.7613\n"):
  with open(path, "w") as f:
    f.write(text)
  old = time.time() - 3600
  os.utime(path, (old, old))


def test_interrupted_note_ignores_stale_csv():
  with tempfile.TemporaryDirectory() as d:
    _make_stale(os.path.join(d, cr.CSV_NAME))
    assert cr._interrupted_note(_args(output_dir=d), time.time()) == ""


def test_interrupted_note_names_fresh_csv():
  with tempfile.TemporaryDirectory() as d:
    started = time.time()
    path = os.path.join(d, cr.CSV_NAME)
    with open(path, "w") as f:
      f.write("step,density,top1_accuracy\n0,1.0,0.5\n")
    note = cr._interrupted_note(_args(output_dir=d), started)
    assert path in note


def test_interrupted_note_all_mode_uses_results_files():
  with tempfile.TemporaryDirectory() as d:
    started = time.time()
    # A density_accuracy.csv, even a fresh one, is never an ALL-mode output.
    with open(os.path.join(d, cr.CSV_NAME), "w") as f:
      f.write("step,density,top1_accuracy\n")
    assert cr._interrupted_note(_args("ALL", d), started) == ""
    results = os.path.join(d, "results_HOPE.csv")
    with open(results, "w") as f:
      f.write("density,accuracy\n1.000000,0.500000\n")
    note = cr._interrupted_note(_args("ALL", d), started)
    assert results in note
    # ... and a stale results file does not count.
    _make_stale(results, "density,accuracy\n")
    assert cr._interrupted_note(_args("ALL", d), time.time()) == ""


def test_interrupted_note_eval_disabled():
  with tempfile.TemporaryDirectory() as d:
    path = os.path.join(d, cr.CSV_NAME)
    with open(path, "w") as f:
      f.write("step,density,top1_accuracy\n")
    assert cr._interrupted_note(_args(output_dir=d, eval_stride=0),
                                time.time() - 60) == ""


def test_main_interrupt_message_end_to_end():
  messages = []

  class _Capture(logging.Handler):

    def emit(self, record):
      messages.append(record.getMessage())

  handler = _Capture()
  cr.logger.addHandler(handler)
  original_run = cr.run

  def interrupted(args):
    raise KeyboardInterrupt

  cr.run = interrupted
  try:
    with tempfile.TemporaryDirectory() as d:
      stale = os.path.join(d, cr.CSV_NAME)
      _make_stale(stale)
      rc = cr.main(["--method", "ALL", "--output_dir", d])
      assert rc == 130
      assert messages, "expected an interruption warning"
      assert stale not in messages[-1], messages[-1]
  finally:
    cr.run = original_run
    cr.logger.removeHandler(handler)


# ----------------------------------------------------------------------------------------------------
# Evaluation cache (verified-correct behaviour, pinned)
# ----------------------------------------------------------------------------------------------------


def _write_cache(d, n_images, n_labels=None):
  images_path, labels_path = cr._cache_paths(d, "imagenet2012", n_images)
  rng = np.random.RandomState(0)
  images = rng.standard_normal((n_images, 224, 224, 3)).astype(np.float32)
  with open(images_path, "wb") as f:
    np.save(f, images)
  labels = np.arange(
      n_images if n_labels is None else n_labels, dtype=np.int64
  )
  with open(labels_path, "wb") as f:
    np.save(f, labels)
  return images_path, labels_path


def test_find_cached_eval_set_ignores_partial_files():
  with tempfile.TemporaryDirectory() as d:
    images_path, labels_path = cr._cache_paths(d, "imagenet2012", 10)
    for p in (images_path, labels_path):
      with open(f"{p}.partial123", "wb") as f:
        f.write(b"junk")
    assert cr.find_cached_eval_set(d, "imagenet2012", 10) is None


def test_find_cached_eval_set_ignores_labels_without_images():
  # build_eval_cache renames the labels first; the images rename is the
  # commit point, so labels alone must not count as a cache.
  with tempfile.TemporaryDirectory() as d:
    _, labels_path = cr._cache_paths(d, "imagenet2012", 10)
    with open(labels_path, "wb") as f:
      np.save(f, np.arange(10, dtype=np.int64))
    assert cr.find_cached_eval_set(d, "imagenet2012", 10) is None


def test_load_eval_set_slices_larger_cache():
  with tempfile.TemporaryDirectory() as d:
    _write_cache(d, 12)
    assert cr.find_cached_eval_set(d, "imagenet2012", 13) is None
    images, labels = cr.load_eval_set(d, "imagenet2012", None, 10,
                                      "tensorflow")
    assert images.shape == (10, 224, 224, 3)
    assert images.dtype == np.float32
    assert labels.shape == (10,)
    assert int(labels[-1]) == 9


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
