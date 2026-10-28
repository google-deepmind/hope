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


def test_load_density_accuracy_file_skips_nonfinite_rows():
  """NaN/Inf rows are not plottable data and must be dropped at load time.

  A single 'nan' accuracy row used to flow into the Pillow fallback and
  crash the whole plot with 'ValueError: cannot convert float NaN to
  integer' (and silently poison matplotlib autoscaling).
  """
  with tempfile.TemporaryDirectory() as tmpdir:
    path = os.path.join(tmpdir, "curves.csv")
    with open(path, "w") as f:
      f.write("density,accuracy\n")
      f.write("1.0,0.76\n")
      f.write("0.8,nan\n")
      f.write("0.6,inf\n")
      f.write("0.5,0.60\n")
    arr = plot_compression_curves.load_density_accuracy_file(path)
    assert arr is not None
    assert arr.shape == (2, 2)
    assert np.isfinite(arr).all()
    assert arr[0].tolist() == [1.0, 0.76]
    assert arr[1].tolist() == [0.5, 0.60]


def test_single_point_curve_is_visible_in_both_renderers():
  """A one-row curve (e.g. an interrupted run's initial point) must render.

  A line with a single vertex and no marker draws nothing, so the curve
  used to appear in the legend but nowhere on the axes - in both the
  matplotlib renderer and the Pillow fallback.
  """
  with tempfile.TemporaryDirectory() as tmpdir:
    with open(os.path.join(tmpdir, "results_HOPE.csv"), "w") as f:
      f.write("density,accuracy\n1.0,0.76\n")

    # Pillow fallback: count curve-coloured pixels inside the axes area
    # (excluding the legend swatch, which always draws a line sample).
    out_pil = os.path.join(tmpdir, "pil.png")
    plot_compression_curves._render_with_pil_fallback(
        [("HOPE", np.array([[1.0, 0.76]]), "blue")], out_pil
    )
    from PIL import Image  # pylint: disable=g-import-not-at-top

    img = np.asarray(Image.open(out_pil).convert("RGB")).astype(int)
    blue = np.abs(img - np.array([0, 0, 255])).sum(axis=2) < 30
    blue[:220, :520] = False  # legend box region
    assert blue.any(), "Pillow fallback drew no marker for a 1-point curve"

    # Matplotlib: a 1-point curve must carry a marker.
    fig, ax = plot_compression_curves.plot_compression_curves(
        data_dir=tmpdir, save_path=os.path.join(tmpdir, "mpl.png")
    )
    if fig is not None:
      assert ax.get_lines(), "expected one plotted line"
      assert ax.get_lines()[0].get_marker() not in (None, "", "None")


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
