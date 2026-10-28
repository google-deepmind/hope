# Copyright 2026 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Licensed under the Apache License, Version 2.0 (the "License");
# http://www.apache.org/licenses/LICENSE-2.0
"""HOPE (Hilbert Operator for Progressive Encoding): Accuracy-vs-Density Curve Plotter.

Standalone plotting utility and command-line script for rendering `(density,
accuracy)` trajectories
produced by `hope.py`, `hope_baselines.py`, and `compress_resnet50.py`:

  - `results_HOPE.csv`              (color='blue',    label='HOPE')
  - `results_L1_IN.csv`             (color='red',     label='L1_In')
  - `results_L1_JOINT.csv`          (color='orange',  label='L1_Joint')
  - `results_BN_SCALE.csv`          (color='green',   label='BN_Scale')
  - `results_FPGM.csv`              (color='purple',  label='FPGM')
  - `results_L1_JOINT_BIASCORR.csv` (color='brown',   label='L1_Joint_BiasCorr')
  - `results_WS_COSINE_MERGE.csv`   (color='magenta', label='WS_Cosine_Merge')
  - `results_SYNFLOW.csv`           (color='teal',    label='SynFlow')

Supports both two-column CSVs (`density,accuracy`), three-column CSVs
(`step,density,top1_accuracy`),
and pipe-delimited (`density | accuracy`) files. Missing curve files are logged
as warnings and
skipped gracefully so partial benchmark runs can still be plotted.

Usage
-----
    python plot_compression_curves.py --data_dir . --output
    accuracy_vs_density.png
"""

import argparse
import logging
import os
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

__all__ = [
    "DEFAULT_CURVE_SPECS",
    "load_density_accuracy_file",
    "plot_compression_curves",
]

logger = logging.getLogger(__name__)

# (Legend label, default filename in data_dir, line color).
DEFAULT_CURVE_SPECS: List[Tuple[str, str, str]] = [
    ("HOPE", "results_HOPE.csv", "blue"),
    ("L1_In", "results_L1_IN.csv", "red"),
    ("L1_Joint", "results_L1_JOINT.csv", "orange"),
    ("BN_Scale", "results_BN_SCALE.csv", "green"),
    ("FPGM", "results_FPGM.csv", "purple"),
    ("L1_Joint_BiasCorr", "results_L1_JOINT_BIASCORR.csv", "brown"),
    ("WS_Cosine_Merge", "results_WS_COSINE_MERGE.csv", "magenta"),
    ("SynFlow", "results_SYNFLOW.csv", "teal"),
]

# RGB palette used by the Pillow fallback renderer when matplotlib is
# unavailable.
_FALLBACK_RGB: Dict[str, Tuple[int, int, int]] = {
    "blue": (0, 0, 255),
    "deepskyblue": (0, 170, 255),
    "red": (255, 0, 0),
    "orange": (255, 165, 0),
    "green": (0, 128, 0),
    "purple": (128, 0, 128),
    "brown": (165, 42, 42),
    "magenta": (204, 0, 204),
    "teal": (0, 128, 128),
}


def load_density_accuracy_file(filepath: str) -> Optional[np.ndarray]:
  """Loads a `(density, accuracy)` array of shape `(N, 2)` from `filepath`.

  Supports comma-delimited CSVs (with or without a header row, either 2-column
  `density,accuracy` or 3-column `step,density,top1_accuracy`) as well as
  pipe-delimited `density | accuracy` files.

  Args:
    filepath: Path to the CSV data file.

  Returns:
    A float64 array of shape `(N, 2)` with columns `(density, accuracy)`, or
    `None`
    if the file does not exist or contains no numeric rows.
  """
  if not os.path.isfile(filepath):
    logger.warning("Data file '%s' not found; skipping this curve.", filepath)
    return None

  rows: List[Tuple[float, float]] = []
  with open(filepath, "r") as f_in:
    for raw_line in f_in:
      line = raw_line.strip()
      if not line or line.startswith("#"):
        continue
      delim = "|" if "|" in line else ","
      parts = [p.strip() for p in line.split(delim)]
      try:
        vals = [float(p) for p in parts]
      except ValueError:
        # Skip header rows (e.g. 'density,accuracy' or
        # 'step,density,top1_accuracy').
        continue
      if len(vals) == 2:
        row = (vals[0], vals[1])
      elif len(vals) >= 3:
        row = (vals[1], vals[2])
      else:
        continue
      # Non-finite pairs (e.g. a 'nan' accuracy recorded by a failing
      # evaluator) are not plottable: they poison matplotlib autoscaling
      # and crash the Pillow fallback's pixel conversion, so drop them.
      if np.isfinite(row[0]) and np.isfinite(row[1]):
        rows.append(row)

  if not rows:
    logger.warning(
        "Data file '%s' contains no valid (density, accuracy) rows; skipping.",
        filepath,
    )
    return None

  return np.asarray(rows, dtype=np.float64)


def _render_with_pil_fallback(
    curves: Sequence[Tuple[str, np.ndarray, str]], save_path: str
) -> None:
  """Renders the accuracy-vs-density plot via Pillow when `matplotlib` is not installed."""
  # pylint: disable=g-import-not-at-top
  from PIL import Image
  from PIL import ImageDraw
  from PIL import ImageFont
  # pylint: enable=g-import-not-at-top

  def get_serif_font(size: int) -> Any:
    for font_path in (
        "/usr/share/fonts/truetype/liberation/LiberationSerif-Regular.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ):
      if os.path.exists(font_path):
        return ImageFont.truetype(font_path, size)
    return ImageFont.load_default()

  width, height = 2000, 1200  # 10x6 aspect ratio at 200 DPI
  pad_l, pad_r, pad_t, pad_b = 180, 60, 50, 140
  pw, ph = width - pad_l - pad_r, height - pad_t - pad_b

  img = Image.new("RGB", (width, height), (255, 255, 255))
  draw = ImageDraw.Draw(img)

  font_label = get_serif_font(36)
  font_tick = get_serif_font(28)
  font_legend = get_serif_font(28)

  all_x = np.concatenate([arr[:, 0] for _, arr, _ in curves])
  all_y = np.concatenate([arr[:, 1] for _, arr, _ in curves])
  xmin, xmax = float(np.min(all_x)), float(np.max(all_x))
  ymin, ymax = float(np.min(all_y)), float(np.max(all_y))
  dx = max((xmax - xmin) * 0.04, 0.01)
  dy = max((ymax - ymin) * 0.04, 0.01)
  x0, x1 = xmin - dx, xmax + dx
  y0, y1 = ymin - dy, ymax + dy

  def to_px(x_val: float, y_val: float) -> Tuple[int, int]:
    px = int(round(pad_l + (x_val - x0) / (x1 - x0) * pw))
    py = int(round(pad_t + (1.0 - (y_val - y0) / (y1 - y0)) * ph))
    return px, py

  for xv in np.arange(0.2, 1.05, 0.1):
    if x0 <= xv <= x1:
      px, _ = to_px(float(xv), y0)
      draw.line([(px, pad_t), (px, pad_t + ph)], fill=(210, 210, 210), width=2)
      draw.line(
          [(px, pad_t + ph), (px, pad_t + ph + 10)], fill=(0, 0, 0), width=3
      )
      txt = f"{xv:.1f}"
      bbox = draw.textbbox((0, 0), txt, font=font_tick)
      draw.text(
          (px - (bbox[2] - bbox[0]) // 2, pad_t + ph + 16),
          txt,
          fill=(0, 0, 0),
          font=font_tick,
      )

  for yv in np.arange(0.0, 0.8, 0.1):
    if y0 <= yv <= y1:
      _, py = to_px(x0, float(yv))
      draw.line([(pad_l, py), (pad_l + pw, py)], fill=(210, 210, 210), width=2)
      draw.line([(pad_l - 10, py), (pad_l, py)], fill=(0, 0, 0), width=3)
      txt = f"{yv:.1f}"
      bbox = draw.textbbox((0, 0), txt, font=font_tick)
      draw.text(
          (pad_l - 18 - (bbox[2] - bbox[0]), py - (bbox[3] - bbox[1]) // 2),
          txt,
          fill=(0, 0, 0),
          font=font_tick,
      )

  draw.rectangle(
      [pad_l, pad_t, pad_l + pw, pad_t + ph], outline=(0, 0, 0), width=3
  )

  for _, arr, color in curves:
    pts = [to_px(float(r[0]), float(r[1])) for r in arr]
    if len(pts) >= 2:
      draw.line(pts, fill=_FALLBACK_RGB.get(color, (0, 0, 0)), width=5)
    elif len(pts) == 1:
      # A single-point curve (e.g. only the initial point of an
      # interrupted run) has no line to draw; mark the point instead so
      # the curve is visible on the axes, not just in the legend.
      px, py = pts[0]
      draw.ellipse(
          [px - 7, py - 7, px + 7, py + 7],
          fill=_FALLBACK_RGB.get(color, (0, 0, 0)),
      )

  xlab = "Model Density"
  xb = draw.textbbox((0, 0), xlab, font=font_label)
  draw.text(
      (pad_l + (pw - (xb[2] - xb[0])) // 2, height - 65),
      xlab,
      fill=(0, 0, 0),
      font=font_label,
  )

  ylab = "Test Accuracy"
  yb = draw.textbbox((0, 0), ylab, font=font_label)
  y_img = Image.new(
      "RGBA", (yb[2] - yb[0] + 20, yb[3] - yb[1] + 20), (255, 255, 255, 0)
  )
  ImageDraw.Draw(y_img).text(
      (10, 10), ylab, fill=(0, 0, 0, 255), font=font_label
  )
  y_rot = y_img.rotate(90, expand=True)
  img.paste(y_rot, (35, pad_t + (ph - y_rot.height) // 2), y_rot)

  leg_x, leg_y = pad_l + 30, pad_t + 30
  max_label_w = max(
      (draw.textbbox((0, 0), lbl, font=font_legend)[2] for lbl, _, _ in curves),
      default=150,
  )
  leg_w = max(280, max_label_w + 130)
  leg_h = 25 + 48 * len(curves)
  draw.rounded_rectangle(
      [leg_x, leg_y, leg_x + leg_w, leg_y + leg_h],
      radius=10,
      fill=(255, 255, 224),
      outline=(0, 0, 0),
      width=2,
  )
  for idx, (label, _, color) in enumerate(curves):
    cy = leg_y + 32 + idx * 48
    draw.line(
        [(leg_x + 20, cy), (leg_x + 80, cy)],
        fill=_FALLBACK_RGB.get(color, (0, 0, 0)),
        width=5,
    )
    draw.text((leg_x + 100, cy - 16), label, fill=(0, 0, 0), font=font_legend)

  img.save(save_path, format="PNG")


def plot_compression_curves(
    data_dir: str = ".",
    save_path: str = "accuracy_vs_density.png",
    show: bool = False,
    file_overrides: Optional[Dict[str, str]] = None,
    curve_specs: Optional[Sequence[Tuple[str, str, str]]] = None,
) -> Tuple[Optional[Any], Optional[Any]]:
  """Loads available `(density, accuracy)` CSV files from `data_dir` and plots them.

  Args:
    data_dir: Directory containing `results_*.csv` (or `density_accuracy.csv`
      for HOPE).
    save_path: Output PNG path (default: `"accuracy_vs_density.png"`).
    show: If True, displays the figure interactively via `plt.show()`.
    file_overrides: Optional dictionary mapping curve label (e.g. `"HOPE"`) to a
      custom CSV path.
    curve_specs: Optional sequence of `(label, default_filename, color)`
      specifications; defaults to `DEFAULT_CURVE_SPECS`.

  Returns:
    `(fig, ax)` when rendered with `matplotlib`, or `(None, None)` when rendered
    via Pillow fallback, when neither plotting library is installed, or when no
    valid curve files are found.
  """
  file_overrides = file_overrides or {}
  specs = (
      list(curve_specs)
      if curve_specs is not None
      else list(DEFAULT_CURVE_SPECS)
  )
  loaded_curves: List[Tuple[str, np.ndarray, str]] = []

  for label, default_fname, color in specs:
    filepath = file_overrides.get(label, os.path.join(data_dir, default_fname))
    # If results_HOPE.csv is not present in data_dir, also check
    # density_accuracy.csv (written by compress_resnet50.py).
    if (
        label == "HOPE"
        and label not in file_overrides
        and not os.path.isfile(filepath)
        and os.path.isfile(os.path.join(data_dir, "density_accuracy.csv"))
    ):
      filepath = os.path.join(data_dir, "density_accuracy.csv")

    arr = load_density_accuracy_file(filepath)
    if arr is None:
      continue
    loaded_curves.append((label, arr, color))

  if not loaded_curves:
    logger.warning("No data files were found to plot in '%s'.", data_dir)
    return None, None

  try:
    import matplotlib.pyplot as plt  # pylint: disable=g-import-not-at-top
  except ImportError:
    if save_path:
      try:
        _render_with_pil_fallback(loaded_curves, save_path)
      except ImportError:
        logger.warning(
            "Neither matplotlib nor Pillow is installed; skipping plot '%s'.",
            save_path,
        )
        return None, None
      logger.info(
          "Saved plot with %d curve(s) to '%s'.", len(loaded_curves), save_path
      )
    return None, None

  rc_overrides = {
      "font.size": 14,
      "axes.labelsize": 16,
      "xtick.labelsize": 12,
      "ytick.labelsize": 12,
      "legend.fontsize": 12,
      "axes.linewidth": 1.5,
      "lines.linewidth": 2.5,
      "font.family": "serif",
  }

  with plt.rc_context(rc_overrides):
    fig, ax = plt.subplots(figsize=(10, 6))
    for label, arr, color in loaded_curves:
      # Single-point curves need a marker: a one-vertex line draws
      # nothing, so the curve would appear in the legend but not on the
      # axes.
      ax.plot(
          arr[:, 0],
          arr[:, 1],
          color=color,
          label=label,
          marker="o" if len(arr) == 1 else None,
          markersize=6,
      )

    ax.set_xlabel("Model Density")
    ax.set_ylabel("Test Accuracy")
    ax.grid(True)
    ax.legend(
        frameon=True,
        facecolor="lightyellow",
        edgecolor="black",
        fancybox=True,
        loc="best",
    )
    fig.tight_layout()

    if save_path:
      plt.savefig(save_path, dpi=300, bbox_inches="tight", format="png")
      logger.info(
          "Saved plot with %d curve(s) to '%s'.", len(loaded_curves), save_path
      )
    if show:
      plt.show()
    plt.close(fig)
  return fig, ax


def main(argv: Optional[Sequence[str]] = None) -> int:
  """Command-line entry point for plotting compression curves."""
  parser = argparse.ArgumentParser(
      description=(
          "Plot HOPE and baseline (density, accuracy) curves from CSV files."
      )
  )
  parser.add_argument(
      "--data_dir",
      type=str,
      default=".",
      help=(
          "Directory containing results_HOPE.csv and baseline"
          " results_<METHOD>.csv files (default: .)."
      ),
  )
  parser.add_argument(
      "--output",
      type=str,
      default="accuracy_vs_density.png",
      help="Output PNG filename (default: accuracy_vs_density.png).",
  )
  parser.add_argument(
      "--show",
      action="store_true",
      help="Display the figure interactively via plt.show().",
  )
  args = parser.parse_args(argv)
  logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
  plot_compression_curves(
      data_dir=args.data_dir, save_path=args.output, show=args.show
  )
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
