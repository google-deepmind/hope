# Copyright 2026 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Licensed under the Apache License, Version 2.0 (the "License");
# http://www.apache.org/licenses/LICENSE-2.0
"""A one-minute tour of HOPE on a tiny synthetic network.

Builds a small ResNet-style weights dictionary (no Keras, no ImageNet, only
NumPy and SciPy), compresses it with ``hope.hope_encode``, and then walks the
resulting curriculum back with ``hope.hope_decode``:

  1. encode      -- print how many PRUNE / MERGE / EVICT steps the encoder
                    chose and how the parameter density falls along the way;
  2. full decode -- undo every step and check that the original weights are
                    restored (lossless up to float32 round-off);
  3. progressive decode -- undo only part of the curriculum to land on an
                    intermediate density, then undo the remaining steps from
                    there and check losslessness again.

Usage:

    python examples/synthetic_compression_demo.py
    python examples/synthetic_compression_demo.py --seed 1 --quiet

Exits 0 when every check passes and 1 otherwise. For the real ImageNet
workflow (Keras ResNet-50, accuracy versus density), see
``compress_resnet50.py`` in the repository root.
"""

import argparse
import collections
from pathlib import Path
import sys
import time
from typing import Dict, List, Optional, Sequence, Tuple

# ``hope`` lives in the repository root, one directory above this file, which
# is not on ``sys.path`` when the demo is run as a script.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
  sys.path.insert(0, str(_REPO_ROOT))

import hope  # noqa: E402  (import after the sys.path bootstrap above)
import numpy as np  # noqa: E402

WeightsDict = Dict[str, np.ndarray]

WIDTH = 16  # Channels entering and leaving each bottleneck block.
MID = 8  # Bottleneck channels: the neurons the encoder prunes and merges.
NUM_BLOCKS = 2  # Bottleneck blocks, named conv2_block1 and conv2_block2.
# Producer kernels are restored as (k * s) / s in float32, so a few ulps of
# round-off are expected there; 1e-5 is a generous bound on the max abs diff.
LOSSLESS_TOLERANCE = 1e-5


def build_synthetic_weights(seed: int = 0) -> WeightsDict:
  """Builds a tiny Keras-style ResNet weights dictionary.

  A compact builder in the spirit of ``make_synthetic_resnet_weights`` in
  ``test_hope.py`` (same tensor naming and shapes, including the projection
  shortcut of the first block), written independently for this demo. As in
  the test fixture, neurons 0 and 1 of the first block are made
  near-duplicates so that the encoder exercises MERGE, not only PRUNE and
  EVICT.

  Args:
    seed: Random seed; the legacy ``RandomState`` stream is frozen across
      NumPy versions, so a given seed always builds the same network.

  Returns:
    Dictionary mapping Keras-style weight paths to float32 arrays.
  """
  rs = np.random.RandomState(seed)
  weights: WeightsDict = {}

  def add_conv(name: str, size: int, channels_in: int, channels_out: int):
    scale = np.sqrt(2.0 / (size * size * channels_in))
    weights[f"{name}/kernel"] = (
        rs.standard_normal((size, size, channels_in, channels_out)) * scale
    ).astype(np.float32)
    weights[f"{name}/bias"] = (
        0.01 * rs.standard_normal(channels_out)
    ).astype(np.float32)

  def add_bn(name: str, channels: int):
    weights[f"{name}/gamma"] = rs.uniform(0.5, 1.5, channels).astype(np.float32)
    weights[f"{name}/beta"] = (0.5 * rs.standard_normal(channels)).astype(
        np.float32
    )
    weights[f"{name}/moving_mean"] = (
        0.5 * rs.standard_normal(channels)
    ).astype(np.float32)
    weights[f"{name}/moving_variance"] = rs.uniform(0.5, 2.0, channels).astype(
        np.float32
    )

  add_conv("conv1_conv", 7, 3, WIDTH)
  add_bn("conv1_bn", WIDTH)
  for block in range(1, NUM_BLOCKS + 1):
    prefix = f"conv2_block{block}"
    if block == 1:
      add_conv(f"{prefix}_0_conv", 1, WIDTH, WIDTH)
      add_bn(f"{prefix}_0_bn", WIDTH)
    add_conv(f"{prefix}_1_conv", 1, WIDTH, MID)
    add_bn(f"{prefix}_1_bn", MID)
    add_conv(f"{prefix}_2_conv", 3, MID, MID)
    add_bn(f"{prefix}_2_bn", MID)
    add_conv(f"{prefix}_3_conv", 1, MID, WIDTH)
    add_bn(f"{prefix}_3_bn", WIDTH)
  weights["predictions/kernel"] = (
      0.1 * rs.standard_normal((WIDTH, 10))
  ).astype(np.float32)
  weights["predictions/bias"] = np.zeros(10, np.float32)

  # Near-duplicate neurons 0 and 1 of the first quadruplet (see docstring).
  first = weights["conv2_block1_1_conv/kernel"]
  second = weights["conv2_block1_2_conv/kernel"]
  first[..., 1] = 1.01 * first[..., 0]
  second[:, :, 1, :] = 0.99 * second[:, :, 0, :]
  for param in ("gamma", "beta", "moving_mean", "moving_variance"):
    weights[f"conv2_block1_1_bn/{param}"][1] = weights[
        f"conv2_block1_1_bn/{param}"
    ][0]
  return weights


def _nonzeros(weights: WeightsDict) -> int:
  return int(sum(np.count_nonzero(v) for v in weights.values()))


def _max_abs_diff(a: WeightsDict, b: WeightsDict) -> float:
  return max(float(np.max(np.abs(a[k] - b[k]))) for k in a)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
  """Parses and validates the command line."""
  parser = argparse.ArgumentParser(
      description=(
          "Compress a tiny synthetic ResNet-style network with HOPE and"
          " verify that decoding its curriculum is lossless."
      ),
      epilog=(
          "examples:\n  python examples/synthetic_compression_demo.py\n "
          " python examples/synthetic_compression_demo.py --seed 1"
      ),
      formatter_class=argparse.RawDescriptionHelpFormatter,
  )
  parser.add_argument(
      "--seed",
      type=int,
      default=0,
      help="seed for the synthetic weights (default: %(default)s)",
  )
  parser.add_argument(
      "--quiet",
      action="store_true",
      help="suppress the summary; print only the checks and the final result",
  )
  args = parser.parse_args(argv)
  if args.seed < 0:
    parser.error("--seed must be >= 0.")
  return args


def run(args: argparse.Namespace) -> int:
  """Runs the demo; returns 0 if every check passes, 1 otherwise."""
  failures: List[str] = []

  def say(message: str = "") -> None:
    if not args.quiet:
      print(message)

  def check(name: str, ok: bool, detail: str) -> None:
    print(f"[{'PASS' if ok else 'FAIL'}] {name}: {detail}")
    if not ok:
      failures.append(name)

  say("HOPE synthetic compression demo")
  say("===============================")
  weights = build_synthetic_weights(args.seed)
  snapshot = {k: v.copy() for k, v in weights.items()}
  initial_nonzeros = _nonzeros(weights)
  quadruplets = hope.find_bottleneck_quadruplets(list(weights))
  say(
      f"Built a synthetic network: {len(weights)} tensors,"
      f" {initial_nonzeros} nonzero parameters, {len(quadruplets)}"
      " bottleneck quadruplets."
  )

  start = time.time()
  compressed, curriculum, history = hope.hope_encode(
      weights, quadruplets, eval_stride=1
  )
  elapsed = time.time() - start
  counts = collections.Counter(packet.action for packet in curriculum)
  say(f"Encoded in {elapsed:.1f}s: {len(curriculum)} curriculum steps")
  for action in ("PRUNE", "MERGE", "EVICT"):
    say(f"  {action:<6} {counts.get(action, 0)}")
  checkpoints = sorted(
      {0, len(history) // 4, len(history) // 2, 3 * len(history) // 4,
       len(history) - 1}
  )
  say("Density trajectory (fraction of parameters remaining):")
  for index in checkpoints:
    say(f"  after step {index + 1:>3}: density {float(history[index][0]):.3f}")
  check(
      "encoder leaves its input untouched",
      all(np.array_equal(weights[k], snapshot[k]) for k in weights),
      "input weights are unchanged by hope_encode",
  )

  # 1. Full decode must restore the original weights.
  restored = hope.hope_decode(compressed, curriculum)
  full_diff = _max_abs_diff(restored, weights)
  check(
      "full decode is lossless",
      full_diff <= LOSSLESS_TOLERANCE,
      f"max abs diff {full_diff:.3e} (tolerance {LOSSLESS_TOLERANCE:.0e})",
  )

  # 2. Progressive decode: undo half of the curriculum to reach an
  # intermediate density, then undo the rest from there.
  half = len(curriculum) // 2
  intermediate = hope.hope_decode(compressed, curriculum, num_steps=half)
  intermediate_density = _nonzeros(intermediate) / initial_nonzeros
  say(
      f"Progressive decode of {half} of {len(curriculum)} steps lands at"
      f" density {intermediate_density:.3f}, between the compressed"
      f" ({_nonzeros(compressed) / initial_nonzeros:.3f}) and the original"
      " (1.000)."
  )
  check(
      "progressive decode reaches an intermediate density",
      _nonzeros(compressed) < _nonzeros(intermediate) < initial_nonzeros,
      f"density {intermediate_density:.3f}",
  )
  resumed = hope.hope_decode(intermediate, curriculum[half:])
  resumed_diff = _max_abs_diff(resumed, weights)
  check(
      "decoding the remaining steps from the intermediate model is lossless",
      resumed_diff <= LOSSLESS_TOLERANCE,
      f"max abs diff {resumed_diff:.3e} (tolerance {LOSSLESS_TOLERANCE:.0e})",
  )

  if failures:
    print(f"RESULT: FAIL ({len(failures)} check(s) failed: {', '.join(failures)})")
    return 1
  print("RESULT: PASS - HOPE encoded, fully decoded, and progressively "
        "decoded the synthetic network losslessly.")
  return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
  return run(parse_args(argv))


if __name__ == "__main__":
  sys.exit(main())
