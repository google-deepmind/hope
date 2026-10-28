# Copyright 2026 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Licensed under the Apache License, Version 2.0 (the "License");
# http://www.apache.org/licenses/LICENSE-2.0
"""HOPE (Hilbert Operator for Progressive Encoding): ResNet-50 compression runner.

Command-line runner: compresses an ImageNet-pretrained Keras ResNet-50 with HOPE
and records the
top-1 accuracy of the compressed network along the whole encoding trajectory.

What the script does
--------------------
1. Loads ResNet-50 with its pretrained ImageNet weights (Keras 3; TensorFlow,
JAX or PyTorch backend).
2. Runs the data-free HOPE encoder (``hope.hope_encode``) until every bottleneck
neuron has been
   consumed. Every ``--eval_stride`` encoding steps the current compressed
   network is evaluated (top-1
   accuracy on the first ``--num_eval_images`` ImageNet validation images) and
   the row
   ``step,density,top1_accuracy`` is appended to ``density_accuracy.csv``
   immediately, so partial results
   survive an interruption. Step 0 is the pretrained network.
3. Saves the compressed weights and the curriculum (the progressive decoding
stream).
4. Verifies that decoding the full curriculum (``hope.hope_decode``) restores
the pretrained
   weights and reports the maximum absolute reconstruction error.
5. Plots accuracy versus density if matplotlib is installed.

Evaluation protocol
-------------------
The paper's numbers use the first 10,144 images of the ImageNet (ILSVRC 2012)
validation split as served
by TensorFlow Datasets (``imagenet2012``, TFDS order, no shuffling). Each image
is resized to 224x224 with
``tf.image.resize`` (bilinear, no antialiasing, aspect ratio not preserved, no
crop) and normalized with
``keras.applications.resnet50.preprocess_input`` ('caffe' mode: RGB to BGR and
subtraction of the ImageNet
channel means), in float32. The preprocessed images are written once to a
float32 ``.npy`` cache
(about 0.6 MB per image: 6.1 GB for 10,144 images, 30 GB for all 50,000) that
every later evaluation
memory-maps. TensorFlow and TensorFlow Datasets are needed only to build this
cache; ImageNet itself must
be downloaded manually and prepared with TFDS beforehand (see the TFDS
``imagenet2012`` catalog page).

Density is the number of nonzero entries over all weight tensors of the network
(including biases,
BatchNorm parameters and statistics, and the classifier) divided by the same
count for the pretrained
network.

Usage
-----
    python compress_resnet50.py --output_dir runs/resnet50
    python compress_resnet50.py --output_dir runs/resnet50 --backend jax
    --batch_size 32
    python compress_resnet50.py --output_dir runs/encode_only --eval_stride 0
    # no ImageNet needed

Outputs (in ``--output_dir``)
-----------------------------
    density_accuracy.csv     step,density,top1_accuracy (written incrementally;
                             single-method runs with evaluation enabled only)
    results_<METHOD>.csv     density,accuracy for the evaluated method(s) (only
                             written when evaluation is enabled)
    compressed_weights.npz   final compressed weights, keyed by Keras weight
                             path (single-method runs only)
    curriculum.pkl           dict with the curriculum (decoding order; replay it
                             with hope.hope_decode / hope_decode_step), the
                             quadruplets and the weight paths; unpickling
                             requires ``hope`` to be importable (single-method
                             runs only)
    summary.json             configuration, software versions, timings, action
                             counts, decoding check (single-method runs only)
    accuracy_vs_density.png  accuracy-versus-density plot (when evaluation is
                             enabled and ``--no_plot`` is not set)
"""

import argparse
import json
import logging
import os
import pickle
import platform
import re
import sys
import time
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple
import warnings

import hope
import hope_baselines
import numpy as np
import plot_compression_curves
import scipy

# Only silences TensorFlow's C++ start-up messages; respects a value set by the
# user.
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

logger = logging.getLogger("compress_resnet50")

IMAGE_SIZE = 224
PAPER_NUM_EVAL_IMAGES = 10144
IMAGENET_VAL_SIZE = 50000
# Max |decoded - pretrained| accepted as lossless (float32 round-off only).
DECODE_TOLERANCE = 1e-5

CSV_NAME = "density_accuracy.csv"
WEIGHTS_NAME = "compressed_weights.npz"
CURRICULUM_NAME = "curriculum.pkl"
SUMMARY_NAME = "summary.json"
PLOT_NAME = "accuracy_vs_density.png"

WeightsDict = Dict[str, np.ndarray]


# ----------------------------------------------------------------------------------------------------
# Model
# ----------------------------------------------------------------------------------------------------


def import_keras(backend: Optional[str] = None) -> Any:
  """Imports Keras 3, selecting ``backend`` first if given."""
  if backend is not None:
    if "keras" in sys.modules and os.environ.get("KERAS_BACKEND") != backend:
      logger.warning(
          "Keras was already imported; --backend %s may have no effect.",
          backend,
      )
    os.environ["KERAS_BACKEND"] = backend
  # Imported lazily: KERAS_BACKEND must be set first.
  import keras  # pylint: disable=g-import-not-at-top

  if int(keras.__version__.split(".")[0]) < 3:
    raise RuntimeError(f"Keras >= 3 is required (found {keras.__version__}).")
  return keras


def load_resnet50(keras: Any, weights: str) -> Any:
  """Builds Keras ResNet-50 with its 1000-way classifier and loads ``weights``.

  Args:
    keras: The imported Keras module.
    weights: ``"imagenet"`` (Keras downloads the official pretrained weights) or
      a path to a weights file (e.g. a local copy of
      ``resnet50_weights_tf_dim_ordering_tf_kernels.h5``).

  Returns:
    The Keras model.

  Raises:
    FileNotFoundError: If ``weights`` is not ``"imagenet"`` and does not exist.
  """
  if weights != "imagenet" and not os.path.isfile(weights):
    raise FileNotFoundError(f"--weights: no such file: {weights}")
  # The cached evaluation images are NHWC.
  keras.config.set_image_data_format("channels_last")
  return keras.applications.ResNet50(weights=weights, include_top=True)


def get_weights_dict(keras: Any, model: Any) -> WeightsDict:
  """Returns ``{weight path: NumPy copy}`` in ``model.weights`` order.

  Args:
    keras: The imported Keras module.
    model: Keras model whose weights are extracted.

  Returns:
    Dictionary mapping weight paths (e.g. ``"conv1_conv/kernel"``) to NumPy
    arrays.
  """
  with warnings.catch_warnings():
    # NumPy >= 2 warns that some backend arrays lack ``__array__(copy=...)``;
    # harmless here.
    warnings.simplefilter("ignore", DeprecationWarning)
    return {
        v.path: np.array(keras.ops.convert_to_numpy(v)) for v in model.weights
    }


def count_nonzero_params(weights: WeightsDict) -> int:
  """Number of nonzero entries over all tensors (the numerator/denominator of the density)."""
  return int(sum(np.count_nonzero(v) for v in weights.values()))


# ----------------------------------------------------------------------------------------------------
# ImageNet evaluation set (preprocessed once, cached as a memory-mapped .npy
# file)
# ----------------------------------------------------------------------------------------------------


def default_cache_dir() -> str:
  """``$XDG_CACHE_HOME/hope`` or ``~/.cache/hope``."""
  base = os.environ.get("XDG_CACHE_HOME") or os.path.join(
      os.path.expanduser("~"), ".cache"
  )
  return os.path.join(base, "hope")


def _cache_prefix(tfds_name: str) -> str:
  return re.sub(r"[^A-Za-z0-9.]+", "-", tfds_name).strip("-") + "_validation"


def _cache_paths(
    cache_dir: str, tfds_name: str, num_images: int
) -> Tuple[str, str]:
  stem = os.path.join(
      cache_dir, f"{_cache_prefix(tfds_name)}_first{num_images}_{IMAGE_SIZE}px"
  )
  return stem + "_images.npy", stem + "_labels.npy"


def find_cached_eval_set(
    cache_dir: str, tfds_name: str, num_images: int
) -> Optional[Tuple[str, str]]:
  """Returns the paths of the smallest complete cache holding at least ``num_images`` images, if any."""
  if not os.path.isdir(cache_dir):
    return None
  pattern = re.compile(
      re.escape(_cache_prefix(tfds_name))
      + rf"_first(\d+)_{IMAGE_SIZE}px_images\.npy$"
  )
  best = None
  for fname in os.listdir(cache_dir):
    match = pattern.match(fname)
    if match is None:
      continue
    n = int(match.group(1))
    images_path, labels_path = _cache_paths(cache_dir, tfds_name, n)
    if (
        n >= num_images
        and os.path.isfile(labels_path)
        and (best is None or n < best[0])
    ):
      best = (n, images_path, labels_path)
  return None if best is None else (best[1], best[2])


def build_eval_cache(
    cache_dir: str,
    tfds_name: str,
    data_dir: Optional[str],
    num_images: int,
    keras_backend: str,
    batch_size: int = 128,
) -> Tuple[str, str]:
  """Preprocesses ImageNet validation images and caches them on disk.

  Images are streamed from TFDS, resized with ``tf.image.resize`` (bilinear, no
  antialiasing, no crop), normalized with
  ``keras.applications.resnet50.preprocess_input`` and written to a float32
  ``.npy`` file of shape (num_images, 224, 224, 3); labels go to an int64
  ``.npy`` file. Files are written under a temporary name and renamed when
  complete, so an interrupted build never leaves a valid-looking cache.

  Args:
    cache_dir: Directory in which to store the cached ``.npy`` files.
    tfds_name: TensorFlow Datasets dataset name (e.g. ``"imagenet2012"``).
    data_dir: Optional TFDS data directory containing prepared dataset files.
    num_images: Number of validation images to preprocess from the split start.
    keras_backend: Active Keras backend name.
    batch_size: Batch size used while streaming from TFDS.

  Returns:
    ``(images_path, labels_path)``.

  Raises:
    RuntimeError: If the TFDS dataset cannot be loaded or has fewer than
      ``num_images`` validation images.
  """
  # TensorFlow and TFDS are only needed to build the cache, so they are
  # imported here.
  # pylint: disable=g-import-not-at-top
  from keras.applications import resnet50
  import tensorflow as tf
  import tensorflow_datasets as tfds
  # pylint: enable=g-import-not-at-top

  if keras_backend != "tensorflow":
    # TensorFlow only decodes and resizes images here; keep it off the
    # accelerators.
    try:
      tf.config.set_visible_devices([], "GPU")
    except RuntimeError:
      pass
  try:
    ds, info = tfds.load(
        tfds_name,
        split="validation",
        as_supervised=True,
        data_dir=data_dir,
        shuffle_files=False,
        download=False,
        with_info=True,
    )
  except Exception as e:
    raise RuntimeError(
        f"Could not load the TFDS dataset '{tfds_name}'"
        f" (data_dir={data_dir or 'TFDS default'}). ImageNet must be downloaded"
        " manually and prepared with TensorFlow Datasets first; see"
        " https://www.tensorflow.org/datasets/catalog/imagenet2012 ."
        " Alternatively run with --eval_stride 0 to encode without evaluation."
    ) from e

  def resize(image: Any, label: Any) -> Tuple[Any, Any]:
    return (
        tf.image.resize(image, (IMAGE_SIZE, IMAGE_SIZE)),
        label,
    )  # Bilinear, antialias=False.

  ds = ds.take(num_images).map(resize, num_parallel_calls=tf.data.AUTOTUNE)
  ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)

  os.makedirs(cache_dir, exist_ok=True)
  images_path, labels_path = _cache_paths(cache_dir, tfds_name, num_images)
  tmp_images_path = f"{images_path}.partial{os.getpid()}"
  tmp_labels_path = f"{labels_path}.partial{os.getpid()}"
  logger.info(
      "Building the evaluation cache (%d images, %.1f GB) in %s ...",
      num_images,
      num_images * IMAGE_SIZE * IMAGE_SIZE * 3 * 4 / 1e9,
      cache_dir,
  )
  t0 = time.time()
  images = np.lib.format.open_memmap(
      tmp_images_path,
      mode="w+",
      dtype=np.float32,
      shape=(num_images, IMAGE_SIZE, IMAGE_SIZE, 3),
  )
  labels = np.empty((num_images,), dtype=np.int64)
  pos, next_log = 0, 0
  try:
    for x, y in tfds.as_numpy(ds):
      n = x.shape[0]
      # NumPy input -> backend-independent 'caffe' preprocessing (RGB->BGR,
      # mean subtraction).
      images[pos : pos + n] = resnet50.preprocess_input(
          np.array(x, dtype=np.float32), data_format="channels_last"
      )
      labels[pos : pos + n] = y
      pos += n
      if pos >= next_log:
        logger.info(
            "  cached %d / %d images (%.0fs)", pos, num_images, time.time() - t0
        )
        next_log += max(num_images // 10, 1)
    if pos != num_images:
      raise RuntimeError(
          f"The validation split has only {pos} images; {num_images} were"
          " requested."
      )
    images.flush()
    images = None  # Release the memory map before renaming the file.
    with open(tmp_labels_path, "wb") as f:
      np.save(f, labels)
    os.replace(tmp_labels_path, labels_path)
    os.replace(
        tmp_images_path, images_path
    )  # Commit point: the images file marks a complete cache.
  finally:
    images = None
    for path in (tmp_images_path, tmp_labels_path):
      if os.path.exists(path):
        os.remove(path)
  meta = {
      "tfds_name": tfds_name,
      "dataset_version": str(info.version),
      "split": "validation",
      "num_images": num_images,
      "preprocessing": (
          "tf.image.resize(224x224, bilinear, antialias=False) -> "
          "keras.applications.resnet50.preprocess_input (caffe), float32 NHWC"
      ),
      "tensorflow": tf.__version__,
      "tensorflow_datasets": tfds.__version__,
  }
  with open(images_path[: -len("_images.npy")] + "_meta.json", "w") as f:
    json.dump(meta, f, indent=1)
  logger.info(
      "Evaluation cache ready in %.0fs (%s %s).",
      time.time() - t0,
      tfds_name,
      info.version,
  )
  return images_path, labels_path


def load_eval_set(
    cache_dir: str,
    tfds_name: str,
    data_dir: Optional[str],
    num_images: int,
    keras_backend: str,
) -> Tuple[np.ndarray, np.ndarray]:
  """Returns (images, labels) for the first ``num_images`` validation images; images are memory-mapped."""
  found = find_cached_eval_set(cache_dir, tfds_name, num_images)
  if found is None:
    found = build_eval_cache(
        cache_dir, tfds_name, data_dir, num_images, keras_backend
    )
  else:
    logger.info("Using cached evaluation images: %s", found[0])
  try:
    images = np.load(found[0], mmap_mode="r")
    labels = np.load(found[1])
  except (ValueError, OSError, EOFError) as e:
    raise RuntimeError(
        f"Corrupt evaluation cache {found[0]}; delete it and rerun."
    ) from e
  if (
      images.ndim != 4
      or images.shape[1:] != (IMAGE_SIZE, IMAGE_SIZE, 3)
      or images.dtype != np.float32
      or len(images) < num_images
      or len(labels) != len(images)
  ):
    raise RuntimeError(
        f"Corrupt evaluation cache {found[0]}; delete it and rerun."
    )
  return images[:num_images], labels[:num_images]


class Top1Evaluator:
  """Top-1 accuracy of the Keras model with a given weights dict, on preprocessed images.

  The weights are copied into the model with ``model.set_weights`` (in
  ``model.weights`` order) and the
  images are streamed in fixed-size batches; the last batch is zero-padded so
  that compiled backends see a
  single input shape.
  """

  def __init__(
      self, model: Any, images: np.ndarray, labels: np.ndarray, batch_size: int
  ):
    self.model = model
    self.images = images
    self.labels = np.asarray(labels)
    self.batch_size = int(batch_size)
    self.paths = [v.path for v in model.weights]

  def __call__(self, weights: WeightsDict) -> float:
    self.model.set_weights([weights[p] for p in self.paths])
    n, bs = len(self.labels), self.batch_size
    correct = 0
    for start in range(0, n, bs):
      x = np.asarray(self.images[start : start + bs], dtype=np.float32)
      m = x.shape[0]
      if m < bs:
        x = np.concatenate(
            [x, np.zeros((bs - m,) + x.shape[1:], dtype=x.dtype)], axis=0
        )
      probs = np.asarray(self.model.predict_on_batch(x))[:m]
      correct += int(
          np.count_nonzero(
              np.argmax(probs, axis=-1) == self.labels[start : start + m]
          )
      )
    return correct / n


# ----------------------------------------------------------------------------------------------------
# Results
# ----------------------------------------------------------------------------------------------------


class DensityAccuracyLog:
  """Appends ``step,density,top1_accuracy`` rows to a CSV file, flushing every row to disk."""

  HEADER = "step,density,top1_accuracy\n"

  def __init__(self, path: str):
    self.path = path
    self.rows: List[Tuple[int, float, float]] = []
    with open(path, "w") as f:
      f.write(self.HEADER)

  def append(self, step: int, density: float, accuracy: float) -> None:
    self.rows.append((int(step), float(density), float(accuracy)))
    with open(self.path, "a") as f:
      f.write(f"{int(step)},{float(density)!r},{float(accuracy)!r}\n")


class EvalHook:
  """The ``eval_fn`` passed to ``hope_encode``: evaluates and logs the network every ``eval_stride`` steps.

  ``hope_encode`` calls it after encoding steps ``eval_stride``, ``2 *
  eval_stride``, ... with the current
  (compressed) weights, which must not be modified.
  """

  def __init__(
      self,
      evaluator: Callable[[WeightsDict], float],
      log: DensityAccuracyLog,
      total_params: int,
      eval_stride: int,
      t_start: float,
  ):
    self.evaluator = evaluator
    self.log = log
    self.total_params = total_params
    self.eval_stride = eval_stride
    self.t_start = t_start
    self.step = 0
    self.eval_seconds = 0.0

  def __call__(self, current_weights: WeightsDict) -> float:
    self.step += self.eval_stride
    # Same definition as the density reported by hope_encode.
    density = float(count_nonzero_params(current_weights)) / float(
        self.total_params
    )
    t0 = time.time()
    accuracy = self.evaluator(current_weights)
    seconds = time.time() - t0
    self.eval_seconds += seconds
    self.log.append(self.step, density, accuracy)
    logger.info(
        "step %5d | density %.4f | top-1 %6.2f%% | eval %5.1fs | elapsed %s",
        self.step,
        density,
        100.0 * accuracy,
        seconds,
        _hms(time.time() - self.t_start),
    )
    return accuracy


def plot_density_accuracy(
    rows: Sequence[Tuple[int, float, float]], path: str, num_images: int
) -> bool:
  """Saves an accuracy-versus-density plot; returns False if matplotlib is not installed."""
  try:
    import matplotlib  # pylint: disable=g-import-not-at-top

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # pylint: disable=g-import-not-at-top
  except ImportError:
    logger.info("matplotlib is not installed; skipping the plot.")
    return False
  density = 100.0 * np.array([r[1] for r in rows])
  accuracy = 100.0 * np.array([r[2] for r in rows])
  fig, ax = plt.subplots(figsize=(8, 5))
  ax.plot(density, accuracy, color="tab:orange", lw=1.8, marker="o", ms=2.5)
  ax.invert_xaxis()
  ax.set_xlabel(
      "Remaining parameter density (% of pretrained ResNet-50 parameters)"
  )
  ax.set_ylabel("Top-1 accuracy (%)")
  ax.set_title(
      f"HOPE on ResNet-50 (data-free), {num_images:,} ImageNet validation"
      " images"
  )
  ax.grid(True, alpha=0.3)
  fig.tight_layout()
  fig.savefig(path, dpi=150)
  plt.close(fig)
  return True


def _hms(seconds: float) -> str:
  seconds = int(round(seconds))
  return f"{seconds // 3600:d}:{seconds // 60 % 60:02d}:{seconds % 60:02d}"


def _versions(keras: Any) -> Dict[str, str]:
  return {
      "python": platform.python_version(),
      "numpy": np.__version__,
      "scipy": scipy.__version__,
      "keras": keras.__version__,
      "keras_backend": keras.backend.backend(),
  }


# ----------------------------------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------------------------------


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
  """Parses and validates the command line."""
  method_choices = ["HOPE", *hope_baselines.BASELINE_METHODS, "ALL"]
  parser = argparse.ArgumentParser(
      description=(
          "Compress an ImageNet-pretrained Keras ResNet-50 with HOPE or"
          " data-free baselines\nand record top-1 accuracy versus parameter"
          " density along the encoding trajectory."
      ),
      epilog=(
          "examples:\n  python compress_resnet50.py --output_dir"
          " runs/resnet50\n  python compress_resnet50.py --output_dir"
          " runs/resnet50 --method ALL\n  python compress_resnet50.py"
          " --output_dir runs/resnet50 --backend jax --batch_size 32\n  python"
          " compress_resnet50.py --output_dir runs/encode_only --eval_stride 0"
      ),
      formatter_class=argparse.RawDescriptionHelpFormatter,
  )
  parser.add_argument(
      "--method",
      choices=method_choices,
      default="HOPE",
      help=(
          "compression method to run: 'HOPE', one of the 7 baselines"
          f" {list(hope_baselines.BASELINE_METHODS)}, or 'ALL' to run HOPE and"
          " all 7 baselines (default: %(default)s)"
      ),
  )
  parser.add_argument(
      "--output_dir",
      default="hope_resnet50_output",
      help=(
          "directory for the CSV, compressed weights, curriculum, summary and"
          " plot (default: %(default)s)"
      ),
  )
  parser.add_argument(
      "--weights",
      default="imagenet",
      help=(
          "'imagenet' (Keras downloads the pretrained weights) or the path of a"
          " Keras ResNet-50 weights file such as"
          " resnet50_weights_tf_dim_ordering_tf_kernels.h5 (default:"
          " %(default)s)"
      ),
  )
  parser.add_argument(
      "--backend",
      choices=["tensorflow", "jax", "torch"],
      default=None,
      help=(
          "Keras backend (default: the KERAS_BACKEND environment variable or"
          " the Keras configuration file)"
      ),
  )
  parser.add_argument(
      "--eval_stride",
      type=int,
      default=10,
      help=(
          "evaluate top-1 accuracy every this many encoding steps; 0 disables"
          " evaluation (default: %(default)s)"
      ),
  )
  parser.add_argument(
      "--num_eval_images",
      type=int,
      default=PAPER_NUM_EVAL_IMAGES,
      help=(
          "number of ImageNet validation images to evaluate on, taken from the"
          f" start of the split, at most {IMAGENET_VAL_SIZE} (default:"
          " %(default)s, the paper protocol)"
      ),
  )
  parser.add_argument(
      "--batch_size",
      type=int,
      default=64,
      help="inference batch size (default: %(default)s)",
  )
  parser.add_argument(
      "--data_dir",
      default=None,
      help=(
          "TFDS data directory containing the prepared imagenet2012 dataset"
          " (default: the TFDS default, usually ~/tensorflow_datasets)"
      ),
  )
  parser.add_argument(
      "--tfds_name",
      default="imagenet2012",
      help=(
          "TFDS dataset name, optionally with a version such as"
          " imagenet2012:5.1.0, the version used in the paper (default:"
          " %(default)s)"
      ),
  )
  parser.add_argument(
      "--cache_dir",
      default=None,
      help=(
          "directory for the preprocessed evaluation images, about 0.6 MB per"
          " image (default: $XDG_CACHE_HOME/hope or ~/.cache/hope)"
      ),
  )
  parser.add_argument(
      "--no_plot",
      action="store_true",
      help="do not save the accuracy-versus-density plot",
  )
  parser.add_argument(
      "--log_level",
      default="INFO",
      choices=["DEBUG", "INFO", "WARNING", "ERROR"],
      help="logging verbosity (default: %(default)s)",
  )
  args = parser.parse_args(argv)
  if args.eval_stride < 0:
    parser.error("--eval_stride must be >= 0.")
  if not 1 <= args.num_eval_images <= IMAGENET_VAL_SIZE:
    parser.error(f"--num_eval_images must be in [1, {IMAGENET_VAL_SIZE}].")
  if args.batch_size < 1:
    parser.error("--batch_size must be >= 1.")
  return args


def run(args: argparse.Namespace) -> int:
  """Loads ResNet-50, encodes it with the requested method(s), saves and verifies the results."""
  t_start = time.time()
  os.makedirs(args.output_dir, exist_ok=True)
  evaluate = args.eval_stride > 0
  csv_path = os.path.join(args.output_dir, CSV_NAME)

  keras = import_keras(args.backend)
  versions = _versions(keras)
  logger.info(
      "Keras %s (%s backend), NumPy %s, SciPy %s, Python %s",
      versions["keras"],
      versions["keras_backend"],
      versions["numpy"],
      versions["scipy"],
      versions["python"],
  )
  model = load_resnet50(keras, args.weights)
  weights = get_weights_dict(keras, model)
  quadruplets = hope.find_bottleneck_quadruplets(list(weights))
  total_params = count_nonzero_params(weights)
  logger.info(
      "ResNet-50: %d weight tensors, %d nonzero parameters, %d bottleneck"
      " quadruplets.",
      len(weights),
      total_params,
      len(quadruplets),
  )

  evaluator: Optional[Top1Evaluator] = None
  pretrained_top1: Optional[float] = None
  eval_seconds = 0.0
  if evaluate:
    images, labels = load_eval_set(
        args.cache_dir or default_cache_dir(),
        args.tfds_name,
        args.data_dir,
        args.num_eval_images,
        versions["keras_backend"],
    )
    evaluator = Top1Evaluator(model, images, labels, args.batch_size)
    t0 = time.time()
    pretrained_top1 = evaluator(weights)
    eval_seconds += time.time() - t0
    logger.info(
        "step     0 | density 1.0000 | top-1 %6.2f%% | eval %5.1fs (pretrained"
        " network, %d images)",
        100.0 * pretrained_top1,
        time.time() - t0,
        args.num_eval_images,
    )

  # Full benchmark suite mode (--method ALL): run HOPE + all 7 baselines and
  # plot all curves.
  if args.method == "ALL":
    all_eval_fn: Optional[Callable[[WeightsDict], float]] = None
    if evaluate:
      assert evaluator is not None and pretrained_top1 is not None

      def _cached_eval(current_weights: WeightsDict) -> float:
        if count_nonzero_params(current_weights) == total_params and all(
            np.array_equal(current_weights[k], weights[k]) for k in weights
        ):
          return pretrained_top1
        return evaluator(current_weights)

      all_eval_fn = _cached_eval
    hope_csv = os.path.join(args.output_dir, "results_HOPE.csv")
    logger.info(
        "Running HOPE encoder (-> %s) ...", hope_csv if evaluate else "no eval"
    )
    hope.hope_encode(
        weights,
        quadruplets,
        eval_fn=all_eval_fn,
        eval_stride=args.eval_stride,
        output_file=hope_csv if evaluate else None,
    )
    logger.info("Running all 7 baseline encoders in %s ...", args.output_dir)
    hope_baselines.run_all_baselines(
        weights,
        quadruplets,
        eval_fn=all_eval_fn,
        eval_stride=args.eval_stride,
        output_dir=args.output_dir,
    )
    if evaluate and not args.no_plot:
      plot_path = os.path.join(args.output_dir, PLOT_NAME)
      plot_compression_curves.plot_compression_curves(
          data_dir=args.output_dir, save_path=plot_path
      )
    logger.info(
        "Done in %s. Outputs are in %s",
        _hms(time.time() - t_start),
        args.output_dir,
    )
    return 0

  # Single-method mode ('HOPE' or one of BASELINE_METHODS).
  log: Optional[DensityAccuracyLog] = None
  hook: Optional[EvalHook] = None
  if evaluate:
    assert pretrained_top1 is not None and evaluator is not None
    log = DensityAccuracyLog(csv_path)
    log.append(0, 1.0, pretrained_top1)
    hook = EvalHook(evaluator, log, total_params, args.eval_stride, t_start)

  logger.info(
      "Running the %s encoder%s ...",
      args.method,
      f" (evaluating every {args.eval_stride} steps -> {csv_path})"
      if evaluate
      else "",
  )
  t0 = time.time()
  if args.method == "HOPE":
    compressed, curriculum, _ = hope.hope_encode(
        weights,
        quadruplets,
        eval_fn=hook,
        eval_stride=args.eval_stride,
    )
    method_csv_name = "results_HOPE.csv"
  else:
    baseline_eval_fn: Optional[Callable[[WeightsDict], float]] = None
    if hook is not None:
      total_baseline_steps = sum(
          weights[f"{q[0]}/kernel"].shape[-1]
          for q in quadruplets
          if f"{q[0]}/kernel" in weights
      )
      expected_steps = [
          s
          for s in range(total_baseline_steps)
          if s % args.eval_stride == 0 or s < 5
      ]
      if total_baseline_steps > 0:
        expected_steps.append(total_baseline_steps)
      baseline_call = 0

      def baseline_hook(current_weights: WeightsDict) -> float:
        nonlocal baseline_call
        assert evaluator is not None and log is not None and hook is not None
        assert pretrained_top1 is not None
        idx = baseline_call
        baseline_call += 1
        if idx == 0:
          return pretrained_top1
        step_num = (
            expected_steps[idx]
            if idx < len(expected_steps)
            else total_baseline_steps
        )
        density = float(count_nonzero_params(current_weights)) / float(
            total_params
        )
        t_ev = time.time()
        acc = evaluator(current_weights)
        dt = time.time() - t_ev
        hook.eval_seconds += dt
        log.append(step_num, density, acc)
        logger.info(
            "step %5d | density %.4f | top-1 %6.2f%% | eval %5.1fs | elapsed"
            " %s",
            step_num,
            density,
            100.0 * acc,
            dt,
            _hms(time.time() - t_start),
        )
        return acc

      baseline_eval_fn = baseline_hook

    compressed, curriculum, _ = hope_baselines.baseline_encode(
        weights,
        quadruplets,
        benchmark_method=args.method,
        eval_fn=baseline_eval_fn,
        eval_stride=args.eval_stride,
        output_file="",
    )
    method_csv_name = hope_baselines.DEFAULT_BASELINE_FILES[args.method]
  t_encode = time.time() - t0
  num_steps = len(curriculum)
  final_density = float(count_nonzero_params(compressed)) / float(total_params)
  if hook is not None:
    assert evaluator is not None and log is not None
    eval_seconds += hook.eval_seconds
    if (
        log.rows[-1][0] != num_steps
    ):  # Also record the final compressed network.
      t0 = time.time()
      accuracy = evaluator(compressed)
      eval_seconds += time.time() - t0
      log.append(num_steps, final_density, accuracy)
      logger.info(
          "step %5d | density %.4f | top-1 %6.2f%% (final compressed network)",
          num_steps,
          final_density,
          100.0 * accuracy,
      )
    # Also write two-column results_<METHOD>.csv for plot_compression_curves.py
    # compatibility.
    method_csv_path = os.path.join(args.output_dir, method_csv_name)
    with open(method_csv_path, "w") as f_csv:
      f_csv.write("density,accuracy\n")
      for _, d_val, a_val in log.rows:
        f_csv.write(f"{float(d_val):.6f},{float(a_val):.6f}\n")

  action_counts: Dict[str, int] = {}
  for packet in curriculum:
    action_counts[packet.action] = action_counts.get(packet.action, 0) + 1
  logger.info(
      "Encoding finished: %d steps %s, final density %.4f, %s including"
      " evaluations (total evaluation time %s).",
      num_steps,
      action_counts,
      final_density,
      _hms(t_encode),
      _hms(eval_seconds),
  )

  # 3. Save the compressed weights and the curriculum (decoding order).
  weights_path = os.path.join(args.output_dir, WEIGHTS_NAME)
  curriculum_path = os.path.join(args.output_dir, CURRICULUM_NAME)
  np.savez(weights_path, **compressed)
  with open(curriculum_path, "wb") as f:
    pickle.dump(
        {
            "curriculum": curriculum,
            "quadruplets": [tuple(q) for q in quadruplets],
            "weight_paths": list(weights),
            "total_params": total_params,
        },
        f,
        protocol=pickle.HIGHEST_PROTOCOL,
    )
  logger.info("Saved %s and %s.", weights_path, curriculum_path)

  # 4. Lossless decoding check.
  t0 = time.time()
  decoded = hope.hope_decode(compressed, curriculum)
  t_decode = time.time() - t0
  if set(decoded) != set(weights):
    raise RuntimeError(
        "The decoded network does not have the same tensors as the pretrained"
        " network."
    )
  max_error = max(
      float(
          np.max(
              np.abs(
                  decoded[k].astype(np.float64) - weights[k].astype(np.float64)
              )
          )
      )
      for k in weights
  )
  lossless = max_error <= DECODE_TOLERANCE
  (logger.info if lossless else logger.error)(
      "Decoding check %s: max |decoded - pretrained| = %.3g over %d tensors"
      " (%.1fs).",
      "PASSED" if lossless else "FAILED",
      max_error,
      len(weights),
      t_decode,
  )

  # 5. Plot and summary.
  plot_path = None
  if log is not None and not args.no_plot:
    plot_path = os.path.join(args.output_dir, PLOT_NAME)
    if not plot_density_accuracy(log.rows, plot_path, args.num_eval_images):
      plot_path = None
  summary = {
      "config": vars(args),
      "versions": versions,
      "num_weight_tensors": len(weights),
      "total_nonzero_params": total_params,
      "num_quadruplets": len(quadruplets),
      "num_encoding_steps": num_steps,
      "action_counts": action_counts,
      "final_density": final_density,
      "pretrained_top1": pretrained_top1,
      "num_evaluations": len(log.rows) if log is not None else 0,
      "decode_max_abs_error": max_error,
      "decode_lossless": lossless,
      "seconds": {
          "encode_including_evaluation": t_encode,
          "evaluation": eval_seconds,
          "decode": t_decode,
          "total": time.time() - t_start,
      },
      "outputs": {
          "csv": csv_path if evaluate else None,
          "compressed_weights": weights_path,
          "curriculum": curriculum_path,
          "plot": plot_path,
      },
  }
  with open(os.path.join(args.output_dir, SUMMARY_NAME), "w") as f:
    json.dump(summary, f, indent=1)
  logger.info(
      "Done in %s. Outputs are in %s",
      _hms(time.time() - t_start),
      args.output_dir,
  )
  return 0 if lossless else 1


def main(argv: Optional[Sequence[str]] = None) -> int:
  """Command-line entry point.

  Args:
    argv: Optional sequence of command-line arguments (defaults to sys.argv).

  Returns:
    Exit code: 0 success, 1 decoding check failed, 2 setup error (e.g. ImageNet
    not prepared), 130 interrupted.
  """
  args = parse_args(argv)
  # Third-party libraries (TensorFlow, JAX, ...) log at WARNING; this script and
  # hope at --log_level.
  logging.basicConfig(
      level=logging.WARNING,
      format="%(asctime)s %(levelname)s %(name)s: %(message)s",
      datefmt="%Y-%m-%d %H:%M:%S",
  )
  for name in (
      logger.name,
      hope.__name__,
      hope_baselines.__name__,
      plot_compression_curves.__name__,
  ):
    logging.getLogger(name).setLevel(getattr(logging, args.log_level))
  try:
    return run(args)
  except (FileNotFoundError, RuntimeError) as e:
    logger.error("%s", e, exc_info=logger.isEnabledFor(logging.DEBUG))
    return 2
  except KeyboardInterrupt:
    csv_path = os.path.join(args.output_dir, CSV_NAME)
    logger.warning(
        "Interrupted.%s",
        f" Accuracy rows written so far are in {csv_path}"
        if args.eval_stride > 0 and os.path.exists(csv_path)
        else "",
    )
    return 130


if __name__ == "__main__":
  sys.exit(main())
