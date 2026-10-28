<!-- Copyright 2026 Google LLC -->
<!-- SPDX-License-Identifier: Apache-2.0 -->
<!-- Licensed under the Apache License, Version 2.0 (http://www.apache.org/licenses/LICENSE-2.0) -->
# Examples

## Synthetic compression demo

`synthetic_compression_demo.py` is a one-minute, self-contained tour of
HOPE. It needs only NumPy and SciPy — no Keras, no TensorFlow, and no
ImageNet download.

```bash
python examples/synthetic_compression_demo.py
```

Useful flags: `--seed N` builds a different synthetic network (default 0);
`--quiet` prints only the check results and the final line. The demo exits
with status 0 when every check passes and 1 otherwise, so it can also be
used as a smoke test after changing `hope.py`.

### What it shows

1. **A synthetic network.** The demo builds a tiny ResNet-style weights
   dictionary (two bottleneck blocks, Keras tensor names) with a fixed
   seed, so every run is identical. Two neurons of the first block are
   near-duplicates, which gives the encoder a reason to MERGE as well as
   PRUNE and EVICT.
2. **Encoding.** `hope.hope_encode` compresses the network to completion.
   The demo prints the curriculum summary — how many steps of each action
   type the encoder chose — and the density trajectory: the fraction of
   parameters remaining after a quarter, half, three quarters, and all of
   the steps. There is no target density; the encoder runs until no
   admissible action is left.
3. **Lossless full decoding.** `hope.hope_decode` undoes every step, and
   the demo verifies that the original weights come back (maximum absolute
   difference at float32 round-off level).
4. **Progressive decoding.** Decoding only half of the curriculum lands on
   an intermediate model at a density between the compressed and original
   networks. Decoding the remaining steps *from that intermediate model*
   restores the original weights again — this is the property that makes
   one HOPE curriculum serve every density along the way.

### Going further

This demo uses random synthetic weights, so the densities and step counts
are illustrative, not accuracy results. For the real workflow — an
ImageNet-pretrained Keras ResNet-50, top-1 accuracy versus density, the
seven data-free baselines, and saved curricula — see
[`compress_resnet50.py`](../compress_resnet50.py) in the repository root
and the main [README](../README.md).
