# H200 Attention Kernel Fixes

These patches target `flash-qla==0.1.2` and `tilelang==0.1.9` and produce paired
`0.1.2+gerty1` / `0.1.9+gerty1` wheels:

- TileLang: preserve named barrier IDs allocated by earlier passes when inserting synchronization for
  `shared` / `shared.dyn` memory. Otherwise, disjoint thread groups may incorrectly share a barrier.
- FlashQLA: add 42 async proxy fences to the Hopper kernels, based on upstream commit
  `c4d72349d23ad66c506f2d509ebdbf17b39dfc4a`. In `prepare_h`, also guard `bar_3.arrive()` with
  the same `calc_mt` condition as its consumers.
- Build compatibility: require Cython <3.3 and enable C/CXX in CMake before CUDA/Threads detection.
  These two changes only address build failures.

Upstream sources: [TileLang](https://github.com/tile-ai/tilelang) and
[FlashQLA](https://github.com/QwenLM/FlashQLA), both under the MIT license.
The wheels retain the upstream licenses.

## Archived wheels

The paired wheels are available in the public
[attention-fixes-gerty1 release](https://github.com/EmbodiedForge/gerty-patches/releases/tag/attention-fixes-gerty1).
They reproduce the original patches; Gerty now uses upstream dependency overrides.
Historical findings are recorded in [the attribution report](reports/gradient_diagnosis_20261012/README.md).

## Build from source

Run on Linux x86_64 with Python 3.12 and CUDA 13. A C/C++ compiler, CMake >=3.26, Ninja, and `patch` are required.
The existing wheels were built with CMake 3.31.4 and Ninja 1.11.1.
Use a build directory on local disk rather than SSHFS. Run the following commands from the repository root;
they use an isolated build environment and leave the training `.venv` unchanged.

```bash
uv venv --python 3.12 /tmp/gerty-attention-build-env
uv pip install --python /tmp/gerty-attention-build-env/bin/python \
  -r flashqla_tilelang/build-requirements.txt
PATH="/tmp/gerty-attention-build-env/bin:$PATH" \
  /tmp/gerty-attention-build-env/bin/python flashqla_tilelang/build.py \
  --work-dir /tmp/gerty-attention-build --jobs 8
```

The script downloads and verifies both upstream packages, applies the patches, and writes the wheels and
a SHA256 manifest to `flashqla_tilelang/dist/`. Each build requires a new `--work-dir`.
For offline builds, use `--input-dir` to point to the two downloaded upstream packages;
their filenames and checksums are listed in `build.py`. Build dependencies must also be cached or installed in advance.
Keep generated wheels and extracted source trees out of Git.

Install the resulting wheels together in an isolated environment. This build does not change
Gerty's dependencies or any installed environment.
Do not rename unpatched upstream wheels to match the patched versions.
