"""Build the paired QLA/TileLang correctness fixes without modifying installed packages.

Requires Python 3.12, CUDA 13, patch, and the build tools in
flashqla_tilelang/build-requirements.txt. Run on local disk, not SSHFS.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile
import urllib.request
import zipfile


_ROOT = Path(__file__).resolve().parents[1]
_PATCHES = _ROOT / "flashqla_tilelang/patches"
_INPUTS = {
    "tilelang-0.1.9.tar.gz": (
        "https://files.pythonhosted.org/packages/56/70/5051f65821baa30a3d61fc48f8ba10c776490315e8c90f82559b92089756/"
        "tilelang-0.1.9.tar.gz",
        "287f727c913bb648fcf6c1968809ba3390e55eeed257a5c6bb9a80bc05966af4",
    ),
    "flash_qla-0.1.2-py3-none-any.whl": (
        "https://files.pythonhosted.org/packages/66/a6/07ddef0a8073ee721f3cc67fc0ee0d2bf91bc5637659d9e9ff4d1943249c/"
        "flash_qla-0.1.2-py3-none-any.whl",
        "fb9dfa8607df9c545cd583dd8c0248ac9b0b673a8fb0af042ba9ef07bbc30446",
    ),
}


def main() -> None:
    """Verify upstream artifacts, apply repository patches, and build both wheels."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=_ROOT / "tmp/attention_build_inputs")
    parser.add_argument("--work-dir", type=Path, required=True, help="New directory on the build machine's local disk.")
    parser.add_argument("--output-dir", type=Path, default=_ROOT / "flashqla_tilelang/dist")
    parser.add_argument("--jobs", type=int, default=8)
    args = parser.parse_args()
    if args.jobs < 1:
        parser.error("--jobs must be positive")
    inputs = args.input_dir.resolve()
    work = args.work_dir.resolve()
    output = args.output_dir.resolve()
    inputs.mkdir(parents=True, exist_ok=True)
    work.mkdir(parents=True, exist_ok=False)
    output.mkdir(parents=True, exist_ok=True)

    for filename, (url, expected) in _INPUTS.items():
        path = inputs / filename
        if not path.exists():
            with urllib.request.urlopen(url, timeout=60) as response, path.open("wb") as destination:
                while chunk := response.read(1024 * 1024):
                    destination.write(chunk)
        with path.open("rb") as stream:
            actual = hashlib.file_digest(stream, "sha256").hexdigest()
        if actual != expected:
            raise ValueError(f"SHA256 mismatch for {path}: {actual} != {expected}")

    with tarfile.open(inputs / "tilelang-0.1.9.tar.gz") as archive:
        archive.extractall(work, filter="data")
    compiler = work / "tilelang-0.1.9"
    for name in ("tilelang_named_barrier_ids.patch", "tilelang_build_compatibility.patch"):
        subprocess.run(["patch", "--batch", "--fuzz=0", "-p1", "-i", str(_PATCHES / name)], cwd=compiler, check=True)
    (compiler / "VERSION").write_text("0.1.9+gerty1\n")

    environment = os.environ | {
        "NO_VERSION_LABEL": "1",
        "GIT_CEILING_DIRECTORIES": str(work),
        "CMAKE_BUILD_PARALLEL_LEVEL": str(args.jobs),
        "USE_CUDA": "1",
        "USE_ROCM": "0",
        "USE_METAL": "0",
        "SOURCE_DATE_EPOCH": "1791244800",
    }
    subprocess.run(
        [sys.executable, "-m", "build", "--wheel", "--no-isolation", "--outdir", str(output),
         f"--config-setting=build-dir={work / 'cmake-build'}",
         "--config-setting=cmake.define.USE_CUDA=ON", "--config-setting=cmake.define.USE_ROCM=OFF",
         "--config-setting=cmake.define.USE_METAL=OFF", str(compiler)],
        env=environment, check=True,
    )

    subprocess.run(
        [sys.executable, "-m", "wheel", "unpack", str(inputs / "flash_qla-0.1.2-py3-none-any.whl"), "-d", str(work)],
        check=True,
    )
    qla = work / "flash_qla-0.1.2"
    subprocess.run(
        ["patch", "--batch", "--fuzz=0", "-p1", "-i", str(_PATCHES / "flash_qla_hopper_correctness.patch")],
        cwd=qla, check=True,
    )
    info = qla / "flash_qla-0.1.2.dist-info"
    metadata = (info / "METADATA").read_text()
    assert metadata.count("Version: 0.1.2\n") == 1
    assert metadata.count("Requires-Dist: tilelang==0.1.9\n") == 1
    metadata = metadata.replace("Version: 0.1.2\n", "Version: 0.1.2+gerty1\n")
    metadata = metadata.replace("Requires-Dist: tilelang==0.1.9\n", "Requires-Dist: tilelang==0.1.9+gerty1\n")
    (info / "METADATA").write_text(metadata)
    info.rename(qla / "flash_qla-0.1.2+gerty1.dist-info")
    initializer = qla / "flash_qla/__init__.py"
    initializer.write_text(initializer.read_text().replace('__version__ = "0.1.2"', '__version__ = "0.1.2+gerty1"'))
    subprocess.run([sys.executable, "-m", "wheel", "pack", str(qla), "-d", str(output)], env=environment, check=True)

    report = {"inputs": {name: digest for name, (_, digest) in _INPUTS.items()}, "wheels": {}}
    for name in ("tilelang-0.1.9+gerty1-cp38-abi3-linux_x86_64.whl", "flash_qla-0.1.2+gerty1-py3-none-any.whl"):
        path = output / name
        with zipfile.ZipFile(path) as archive:
            assert archive.testzip() is None, name
        with path.open("rb") as stream:
            report["wheels"][name] = hashlib.file_digest(stream, "sha256").hexdigest()
    (output / "build-manifest.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
