import hashlib
import importlib.metadata
import json
import platform
from pathlib import Path


TRACKED_PACKAGES = (
    "matplotlib",
    "numpy",
    "opencv-python-headless",
    "Pillow",
    "scikit-image",
    "scikit-learn",
    "scipy",
    "torch",
    "torchvision",
)


def sha256_file(path, block_size=1024 * 1024):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while block := handle.read(block_size):
            digest.update(block)
    return digest.hexdigest()


def software_versions():
    versions = {"python": platform.python_version()}
    for package in TRACKED_PACKAGES:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "not-installed"
    return versions


def write_json(path, payload):
    path = Path(path)
    temporary_path = path.with_suffix(f"{path.suffix}.tmp")
    with temporary_path.open("w") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    temporary_path.replace(path)
