import argparse
import os
import pickle
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from torchvision.models import resnet50
from torchvision.transforms import InterpolationMode

from .provenance import sha256_file, software_versions, write_json


LEGACY_RESNET50_URL = (
    "https://download.pytorch.org/models/resnet50-19c8e357.pth"
)


class ResNet50Stage3(torch.nn.Module):
    """ResNet-50 truncated after layer3 with the manuscript weights."""

    def __init__(self, weights_path=None):
        super().__init__()
        backbone = resnet50(weights=None)
        if weights_path:
            weights_path = Path(weights_path)
            state_dict = torch.load(weights_path, map_location="cpu")
            if "state_dict" in state_dict:
                state_dict = state_dict["state_dict"]
            self.weights_source = str(weights_path.resolve())
            self.weights_sha256 = sha256_file(weights_path)
        else:
            state_dict = torch.hub.load_state_dict_from_url(
                LEGACY_RESNET50_URL,
                map_location="cpu",
                check_hash=True,
                progress=True,
            )
            checkpoint_path = (
                Path(torch.hub.get_dir())
                / "checkpoints"
                / Path(LEGACY_RESNET50_URL).name
            )
            self.weights_source = LEGACY_RESNET50_URL
            self.weights_sha256 = sha256_file(checkpoint_path)
        backbone.load_state_dict(state_dict, strict=True)
        self.features = torch.nn.Sequential(
            backbone.conv1,
            backbone.bn1,
            backbone.relu,
            backbone.maxpool,
            backbone.layer1,
            backbone.layer2,
            backbone.layer3,
            torch.nn.AdaptiveAvgPool2d(1),
        )

    def forward(self, images):
        return self.features(images).flatten(1)


def build_transform():
    return transforms.Compose(
        [
            transforms.Resize(
                (224, 224),
                interpolation=InterpolationMode.BILINEAR,
                antialias=True,
            ),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
            ),
        ]
    )


def patch_to_rgb_image(patch):
    patch = np.asarray(patch, dtype=np.float32)
    patch = np.clip(patch, 0.0, 1.0)
    # Match the manuscript implementation's truncating uint8 conversion.
    grayscale = (patch * 255).astype(np.uint8)
    return Image.fromarray(grayscale, mode="L").convert("RGB")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Extract 1024-dimensional ResNet-50 patch features"
    )
    parser.add_argument("--patches-dir", type=Path, required=True)
    parser.add_argument("--features-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument(
        "--weights-path",
        type=Path,
        help="Optional local resnet50-19c8e357.pth for offline use",
    )
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--cpu", action="store_true")
    return parser.parse_args()


def set_reproducibility(seed=0):
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)


def main():
    args = parse_args()
    set_reproducibility()
    device = torch.device(
        "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    )
    args.features_dir.mkdir(parents=True, exist_ok=True)
    patch_paths = sorted(args.patches_dir.glob("*.pkl"))
    if not patch_paths:
        raise FileNotFoundError(f"No patch files found in {args.patches_dir}")

    model = ResNet50Stage3(args.weights_path).to(device).eval()
    transform = build_transform()
    print(f"Using device: {device}")

    expected_outputs = {path.name for path in patch_paths}
    orphaned_outputs = {
        path.name for path in args.features_dir.glob("*.pkl")
    } - expected_outputs
    if orphaned_outputs:
        raise RuntimeError(
            "Feature directory contains outputs without matching patch bags: "
            f"{sorted(orphaned_outputs)}. Use a fresh output directory."
        )

    feature_records = []
    for patch_path in patch_paths:
        output_path = args.features_dir / patch_path.name
        patch_sha256 = sha256_file(patch_path)
        if args.skip_existing and output_path.exists():
            with output_path.open("rb") as handle:
                existing = pickle.load(handle)
            provenance = existing.get("provenance", {})
            if (
                provenance.get("patch_sha256") != patch_sha256
                or provenance.get("weights_sha256") != model.weights_sha256
            ):
                raise RuntimeError(
                    f"Existing features do not match the current patches or "
                    f"weights: {output_path}"
                )
            existing_features = np.asarray(existing["features"])
            feature_records.append(
                {
                    "file": patch_path.name,
                    "input_sha256": patch_sha256,
                    "output_sha256": sha256_file(output_path),
                    "instances": int(existing_features.shape[0]),
                    "feature_dimension": int(existing_features.shape[1]),
                }
            )
            continue

        with patch_path.open("rb") as handle:
            patch_data = pickle.load(handle)
        patches = patch_data.get("patches")
        coordinates = patch_data.get("coordinates")
        if patches is None or coordinates is None or len(patches) == 0:
            print(f"Skipping empty patch bag: {patch_path.name}")
            continue
        if len(patches) != len(coordinates):
            raise ValueError(f"Patch/coordinate mismatch in {patch_path}")

        feature_batches = []
        with torch.inference_mode():
            for start in range(0, len(patches), args.batch_size):
                images = torch.stack(
                    [
                        transform(patch_to_rgb_image(patch))
                        for patch in patches[start : start + args.batch_size]
                    ]
                ).to(device)
                feature_batches.append(model(images).cpu().numpy())

        payload = {
            "features": np.concatenate(feature_batches).astype(np.float32),
            "coordinates": coordinates,
            "tissue_fractions": patch_data.get("tissue_fractions"),
            "provenance": {
                "patch_file": patch_path.name,
                "patch_sha256": patch_sha256,
                "weights_sha256": model.weights_sha256,
                "weights_source": model.weights_source,
            },
        }
        with output_path.open("wb") as handle:
            pickle.dump(payload, handle)
        feature_records.append(
            {
                "file": patch_path.name,
                "input_sha256": patch_sha256,
                "output_sha256": sha256_file(output_path),
                "instances": int(payload["features"].shape[0]),
                "feature_dimension": int(payload["features"].shape[1]),
            }
        )
        print(f"{patch_path.name}: {payload['features'].shape}")

    write_json(
        args.features_dir / "feature_manifest.json",
        {
            "schema_version": 1,
            "source_directory": str(args.patches_dir.resolve()),
            "extractor": "ImageNet ResNet-50 truncated after layer3",
            "weights_source": model.weights_source,
            "weights_sha256": model.weights_sha256,
            "resize": [224, 224],
            "interpolation": "bilinear",
            "imagenet_mean": [0.485, 0.456, 0.406],
            "imagenet_std": [0.229, 0.224, 0.225],
            "software": software_versions(),
            "features": feature_records,
        },
    )


if __name__ == "__main__":
    main()
