# SPDX-License-Identifier: MIT

# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# MONAI Spleen CT Segmentation - Inference Script
# This example demonstrates running inference with a MONAI bundle on ROCm GPUs (or CPU if GPU is unavailable).
#
# Usage:
#   python infer.py --input /path/to/image.nii.gz --bundle-dir ./models --verbose
#   python infer.py --input /path/to/image.nii.gz
#   python infer.py --input /path/to/image.nii.gz 
#
# Notes:
# - The script will download the "spleen_ct_segmentation" bundle the first time it's run.
# - Device selection prefers ROCm-enabled PyTorch builds (reported as torch.cuda with HIP backend).
# - Outputs basic statistics about the predicted segmentation; extend to save NIfTI as needed.

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import torch
from monai.bundle import download, load
from monai.transforms import (
    Compose, LoadImage, EnsureChannelFirst, Orientation,
    Spacing, ScaleIntensityRange, ScaleIntensity, Resize, EnsureType
)
try:
    import nibabel as nib  # optional for saving outputs
except Exception:
    nib = None


def setup_logging(verbosity: int = 1):
    level = logging.WARNING if verbosity <= 0 else logging.INFO if verbosity == 1 else logging.DEBUG
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def select_device() -> torch.device:
    # In ROCm builds, torch.cuda.is_available() is true, and torch.version.hip is present
    if torch.cuda.is_available():
        hip_ver = getattr(torch.version, "hip", None)
        logging.info("Using GPU device (ROCm/HIP backend): torch.version.hip=%s", hip_ver)
        return torch.device("cuda")
    logging.info("Using CPU device")
    return torch.device("cpu")


def build_transforms() -> Compose:
    """
    Build preprocessing transforms corresponding to the training recipe.
    """
    return Compose([
        LoadImage(image_only=True),
        EnsureChannelFirst(),
        Orientation(axcodes="RAS"),
        Spacing(pixdim=(1.5, 1.5, 2.0), mode="bilinear"),
        ScaleIntensityRange(a_min=-57, a_max=164, b_min=0, b_max=1, clip=True),
        Resize(spatial_size=(96, 96, 96)),
        EnsureType()
    ])


def run_inference(input_path: Path, bundle_dir: Path, save_output: bool, output_path: Path):
    # Step 1: Download/load bundle
    bundle_name = "spleen_ct_segmentation"
    download(name=bundle_name, bundle_dir=str(bundle_dir))
    model = load(name=bundle_name, bundle_dir=str(bundle_dir))

    logging.info("Loaded model: %s", type(model))

    # Step 2: Transforms
    transforms = build_transforms()

    # Step 3: Load and preprocess image
    image = transforms(str(input_path))
    if image.dim() == 4:  # [C, D, H, W] -> add batch
        image = image.unsqueeze(0)
    logging.debug("Preprocessed image shape: %s | dtype: %s", tuple(image.shape), image.dtype)

    # Step 4: Device and inference
    device = select_device()
    model = model.to(device)
    image = image.to(device)

    model.eval()
    with torch.no_grad():
        output = model(image)
        pred = torch.argmax(output, dim=1).squeeze().cpu().numpy()

    # Step 5: Stats and optional save
    logging.info("--- Inference Results Summary ---")
    logging.info("Prediction shape: %s | dtype: %s", pred.shape, pred.dtype)
    uniq = np.unique(pred)
    logging.info("Unique labels: %s | min: %s | max: %s", uniq, pred.min(), pred.max())
    logging.info("# voxels spleen (class 1): %d | background (class 0): %d", (pred == 1).sum(), (pred == 0).sum())

    if save_output and nib is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        # Save as NIfTI with identity affine; for real-world usage, pass-through original affine
        out_img = nib.Nifti1Image(pred.astype(np.uint8), affine=np.eye(4))
        nib.save(out_img, str(output_path))
        logging.info("Saved prediction to: %s", output_path)
    elif save_output and nib is None:
        logging.warning("nibabel not installed; install nibabel to enable NIfTI output saving.")


def main():
    parser = argparse.ArgumentParser(description="MONAI Spleen CT Segmentation - Unified Inference")
    parser.add_argument("--input", type=Path, required=True, help="Path to input NIfTI image (.nii or .nii.gz)")
    parser.add_argument("--bundle-dir", type=Path, default=Path("./models"), help="Directory for MONAI bundle cache")
    parser.add_argument("--save-output", action="store_true", help="Save predicted segmentation as NIfTI")
    parser.add_argument("--output", type=Path, default=Path("./outputs/pred.nii.gz"),
                        help="Output path for saved prediction (used with --save-output)")
    parser.add_argument("-v", "--verbose", action="count", default=1, help="Increase verbosity (-v, -vv)")
    args = parser.parse_args()

    setup_logging(args.verbose)

    if not args.input.exists():
        logging.error("Input image not found: %s", args.input)
        sys.exit(1)

    run_inference(
        input_path=args.input,
        bundle_dir=args.bundle_dir,
        save_output=args.save_output,
        output_path=args.output,
    )


if __name__ == "__main__":
    main()
