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
# MONAI Spleen CT Segmentation - Batched Inference
# This example runs a batch-size sweep over a directory of NIfTI images to measure throughput on
# ROCm GPUs (or CPU if GPU is unavailable). It reports CSV summary metrics for each batch size.
#
# Usage:
#   python infer_batched.py --input-dir /data/spleen/imagesVal --bundle-dir ./models --verbose
#
# Notes:
# - Discovers *.nii and *.nii.gz files recursively under the input directory.
# - Uses the training-aligned preprocessing pipeline (Orientation + Spacing + ScaleIntensityRange + Resize(96^3)).
# - Automatically detects ROCm GPU via torch.cuda and reports HIP version if present.

import argparse
import glob
import os
import time
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
from monai.bundle import download, load
from monai.transforms import (
    Compose, LoadImage, EnsureChannelFirst, Orientation,
    Spacing, ScaleIntensityRange, Resize, EnsureType
)


def setup_logging(verbose: bool):
    # Simple print-based logging 
    def log(msg: str):
        if verbose:
            print(msg)
    return log


def get_image_files(input_dir: Path) -> List[Path]:
    nii_files = glob.glob(str(input_dir / "**" / "*.nii"), recursive=True)
    niigz_files = glob.glob(str(input_dir / "**" / "*.nii.gz"), recursive=True)
    return [Path(p) for p in sorted(set(nii_files + niigz_files))]


def build_transforms() -> Compose:
    return Compose([
        LoadImage(image_only=True),
        EnsureChannelFirst(),
        Orientation(axcodes="RAS"),
        Spacing(pixdim=(1.5, 1.5, 2.0), mode="bilinear"),
        ScaleIntensityRange(a_min=-57, a_max=164, b_min=0, b_max=1, clip=True),
        Resize(spatial_size=(96, 96, 96)),
        EnsureType()
    ])


def preprocess_batch(batch_paths: List[Path], pre_transforms: Compose, log_fn) -> torch.Tensor:
    batch_images = []
    for image_path in batch_paths:
        img = pre_transforms(str(image_path))
        if img.dim() == 3:
            img = img.unsqueeze(0)
        batch_images.append(img)
        log_fn(f"Loaded & transformed {image_path}")
    return torch.stack(batch_images)


def select_device() -> torch.device:
    if torch.cuda.is_available():
        hip_ver = getattr(torch.version, "hip", None)
        print(f"Using GPU device (ROCm/HIP backend). torch.version.hip={hip_ver}")
        return torch.device("cuda")
    print("Using CPU device")
    return torch.device("cpu")


def warmup(model: torch.nn.Module, batch_tensor: torch.Tensor, device: torch.device):
    model.to(device)
    model.eval()
    with torch.no_grad():
        for _ in range(2):
            _ = model(batch_tensor.to(device))


def run_inference(model, device, image_files, batch_size, pre_transforms, verbose) -> Tuple[float, float, float, float]:
    model = model.to(device)
    model.eval()
    num_images = len(image_files)
    batch_times = []
    total_start_time = time.perf_counter()

    for start in range(0, num_images, batch_size):
        end = min(start + batch_size, num_images)
        batch_paths = image_files[start:end]
        batch_tensor = preprocess_batch(batch_paths, pre_transforms, print if verbose else (lambda *_: None)).to(device)
        batch_start_time = time.perf_counter()
        with torch.no_grad():
            _ = model(batch_tensor)
        batch_time = time.perf_counter() - batch_start_time
        batch_times.append(batch_time)
        if verbose:
            print(f"[{device.type.upper()}] Batch {start//batch_size + 1}: {end - start} images in {batch_time:.3f}s")

    total_time = time.perf_counter() - total_start_time
    avg_batch_time = np.mean(batch_times) if batch_times else 0.0
    avg_time_per_image = (total_time / num_images) if num_images > 0 else 0.0
    throughput = (num_images / total_time) if total_time > 0 else float("nan")
    return total_time, avg_batch_time, avg_time_per_image, throughput


def main():
    parser = argparse.ArgumentParser(description="MONAI Spleen CT Segmentation - Batched Inference")
    parser.add_argument("--input-dir", type=Path, required=True, help="Directory containing NIfTI images (.nii/.nii.gz)")
    parser.add_argument("--bundle-dir", type=Path, default=Path("./models"), help="Directory to store/load the MONAI bundle")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose logging: show batch details")
    args = parser.parse_args()

    log = setup_logging(args.verbose)

    image_files = get_image_files(args.input_dir)
    num_images = len(image_files)
    if num_images == 0:
        print("No .nii or .nii.gz images found in input directory.")
        return

    print(f"Discovered {num_images} images in '{args.input_dir}':")
    for f in image_files:
        print(f)

    # Download and load model
    bundle_name = "spleen_ct_segmentation"
    download(name=bundle_name, bundle_dir=str(args.bundle_dir))
    model = load(name=bundle_name, bundle_dir=str(args.bundle_dir))

    pre_transforms = build_transforms()

    # Prepare batch sizes: powers of two up to number of images, ensuring to include num_images
    max_exp = int(np.floor(np.log2(num_images)))
    batch_sizes = [2 ** i for i in range(max_exp + 1)]
    if num_images not in batch_sizes:
        batch_sizes.append(num_images)

    device_cpu = torch.device("cpu")
    device_gpu = select_device()
    can_run_gpu = device_gpu.type == "cuda"

    summary_rows = []
    print("\nStarting batch-size sweep...\n")
    for batch_size in batch_sizes:
        print(f"\n===> Batch size: {batch_size} <===")

        # CPU warmup
        print("Warming up CPU...")
        warmup_tensor = preprocess_batch(image_files[:batch_size], pre_transforms, log)
        warmup(model, warmup_tensor, device_cpu)

        # CPU inference
        print("Inference on CPU...")
        total, avg_batch, avg_img, thru = run_inference(
            model, device_cpu, image_files, batch_size, pre_transforms, args.verbose
        )
        summary_rows.append(["CPU", batch_size, f"{total:.4f}", f"{avg_batch:.4f}", f"{avg_img:.4f}", f"{thru:.2f}"])

        # GPU warmup/inference
        if can_run_gpu:
            print("Warming up GPU...")
            warmup_tensor = preprocess_batch(image_files[:batch_size], pre_transforms, log)
            warmup(model, warmup_tensor, device_gpu)

            print("Inference on GPU...")
            total, avg_batch, avg_img, thru = run_inference(
                model, device_gpu, image_files, batch_size, pre_transforms, args.verbose
            )
            summary_rows.append(["GPU", batch_size, f"{total:.4f}", f"{avg_batch:.4f}", f"{avg_img:.4f}", f"{thru:.2f}"])
        else:
            print("No ROCm GPU detected (torch.cuda.is_available() == False). Skipping GPU inference.")

    # Summary in CSV format
    print("\nDevice,Batch size,Total time (s),Avg. batch time (s),Avg. time per image (s),Throughput (img/s)")
    for row in summary_rows:
        print(",".join(map(str, row)))


if __name__ == "__main__":
    main()
