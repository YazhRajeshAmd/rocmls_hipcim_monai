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
# MONAI Spleen CT Segmentation - Training Script
# This example trains a 3D UNet for spleen CT segmentation using the MSD Task09_Spleen dataset.
# It includes data preparation (download/extract), a training-aligned transform pipeline, validation,
# Dice metric reporting, and optional checkpoint saving. Designed to run on ROCm GPUs or CPU.
#
# Usage:
#   python train.py --data-root ./data/spleen_ct_seg/spleen_data --epochs 10 --batch-size 2 --save-checkpoint
#
# Notes:
# - For multi-GPU, launch with torchrun --nproc_per_node=N and add DistributedSampler; this script is single-GPU/CPU.

import argparse
import logging
import os
import tarfile
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

import monai
from monai.apps import download_url
from monai.data import CacheDataset
from monai.metrics import DiceMetric
from monai.losses import DiceLoss
from monai.networks.nets import UNet
from monai.transforms import (
    Compose, EnsureChannelFirstd, LoadImaged,
    Orientationd, RandCropByPosNegLabeld, RandFlipd,
    ScaleIntensityRanged, Spacingd, ToTensord
)
from sklearn.model_selection import train_test_split


def setup_logging(verbosity: int = 1):
    level = logging.WARNING if verbosity <= 0 else logging.INFO if verbosity == 1 else logging.DEBUG
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def seed_everything(seed: int = 42):
    import random, os
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def select_device() -> torch.device:
    if torch.cuda.is_available():
        hip_ver = getattr(torch.version, "hip", None)
        logging.info("Using GPU device (ROCm/HIP backend). torch.version.hip=%s", hip_ver)
        return torch.device("cuda")
    logging.info("Using CPU device")
    return torch.device("cpu")


def is_nii(filename: str) -> bool:
    return filename.endswith(".nii") or filename.endswith(".nii.gz")


def ensure_dataset(data_root: Path) -> Path:
    """
    Download and extract MSD Task09_Spleen dataset into data_root/Task09_Spleen if not present.
    Returns the path to the dataset folder.
    """
    resource = "https://msd-for-monai.s3-us-west-2.amazonaws.com/Task09_Spleen.tar"
    dataset_dir = data_root / "Task09_Spleen"
    tar_path = data_root / "Task09_Spleen.tar"

    if not dataset_dir.exists():
        data_root.mkdir(parents=True, exist_ok=True)
        if not tar_path.exists():
            logging.info("Downloading dataset: %s -> %s", resource, tar_path)
            download_url(resource, str(tar_path))
        logging.info("Extracting dataset: %s", tar_path)
        with tarfile.open(tar_path) as tar:
            tar.extractall(path=str(data_root))

    return dataset_dir


def build_dataloaders(dataset_dir: Path, batch_size: int, num_workers: int, val_frac: float, seed: int
                      ) -> Tuple[DataLoader, DataLoader]:
    images_dir = dataset_dir / "imagesTr"
    labels_dir = dataset_dir / "labelsTr"

    train_images = sorted([str(images_dir / f) for f in os.listdir(images_dir) if is_nii(f) and not f.startswith("._")])
    train_labels = sorted([str(labels_dir / f) for f in os.listdir(labels_dir) if is_nii(f) and not f.startswith("._")])

    files = [{"image": img, "label": seg} for img, seg in zip(train_images, train_labels)]
    train_files, val_files = train_test_split(files, test_size=val_frac, random_state=seed)

    transforms = Compose([
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        Spacingd(keys=["image", "label"], pixdim=(1.5, 1.5, 2.0), mode=("bilinear", "nearest")),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        ScaleIntensityRanged(
            keys=["image"], a_min=-175, a_max=250, b_min=0.0, b_max=1.0, clip=True
        ),
        RandCropByPosNegLabeld(
            keys=["image", "label"], label_key="label",
            spatial_size=(96, 96, 96), pos=1, neg=1, num_samples=4,
            image_key="image", image_threshold=0,
        ),
        RandFlipd(keys=["image", "label"], prob=0.2, spatial_axis=0),
        RandFlipd(keys=["image", "label"], prob=0.2, spatial_axis=1),
        RandFlipd(keys=["image", "label"], prob=0.2, spatial_axis=2),
        ToTensord(keys=["image", "label"]),
    ])

    train_ds = CacheDataset(train_files, transform=transforms, cache_rate=1.0, num_workers=num_workers)
    val_ds = CacheDataset(val_files, transform=transforms, cache_rate=1.0, num_workers=num_workers)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_loader, val_loader


def build_model() -> torch.nn.Module:
    return UNet(
        spatial_dims=3,
        in_channels=1,
        out_channels=2,
        channels=(16, 32, 64, 128, 256),
        strides=(2, 2, 2, 2),
        num_res_units=2,
        norm=monai.networks.layers.Norm.BATCH,
    )


def train_one_epoch(model, loader, loss_fn, optimizer, device) -> float:
    model.train()
    epoch_loss = 0.0
    for i, batch in enumerate(loader):
        # Debug: Print batch structure for first batch only
        if i == 0:
            print(f"Batch type: {type(batch)}")
            if isinstance(batch, dict):
                print(f"Batch keys: {batch.keys()}")
                print(f"Image shape: {batch['image'].shape if 'image' in batch else 'No image key'}")
            elif isinstance(batch, (list, tuple)):
                print(f"Batch length: {len(batch)}")
                print(f"First item type: {type(batch[0])}")
                print(f"Second item type: {type(batch[1]) if len(batch) > 1 else 'No second item'}")
        
        # Handle different batch formats
        try:
            if isinstance(batch, dict):
                inputs = batch["image"].to(device)
                labels = batch["label"].to(device)
            elif isinstance(batch, (list, tuple)) and len(batch) >= 2:
                # Handle list/tuple format
                if isinstance(batch[0], dict):
                    # List of dictionaries - extract and concatenate properly
                    images = [item["image"] for item in batch]
                    labels_list = [item["label"] for item in batch]
                    
                    # Check if we need to add batch dimension or concatenate
                    if images[0].dim() == 4:  # [C, D, H, W] - add batch dimension
                        inputs = torch.stack(images, dim=0).to(device)  # [B, C, D, H, W]
                        labels = torch.stack(labels_list, dim=0).to(device)
                    else:  # Already has batch dimension, concatenate
                        inputs = torch.cat(images, dim=0).to(device)
                        labels = torch.cat(labels_list, dim=0).to(device)
                else:
                    # Direct tensor format [images, labels]
                    inputs = batch[0].to(device)
                    labels = batch[1].to(device)
            else:
                raise ValueError(f"Unexpected batch format: {type(batch)}")
            
            # Debug: Print tensor shapes for first batch
            if i == 0:
                print(f"Input shape: {inputs.shape}")
                print(f"Labels shape: {labels.shape}")
                
        except Exception as e:
            print(f"Error processing batch {i}: {e}")
            print(f"Batch structure: {batch}")
            raise
        
        optimizer.zero_grad()
        outputs = model(inputs)
        # print("output device: ", outputs.device)
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    return epoch_loss / max(1, len(loader))


def validate(model, loader, device) -> float:
    dice_metric = DiceMetric(include_background=False, reduction="mean", ignore_empty=True)
    model.eval()
    with torch.no_grad():
        for i, batch in enumerate(loader):
            # Handle different batch formats (same logic as training)
            try:
                if isinstance(batch, dict):
                    inputs = batch["image"].to(device)
                    labels = batch["label"].to(device)
                elif isinstance(batch, (list, tuple)) and len(batch) >= 2:
                    # Handle list/tuple format
                    if isinstance(batch[0], dict):
                        # List of dictionaries - extract and concatenate properly
                        images = [item["image"] for item in batch]
                        labels_list = [item["label"] for item in batch]
                        
                        # Check if we need to add batch dimension or concatenate
                        if images[0].dim() == 4:  # [C, D, H, W] - add batch dimension
                            inputs = torch.stack(images, dim=0).to(device)  # [B, C, D, H, W]
                            labels = torch.stack(labels_list, dim=0).to(device)
                        else:  # Already has batch dimension, concatenate
                            inputs = torch.cat(images, dim=0).to(device)
                            labels = torch.cat(labels_list, dim=0).to(device)
                    else:
                        # Direct tensor format [images, labels]
                        inputs = batch[0].to(device)
                        labels = batch[1].to(device)
                else:
                    raise ValueError(f"Unexpected batch format: {type(batch)}")
                    
            except Exception as e:
                print(f"Error processing validation batch {i}: {e}")
                print(f"Batch structure: {batch}")
                raise
            
            # For validation, we usually run a sliding window inferer; for simplicity, direct forward:
            outputs = model(inputs)
            # Convert to discrete predictions
            outputs = torch.softmax(outputs, dim=1)
            dice_metric(y_pred=outputs, y=monai.networks.one_hot(labels, num_classes=2))
        mean_dice = dice_metric.aggregate().item()
        dice_metric.reset()
    return mean_dice


def main():
    parser = argparse.ArgumentParser(description="Train MONAI 3D UNet for Spleen CT Segmentation (ROCm-ready)")
    parser.add_argument("--data-root", type=Path, default=Path("./data/spleen_ct_seg/spleen_data"),
                        help="Root directory for dataset (Task09_Spleen will be placed here)")
    parser.add_argument("--batch-size", type=int, default=2, help="Training batch size")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--num-workers", type=int, default=2, help="DataLoader workers")
    parser.add_argument("--val-frac", type=float, default=0.2, help="Validation fraction")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--save-checkpoint", action="store_true", help="Save best checkpoint to ./checkpoints/best.pt")
    parser.add_argument("-v", "--verbose", action="count", default=1, help="Increase verbosity (-v, -vv)")
    args = parser.parse_args()

    setup_logging(args.verbose)
    seed_everything(args.seed)
    device = select_device()

    dataset_dir = ensure_dataset(args.data_root)
    train_loader, val_loader = build_dataloaders(dataset_dir, args.batch_size, args.num_workers, args.val_frac, args.seed)

    model = build_model().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = DiceLoss(to_onehot_y=True, softmax=True)

    best_dice = -1.0
    ckpt_dir = Path("./checkpoints")
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()
        train_loss = train_one_epoch(model, train_loader, loss_fn, optimizer, device)
        val_dice = validate(model, val_loader, device)
        elapsed = time.time() - epoch_start

        logging.info(f"Epoch {epoch}/{args.epochs} | Train Loss: {train_loss:.4f} | Val Dice: {val_dice:.4f} | Time: {elapsed:.1f}s")

        if args.save_checkpoint and val_dice > best_dice:
            best_dice = val_dice
            ckpt_path = ckpt_dir / "best.pt"
            torch.save({"epoch": epoch, "model_state": model.state_dict(), "val_dice": val_dice}, ckpt_path)
            logging.info("Saved checkpoint: %s", ckpt_path)

    logging.info("Training complete. Best Val Dice: %.4f", best_dice)


if __name__ == "__main__":
    main()
