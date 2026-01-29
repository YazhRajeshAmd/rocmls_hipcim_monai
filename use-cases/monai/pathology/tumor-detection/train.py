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
# Pathology Tumor Detection - Training Script (ROCm-ready)
# This example trains a tile classifier for tumor detection using MONAI's TorchVisionFCModel
# and WSI patching utilities. It reads training/validation CSVs for a Camelyon-like dataset and
# applies simple transforms, then reports basic validation metrics.
#
# Usage:
#   python train.py \
#     --image-root sample_images/camelyon \
#     --train-csv demo/data/pathology_tumor_detection/training.csv \
#     --valid-csv demo/data/pathology_tumor_detection/validation.csv \
#     --epochs 2 --batch-size 32

import argparse
import logging
import os
import numpy as np
import torch
from torch.utils.data import DataLoader
from monai.data import CSVDataset, PatchWSIDataset
from monai.networks.nets import TorchVisionFCModel
from monai.transforms import Compose, Lambdad, GridSplitd, ToTensord
from typing import Tuple


def setup_logging(verbosity: int = 1):
    level = logging.WARNING if verbosity <= 0 else logging.INFO if verbosity == 1 else logging.DEBUG
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def select_device() -> torch.device:
    if torch.cuda.is_available():
        logging.info("Using GPU device (ROCm/HIP backend). torch.version.hip=%s", getattr(torch.version, "hip", "unknown"))
        return torch.device("cuda")
    logging.info("Using CPU device")
    return torch.device("cpu")


def build_dataloaders(image_root: str, train_csv: str, valid_csv: str, grid_shape: int,
                      region_size: int, patch_size: int, batch_size: int, num_workers: int
                      ) -> Tuple[DataLoader, DataLoader]:
    preprocess_cpu = Compose([
        Lambdad(keys="label", func=lambda x: x.reshape((1, grid_shape, grid_shape))),
        GridSplitd(
            keys=("image", "label"),
            grid=(grid_shape, grid_shape),
            size={"image": patch_size, "label": 1},
        ),
        ToTensord(keys="label"),
    ])

    train_data_list = CSVDataset(
        train_csv,
        col_groups={"image": 0, "location": [2, 1], "label": [3, 6, 9, 4, 7, 10, 5, 8, 11]},
        kwargs_read_csv={"header": None},
        transform=Lambdad("image", lambda x: os.path.join(image_root, x + ".tif")),
    )
    train_dataset = PatchWSIDataset(
        data=train_data_list,
        patch_size=region_size,
        patch_level=0,
        transform=preprocess_cpu,
        reader="cuCIM",
    )
    train_loader = DataLoader(train_dataset, num_workers=num_workers, batch_size=batch_size, pin_memory=True)

    valid_data_list = CSVDataset(
        valid_csv,
        col_groups={"image": 0, "location": [2, 1], "label": [3, 6, 9, 4, 7, 10, 5, 8, 11]},
        kwargs_read_csv={"header": None},
        transform=Lambdad("image", lambda x: os.path.join(image_root, x + ".tif")),
    )
    valid_dataset = PatchWSIDataset(
        data=valid_data_list,
        patch_size=region_size,
        patch_level=0,
        transform=preprocess_cpu,
        reader="cuCIM",
    )
    val_loader = DataLoader(valid_dataset, num_workers=num_workers, batch_size=batch_size, pin_memory=True)
    return train_loader, val_loader


def build_model(device: torch.device) -> torch.nn.Module:
    model = TorchVisionFCModel("resnet18", num_classes=1, use_conv=True, pretrained=False)
    return model.to(device)


def train_and_validate(model, train_loader, val_loader, device, epochs: int, lr: float):
    loss_function = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        for i, batch_data in enumerate(train_loader):
            # Debug: Print batch structure for first batch only
            if i == 0:
                print(f"Batch type: {type(batch_data)}")
                if isinstance(batch_data, dict):
                    print(f"Batch keys: {batch_data.keys()}")
                    if "image" in batch_data:
                        print(f"Image shape: {batch_data['image'].shape}")
                    if "label" in batch_data:
                        print(f"Label shape: {batch_data['label'].shape}")
                elif isinstance(batch_data, (list, tuple)):
                    print(f"Batch length: {len(batch_data)}")
                    for j, item in enumerate(batch_data[:3]):  # Show first 3 items
                        print(f"Item {j} type: {type(item)}")
            
            # Handle different batch formats
            try:
                if isinstance(batch_data, dict):
                    x = batch_data["image"].to(device).float()
                    y = batch_data["label"].to(device).float()
                elif isinstance(batch_data, (list, tuple)) and len(batch_data) >= 2:
                    # Handle list/tuple format
                    if isinstance(batch_data[0], dict):
                        # List of dictionaries - extract and stack
                        images = [item["image"] for item in batch_data]
                        labels = [item["label"] for item in batch_data]
                        x = torch.stack(images, dim=0).to(device).float()
                        y = torch.stack(labels, dim=0).to(device).float()
                    else:
                        # Direct tensor format [images, labels]
                        x = batch_data[0].to(device).float()
                        y = batch_data[1].to(device).float()
                else:
                    raise ValueError(f"Unexpected batch format: {type(batch_data)}")
                
                # Handle GridSplit output: reshape from [batch_size, num_patches, C, H, W] to [batch_size*num_patches, C, H, W]
                if x.dim() == 5:  # [batch_size, num_patches, C, H, W]
                    batch_size, num_patches = x.shape[0], x.shape[1]
                    x = x.view(-1, x.shape[2], x.shape[3], x.shape[4])  # [batch_size*num_patches, C, H, W]
                    y = y.view(-1, *y.shape[2:])  # Flatten first two dimensions
                
                # Debug: Print tensor shapes for first batch
                if i == 0:
                    print(f"Input shape after reshape: {x.shape}")
                    print(f"Label shape after reshape: {y.shape}")
                    
            except Exception as e:
                print(f"Error processing training batch {i}: {e}")
                print(f"Batch structure: {batch_data}")
                raise
            
            optimizer.zero_grad()
            outputs = model(x)
            loss = loss_function(outputs, y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        logging.info("Epoch %d/%d, Train Loss: %.4f", epoch, epochs, epoch_loss / max(1, len(train_loader)))

        # Validation
        model.eval()
        all_tp, all_tn, all_fp, all_fn = 0, 0, 0, 0
        with torch.no_grad():
            for i, val_data in enumerate(val_loader):
                # Handle different batch formats (same logic as training)
                try:
                    if isinstance(val_data, dict):
                        x = val_data["image"].to(device).float()
                        y = val_data["label"].to(device).float()
                    elif isinstance(val_data, (list, tuple)) and len(val_data) >= 2:
                        # Handle list/tuple format
                        if isinstance(val_data[0], dict):
                            # List of dictionaries - extract and stack
                            images = [item["image"] for item in val_data]
                            labels = [item["label"] for item in val_data]
                            x = torch.stack(images, dim=0).to(device).float()
                            y = torch.stack(labels, dim=0).to(device).float()
                        else:
                            # Direct tensor format [images, labels]
                            x = val_data[0].to(device).float()
                            y = val_data[1].to(device).float()
                    else:
                        raise ValueError(f"Unexpected validation batch format: {type(val_data)}")
                    
                    # Handle GridSplit output: reshape from [batch_size, num_patches, C, H, W] to [batch_size*num_patches, C, H, W]
                    if x.dim() == 5:  # [batch_size, num_patches, C, H, W]
                        batch_size, num_patches = x.shape[0], x.shape[1]
                        x = x.view(-1, x.shape[2], x.shape[3], x.shape[4])  # [batch_size*num_patches, C, H, W]
                        y = y.view(-1, *y.shape[2:])  # Flatten first two dimensions
                        
                except Exception as e:
                    print(f"Error processing validation batch {i}: {e}")
                    print(f"Batch structure: {val_data}")
                    raise
                
                outputs = model(x)
                probs = torch.sigmoid(outputs)
                preds = (probs > 0.5).float()
                y_true = y
                tp = (preds * y_true).sum().item()
                tn = ((1 - preds) * (1 - y_true)).sum().item()
                fp = (preds * (1 - y_true)).sum().item()
                fn = ((1 - preds) * y_true).sum().item()
                all_tp += tp
                all_tn += tn
                all_fp += fp
                all_fn += fn

        accuracy = (all_tp + all_tn) / (all_tp + all_tn + all_fp + all_fn + 1e-7)
        precision = all_tp / (all_tp + all_fp + 1e-7) if (all_tp + all_fp) > 0 else 0.0
        recall = all_tp / (all_tp + all_fn + 1e-7) if (all_tp + all_fn) > 0 else 0.0
        dice = (2 * all_tp) / (2 * all_tp + all_fp + all_fn + 1e-7) if (2 * all_tp + all_fp + all_fn) > 0 else 0.0
        logging.info("Validation - Acc: %.4f, Prec: %.4f, Recall: %.4f, Dice: %.4f", accuracy, precision, recall, dice)


def main():
    parser = argparse.ArgumentParser(description="Train WSI Tile Classifier for Tumor Detection (ROCm-ready)")
    parser.add_argument("--image-root", type=str, default="sample_images/camelyon", help="Root folder for images")
    parser.add_argument("--train-csv", type=str, default="demo/data/pathology_tumor_detection/training.csv", help="Training CSV path")
    parser.add_argument("--valid-csv", type=str, default="demo/data/pathology_tumor_detection/validation.csv", help="Validation CSV path")
    parser.add_argument("--region-size", type=int, default=256 * 3, help="WSI patch region size")
    parser.add_argument("--grid-shape", type=int, default=3, help="Grid split dimension (e.g., 3x3)")
    parser.add_argument("--patch-size", type=int, default=224, help="Patch size for tile classifier input")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--epochs", type=int, default=2, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--num-workers", type=int, default=2, help="DataLoader workers")
    parser.add_argument("-v", "--verbose", action="count", default=1, help="Increase verbosity (-v, -vv)")
    args = parser.parse_args()

    setup_logging(args.verbose)
    device = select_device()

    train_loader, val_loader = build_dataloaders(
        image_root=args.image_root,
        train_csv=args.train_csv,
        valid_csv=args.valid_csv,
        grid_shape=args.grid_shape,
        region_size=args.region_size,
        patch_size=args.patch_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    model = build_model(device)
    train_and_validate(model, train_loader, val_loader, device, epochs=args.epochs, lr=args.lr)


if __name__ == "__main__":
    main()

