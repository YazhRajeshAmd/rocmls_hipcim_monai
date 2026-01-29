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
# Pathology Tumor Detection - CSV-driven Inference (ROCm-ready)
# This example performs inference over a CSV list of WSIs using simple tiling via PatchWSIDataset,
# loading a TorchVision-based MONAI model and reporting elapsed time. Intended as a minimal, scriptable variant.
#
# Usage:
#   python wsi_infer.py \
#     --root /path/to/dataset_root \
#     --csv-file demo/data/pathology_tumor_detection/validation.csv \
#     --model-path /path/to/model.pt \
#     --out-dir ./infer_out \
#     --region-size 768 --batch-size 4

import os
import torch
import logging
import argparse
import numpy as np
from monai.data import CSVDataset, DataLoader, PatchWSIDataset
from monai.networks.nets import TorchVisionFCModel
from monai.transforms import Compose, Lambdad, GridSplitd, ToTensord
from monai.utils import set_determinism
import time


def select_device(force_cpu: bool) -> torch.device:
    if not force_cpu and torch.cuda.is_available():
        logging.info("Using GPU device (ROCm/HIP backend). torch.version.hip=%s", getattr(torch.version, "hip", "unknown"))
        return torch.device("cuda")
    logging.info("Using CPU device")
    return torch.device("cpu")


def build_dataset(root: str, csv_file: str, region_size: int, grid: int, patch_size: int) -> PatchWSIDataset:
    datalist = CSVDataset(
        csv_file,
        col_groups={"image": 0, "location": [2, 1], "label": [3, 6, 9, 4, 7, 10, 5, 8, 11]},
        kwargs_read_csv={"header": None},
        transform=Lambdad("image", lambda x: os.path.join(root, "camelyon", x + ".tif")),
    )

    preprocess = Compose([
        Lambdad(keys="label", func=lambda x: x.reshape((1, grid, grid))),
        GridSplitd(
            keys=("image", "label"),
            grid=(grid, grid),
            size={"image": patch_size, "label": 1},
        ),
        ToTensord(keys="label"),
    ])

    dataset = PatchWSIDataset(
        data=datalist,
        patch_size=region_size,
        patch_level=0,
        transform=preprocess,
        reader="cuCIM",
    )
    return dataset


def main(cfg):
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    device = select_device(cfg.cpu)

    # Model
    model = TorchVisionFCModel("resnet18", num_classes=1, use_conv=True, pretrained=False).to(device)
    if not os.path.exists(cfg.model_path):
        raise FileNotFoundError(f"Checkpoint not found: {cfg.model_path}")
    state_dict = torch.load(cfg.model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    logging.info(f"Loaded checkpoint: {cfg.model_path}")

    # Dataset and loader
    dataset = build_dataset(cfg.root, cfg.csv_file, cfg.region_size, grid=3, patch_size=256)
    dataloader = DataLoader(dataset, batch_size=cfg.batch_size, num_workers=cfg.num_workers)

    os.makedirs(cfg.out_dir, exist_ok=True)

    start_time = time.time()
    with torch.no_grad():
        for batch in dataloader:
            x = batch["image"].to(device).float()
            _ = torch.sigmoid(model(x)).cpu().numpy()
    elapsed = time.time() - start_time
    logging.info("Inference completed in %.2f seconds over %d batches", elapsed, len(dataloader))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inference on pathology WSIs via CSV (ROCm-ready)")
    parser.add_argument("--root", type=str, required=True, help="Dataset root (expects camelyon/ subdir)")
    parser.add_argument("--csv-file", type=str, required=True, help="CSV with WSI entries")
    parser.add_argument("--model-path", type=str, required=True, help="Path to .pt checkpoint")
    parser.add_argument("--out-dir", type=str, default="./infer_out", help="Output folder")
    parser.add_argument("--region-size", type=int, default=768, help="WSI region size")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size")
    parser.add_argument("--num-workers", type=int, default=4, help="DataLoader workers")
    parser.add_argument("--cpu", action="store_true", help="Force CPU mode")
    cfg = parser.parse_args()
    set_determinism(seed=0)
    main(cfg)
