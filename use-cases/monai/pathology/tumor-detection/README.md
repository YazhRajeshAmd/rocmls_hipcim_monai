# MONAI Pathology Tumor Detection (ROCm)

This folder contains AMD ROCm-ready examples for training and inference on whole slide 
images (WSIs) for tumor detection with MONAI and TorchVision backbones.

## Contents
- train.py: Training script for a tile classifier using PatchWSIDataset and CSV inputs.
- wsi_infer.py: Minimal CSV-driven inference script over WSIs.

## Prerequisites
- ROCm: 6.4
- Python: 3.10
- PyTorch (ROCm build): 2.x
- MONAI: 1.x
- hipCIM (for fast WSI I/O), OpenSlide (optional)
- Pillow, numpy, opencv-python

Refer to the documentation for installation steps:
https://rocm.docs.amd.com/projects/monai/en/latest/install/installation.html

Data layout Training and inference expect CSV files with columns mapping to:
- image name, location (y, x), and label entries per grid cell (e.g., 9 entries for a 3x3 grid).

## Commands

### Train (single GPU/CPU)
```
python train.py \
  --image-root ../../../../sample_images/camelyon \
  --train-csv ../../../../demo/data/pathology_tumor_detection/training.csv \
  --valid-csv ../../../../demo/data/pathology_tumor_detection/validation.csv \
  --epochs 2 --batch-size 32
```

### CSV-driven inference (iterates dataset entries in a CSV)
```
python wsi_infer.py \
  --root /path/to/dataset_root \
  --csv-file ../../../../demo/data/pathology_tumor_detection/validation.csv \
  --model-path /path/to/model.pt \
  --out-dir ./infer_out \
  --region-size 768 --batch-size 4
```

## Notes
- Device selection prefers ROCm GPU automatically (torch.cuda with HIP backend), otherwise falls back to CPU.
- Mixed precision (--fp16) can improve throughput on ROCm GPUs.
- Background filtering is a simple heuristic; tune thresholds for your data.
- For multi-GPU training, consider torchrun with DistributedSampler; the provided scripts target single-GPU/CPU runs.

# License 
SPDX-License-Identifier: MIT
Copyright (c) 2025 Advanced Micro Devices, Inc.
