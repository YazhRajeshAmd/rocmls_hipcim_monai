# MONAI Spleen CT Segmentation (ROCm)

This folder contains AMD ROCm-ready examples for training and inference using 
the MONAI "spleen_ct_segmentation" bundle. 

It demonstrates:
- Training a 3D UNet on the MSD Task09_Spleen dataset
- Single-image inference with multiple preprocessing variants
- Batched inference with batch-size sweep on ROCm GPUs

## Prerequisites

- ROCm: 6.4
- Python: 3.10
- PyTorch (ROCm build): 2.x (torch.cuda should report available, HIP backend in `torch.version.hip`)
- MONAI: 1.x
- Optional: `nibabel` (for saving NIfTI outputs)

You can install typical dependencies as documented here:
https://rocm.docs.amd.com/projects/monai/en/latest/install/installation.html

## Data
The training script will download and extract the MSD Task09_Spleen dataset 
into: `./data/spleen_ct_seg/spleen_data/Task09_Spleen`

Structure:
```
Task09_Spleen/
  imagesTr/
  labelsTr/
  imagesTs/          # not used here
```

## Commands
Train (single GPU/CPU):
```
python train.py --data-root ./data/spleen_ct_seg/spleen_data --epochs 10 --batch-size 2 --save-checkpoint
```

### Inference (single image):
```
# Training-aligned preprocessing
python infer.py --input ./data/spleen_ct_seg/spleen_data/Task09_Spleen/imagesTr/spleen_25.nii.gz --save-output
```

### Batched inference (batch-size sweep):
```
python infer_batched.py --input-dir ./data/spleen_ct_seg/spleen_data/Task09_Spleen/imagesTr --verbose
```

The inference prints a CSV summary:
```
Device,Batch size,Total time (s),Avg. batch time (s),Avg. time per image (s),Throughput (img/s)
GPU,1,12.3456,12.3456,12.3456,0.08
GPU,2,7.6543,3.8272,3.8272,0.26
...
```

# Notes
- Device selection prefers ROCm GPU automatically; falls back to CPU if unavailable.
- For multi-GPU training, consider torchrun --nproc_per_node=N, add DistributedSampler and gradient synchronization.
- You can parameterize paths and hyperparameters further via config files or environment variables.

# License
SPDX-License-Identifier: MIT
Copyright (c) 2025 Advanced Micro Devices, Inc.
