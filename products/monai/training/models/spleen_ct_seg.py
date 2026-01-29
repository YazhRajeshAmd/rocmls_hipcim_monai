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

import os
import random
import tarfile
import time
import numpy as np
from sklearn.model_selection import train_test_split
from typing import List, Dict, Any, Optional, Tuple
from torch.optim.lr_scheduler import StepLR

import torch
import monai
from monai.apps import download_url
from monai.data import DataLoader, CacheDataset
from monai.losses import DiceLoss
from monai.metrics import DiceMetric
from monai.networks.nets import UNet
from monai.inferers import SlidingWindowInferer
from monai.transforms import (
    LoadImaged, EnsureChannelFirstd, Spacingd, Orientationd, ScaleIntensityRanged,
    RandCropByPosNegLabeld, RandFlipd, ToTensord
)

from components.diagnostics import warning, error, exception, info
from products.monai.console import TRAIN_CONSOLE_LOG_KEY

def get_bundle_components(params):
    """
    Returns preprocessing transforms, postprocessing transforms, and inferer 
    for MONAI bundle export.
    
    Args:
        params: Dict of hyperparameters/config for customization.
    Returns:
        pre_transforms: monai.transforms.Compose for preprocessing
        post_transforms: monai.transforms.Compose for postprocessing
        inferer: MONAI inferer (e.g., SlidingWindowInferer)
    """
    # Preprocessing = baseline pipeline as used in training (exc. random cropping/flipping)
    pre_transforms = monai.transforms.Compose([
        LoadImaged(keys=["image"]),
        EnsureChannelFirstd(keys=["image"]),
        Spacingd(keys=["image"], pixdim=(1.5, 1.5, 2.0), mode="bilinear"),
        Orientationd(keys=["image"], axcodes="RAS"),
        ScaleIntensityRanged(
            keys=["image"], a_min=-57, a_max=164, b_min=0.0, b_max=1.0, clip=True
        ),
        ToTensord(keys=["image"])
    ])

    # Postprocessing (could be expanded; e.g., argmax, thresholding)
    post_transforms = monai.transforms.Compose([
        # Example: Convert predictions to numpy or apply squeeze
        # (extend as needed, or leave as identity if no steps)
    ])

    # Inferer: use MONAI's sliding window as default for 3D volumes
    inferer = SlidingWindowInferer(
        roi_size=(96, 96, 96),
        sw_batch_size=1,
        overlap=0.25 # adjust for memory/performance needs
    )

    return pre_transforms, post_transforms, inferer

# ---- DATA PREPARATION ----

def prepare_training_data(params, val_frac: float = 0.2, random_seed: int = 42) -> Dict[str, DataLoader]:
    """
    Download, split, and return training and validation dataloaders.

    Args:
        params: dict, training parameters (e.g. batch_size)
        val_frac: float, fraction of files to use for validation.
        random_seed: int, for reproducibility.

    Returns:
        loaders: dict with keys 'train' and 'val' as DataLoaders.
    """
    # --- Download and extract data if needed ---
    resource = "https://msd-for-monai.s3-us-west-2.amazonaws.com/Task09_Spleen.tar"
    root_dir = "./data/spleen_ct_seg/spleen_data"
    data_dir = os.path.join(root_dir, "Task09_Spleen")
    compressed_file = os.path.join(root_dir, "Task09_Spleen.tar")
    if not os.path.exists(data_dir):
        if not os.path.exists(compressed_file):
            download_url(resource, compressed_file)
        with tarfile.open(compressed_file) as tar:
            tar.extractall(path=root_dir)

    # --- List and clean filepaths ---
    def is_nii(filename):
        return filename.endswith(".nii") or filename.endswith(".nii.gz")
    train_images = sorted([
        os.path.join(data_dir, "imagesTr", f)
        for f in os.listdir(os.path.join(data_dir, "imagesTr"))
        if is_nii(f) and not f.startswith("._")
    ])
    train_labels = sorted([
        os.path.join(data_dir, "labelsTr", f)
        for f in os.listdir(os.path.join(data_dir, "labelsTr"))
        if is_nii(f) and not f.startswith("._")
    ])
    files = [{"image": img, "label": seg} for img, seg in zip(train_images, train_labels)]

    # --- Split train/val using sklearn for reproducibility ---
    train_files, val_files = train_test_split(
        files, test_size=val_frac, random_state=random_seed
    )

    # --- Define transforms: same for both splits for this template ---
    transforms = monai.transforms.Compose([
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        Spacingd(keys=["image", "label"], pixdim=(1.5,1.5,2.0), mode=("bilinear", "nearest")),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        ScaleIntensityRanged(
            keys=["image"], a_min=-57, a_max=164, b_min=0.0, b_max=1.0, clip=True
        ),
        RandCropByPosNegLabeld(
            keys=["image", "label"], label_key="label",
            spatial_size=(96, 96, 96), pos=1, neg=1, num_samples=4, image_key="image", image_threshold=0,
        ),
        RandFlipd(keys=["image", "label"], prob=0.2, spatial_axis=0),
        RandFlipd(keys=["image", "label"], prob=0.2, spatial_axis=1),
        RandFlipd(keys=["image", "label"], prob=0.2, spatial_axis=2),
        ToTensord(keys=["image", "label"])
    ])

    # --- Build DataLoaders ---
    batch_size = params.get("batch_size", 2)
    train_ds = CacheDataset(train_files, transform=transforms, cache_rate=1.0, num_workers=2)
    val_ds = CacheDataset(val_files, transform=transforms, cache_rate=1.0, num_workers=2)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    return {"train": train_loader, "val": val_loader}

# ---- RANDOM VISUALIZATION SAMPLE EXTRACTION ----

def extract_visualization_samples_from_loader(loader, n=10, with_label=True):
    """
    Extract n random samples from the loader (using its batches).
    Each call returns randomized images.
    """
    # Collect all image samples from DataLoader
    all_samples = []
    for batch in loader:
        img = batch["image"][0].cpu().numpy()
        ch, d, h, w = img.shape
        mid = d // 2
        sample = {"input": img[0, mid, :, :]}
        if with_label and "label" in batch:
            label = batch["label"][0].cpu().numpy()
            sample["label"] = label[0, mid, :, :]
        all_samples.append(sample)
        # Optionally break if you know there can't be more than a few batches for efficiency
        if len(all_samples) >= max(n * 2, n + 5):
            break
    # Randomly sample n items out of all_samples, or as many as available
    if not all_samples:
        return []
    n_actual = min(len(all_samples), n)
    rand_indices = random.sample(range(len(all_samples)), n_actual)
    return [all_samples[i] for i in rand_indices]

# ---- NETWORK / OPTIMIZER / LOSS ----

def define_training_network(params, device):
    net = UNet(
        spatial_dims=3,
        in_channels=1,
        out_channels=2,
        channels=(16, 32, 64, 128, 256),
        strides=(2, 2, 2, 2),
        num_res_units=2,
        norm=monai.networks.layers.Norm.BATCH,
    ).to(device)
    return net

def define_loss_function(params):
    return DiceLoss(to_onehot_y=True, softmax=True)

def define_training_optimizer(params, model_parameters):
    lr = params.get('learning_rate', 1e-4)
    return torch.optim.Adam(model_parameters, lr=lr)

def define_training_metric(params):
    return DiceMetric(include_background=False, reduction="mean", ignore_empty=True)

def define_training_scheduler(params, optimizer):
    step_size = params.get('step_size', 10)
    gamma = params.get('gamma', 0.1)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=step_size, 
            gamma=gamma
    )
    return lr_scheduler

# ---- BATCH TRAIN STEP ----
def train_batch(
    network, batch_data, loss_fn, optimizer, metric_fn, device, params, epoch
):
    """
    Single training step. Updates and returns:
      - loss (float)
      - mean Dice (float, all classes)
      - per-class Dice (np.ndarray or list)
      - accuracy (float)
      - precision (float)
      - recall (float)
    """
    network.train()

    # Standard forward-backward
    inputs, labels = batch_data["image"].to(device), batch_data["label"].to(device)
    optimizer.zero_grad()
    outputs = network(inputs)
    loss = loss_fn(outputs, labels)
    loss.backward()
    optimizer.step()

    with torch.no_grad():
        # Get predicted classes
        preds = torch.argmax(outputs, dim=1, keepdim=True)
        
        # Mean Dice (all classes except background if desired)
        dice = metric_fn(y_pred=preds, y=labels)
        dice_np = dice.cpu().numpy().flatten()
        dice_no_nan = dice_np[~np.isnan(dice_np)]
        mean_dice = dice_no_nan.mean() if dice_no_nan.size > 0 else None

        # Per-class Dice (using DiceMetric with reduction='none')
        n_classes = outputs.shape[1]
        per_class_metric_fn = DiceMetric(include_background=True, reduction="none")
        per_class_dice_tensor = per_class_metric_fn(y_pred=preds, y=labels)
        # Shape: [n_classes]; convert to list or np.ndarray
        per_class_dice = per_class_dice_tensor.cpu().numpy().tolist() if per_class_dice_tensor is not None else None

        # Pixel-wise accuracy, precision, recall
        pred_binary = preds.cpu().numpy().astype(bool)
        label_binary = labels.cpu().numpy().astype(bool)
        # For batch 0, foreground only
        pred_bin = pred_binary[0, 0]
        label_bin = label_binary[0, 0]
        tp = np.logical_and(pred_bin, label_bin).sum()
        tn = np.logical_and(~pred_bin, ~label_bin).sum()
        fp = np.logical_and(pred_bin, ~label_bin).sum()
        fn = np.logical_and(~pred_bin, label_bin).sum()
        accuracy = (tp + tn) / (tp + tn + fp + fn + 1e-7)
        precision = tp / (tp + fp + 1e-7) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn + 1e-7) if (tp + fn) > 0 else 0

    return (
        loss.item(),
        mean_dice,
        per_class_dice,
        accuracy,
        precision,
        recall
    )

# ---- EVAL STEP WITH VISUALIZATION ----

def evaluate_batch(
    network, batch_data, loss_fn, metric_fn, device, params, epoch
):
    network.eval()
    inputs, labels = batch_data["image"].to(device), batch_data["label"].to(device)
    with torch.no_grad():
        outputs = network(inputs)
        loss = loss_fn(outputs, labels).item()
        preds = torch.argmax(outputs, dim=1, keepdim=True)
        dice = metric_fn(y_pred=preds, y=labels)
        dice_np = dice.cpu().numpy().flatten()
        dice_no_nan = dice_np[~np.isnan(dice_np)]
        val_dice = dice_no_nan.mean() if dice_no_nan.size > 0 else None

        n_classes = outputs.shape[1]
        per_class_metric_fn = DiceMetric(include_background=True, reduction="none")
        per_class_dice_tensor = per_class_metric_fn(y_pred=preds, y=labels)
        per_class_dice = per_class_dice_tensor.cpu().numpy().tolist() if per_class_dice_tensor is not None else None

        # Accuracy, precision, recall
        pred_binary = preds.cpu().numpy().astype(bool)
        label_binary = labels.cpu().numpy().astype(bool)
        pred_bin = pred_binary[0, 0]
        label_bin = label_binary[0, 0]
        tp = np.logical_and(pred_bin, label_bin).sum()
        tn = np.logical_and(~pred_bin, ~label_bin).sum()
        fp = np.logical_and(pred_bin, ~label_bin).sum()
        fn = np.logical_and(~pred_bin, label_bin).sum()
        accuracy = (tp + tn) / (tp + tn + fp + fn + 1e-7)
        precision = tp / (tp + fp + 1e-7) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn + 1e-7) if (tp + fn) > 0 else 0

        # Visualization (single mid-slice)
        img = inputs[0, 0].cpu().numpy()
        mid = img.shape[0] // 2
        pred_np = preds[0, 0].cpu().numpy()
        label_np = labels[0, 0].cpu().numpy()
        viz = [{
            "input": img[mid, :, :],
            "prediction": pred_np[mid, :, :],
            "label": label_np[mid, :, :],
        }]
    return dice, loss, val_dice, per_class_dice, accuracy, precision, recall, viz

def save_trained_model(network, params, model_dir="trained_models", model_name=None):
    """
    Saves the trained PyTorch model to disk.
    - network: torch.nn.Module (model to save)
    - params: dict, include anything relevant for file naming (model/epoch etc.)
    - model_dir: Path to save models
    - model_name: Optional filename (default uses model id and timestamp)
    Returns the model path.
    """
    os.makedirs(model_dir, exist_ok=True)
    if model_name is None:
        model_id = params.get("model_id", "ct_spleen_unet")
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        model_name = f"{model_id}_{timestamp}_final.pt"
    model_path = os.path.join(model_dir, model_name)

    # Save full state_dict for recommended model reproducibility
    torch.save(network.state_dict(), model_path)
    return model_path

