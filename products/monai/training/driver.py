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

import streamlit as st
import time
import importlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Callable, Dict, Any
from typing import Sequence

from components.state import session_state_set, session_state_get
from components.diagnostics import warning, error, exception, info
from components.io import render_markdown

from products.monai.training.metadata import display_metadata
from products.monai.utils import save_monai_bundle
from products.monai.console import TRAIN_CONSOLE_LOG_KEY
from products.monai.styles import (
        BEST_METRIC_HTML,
        TRAINING_RANDOM_SAMPLE_HEADER_HTML,
        TRAINING_STATS_HEADER_HTML,
        TRAINING_DEVICE_ID_HTML,
        TRAINING_SAMPLE_MISSING_HTML,
)

# Number of visualization columns per row
N_VIZ_COLUMNS = 5  

# Train button label
TRAIN_BTN_LBL = """
&nbsp;     
&nbsp;     
&nbsp;
Train Model >>
&nbsp;     
&nbsp;     
&nbsp;
"""

# Define default values for the training stats
default_stats = {
    "phase": "waiting",
    "msg": "training not started yet",
    "progress": 0.0,
    "epoch": 0,
    "total_epochs": 0,
    "avg_loss": 0,
    "mean_dice": None,
    "train_losses": [],
    "val_losses": [],
    "train_dices": [],
    "val_dices": [],
    "lrs": [],
    "visualization": [],
    "train_thumbnails": [],
    "val_thumbnails": [],
    "per_class_metrics": [],    # Optional: shape [epochs, n_classes]
    "accuracy": [],             # Optional
    "precision": [],            # Optional
    "recall": [],               # Optional
    "epoch_times": [],
    "train_batch_times": [],
    "val_batch_times": [],
    "batch_times": [],
    "batch_phases": [],
}

# Update the training stats with provided values and callback, if specified
def progress_callback_updated_stats(progress_callback: Callable = None, reset: bool = False, **kwargs):
    # Track whether stats was just initialized
    just_initialized = False

    # Initialize stats if None
    if reset or ("training_stats" not in st.session_state):
        # Initialize with defaults
        stats = dict(default_stats)

        # If max_epochs is provided, set total_epochs accordingly
        if "max_epochs" in kwargs:
            stats["total_epochs"] = kwargs["max_epochs"]
        
        # Update stats further with other provided key-values
        for key, value in kwargs.items():
            stats[key] = value
        
        # Don't callback
        just_initialized = True
    else:
        # Fetch the latest training data set 
        stats = session_state_get("training_stats")

        # Add any missing keys with default values
        for key in default_stats:
            if key not in stats:
                stats[key] = default_stats[key]

    # Update keys with provided kwargs
    for key, value in kwargs.items():
        if key in stats:
            stats[key] = value

        # If key doesn't exist in stats, insert with value and not default
        elif key in default_stats:
            stats[key] = value

        # Unknown key provided; must be added to defaults
        else:
            error(TRAIN_CONSOLE_LOG_KEY,
                  f"unknown key \"{key}\" provided in training stats")

    # Squirrel away the stats into session state
    session_state_set("training_stats", stats)

    # If callback is defined, call it with the updated stats
    if progress_callback and not just_initialized:
        progress_callback()

def compute_progress_nested(level_counts: Sequence[int], indices: Sequence[int]) -> float:
    """
    Compute normalized progress (0.0 to 1.0) for nested steps/substeps.

    Args:
        level_counts (Sequence[int]): total count at each level
                                      e.g. [total_steps, subcount, subsubcount, ...]
        indices (Sequence[int]): current index at each level (1-based, same length as level_counts)

    Returns:
        float: normalized progress value in [0.0, 1.0].

    Examples:
        # Example 1: Just 8 steps (like before)
        print(compute_progress_nested([8], [1]))  # 0.125
        print(compute_progress_nested([8], [8]))  # 1.0
        
        # Example 2: Step 3 of 8, with 5 sub-steps
        print(compute_progress_nested([8, 5], [3, 1]))  # 0.25
        print(compute_progress_nested([8, 5], [3, 5]))  # 0.35
        
        # Example 3: Step 3 of 8, substep 2 of 5, sub-substep 4 of 10
        print(compute_progress_nested([8, 5, 10], [3, 2, 4]))
        # -> progress somewhere between 0.26 and 0.28
    """
    if len(level_counts) != len(indices):
        raise ValueError("level_counts and indices must have the same length")

    progress = 0.0
    span = 1.0  # total fraction available at current level

    for count, idx in zip(level_counts, indices):
        if count <= 0:
            raise ValueError("All counts must be positive")
        if not (1 <= idx <= count):
            raise ValueError(f"Index {idx} must be in [1, {count}]")

        step_fraction = span / count
        progress += (idx / count) * span
        span = step_fraction  # shrink span for deeper levels

    return progress

def run_dynamic_monai_training(model_impl: str, 
                               device,
                               params: dict, 
                               max_epochs: int, 
                               progress_callback: Callable = None):
    """
    Orchestrates the training pipeline for a given MONAI Model Zoo model.
    
    Args:
        model_impl: str - Full module import path of the model training implementation.
        params: dict - Hyperparameters and config options.
        max_epochs: int - Number of training epochs.
        progress_callback: Callable - Optional function for visualization/UI streaming.

    Returns:
        dict containing losses, metrics, and any visualization hooks.
    """
    # Dynamically import the selected model training interface
    model_module = importlib.import_module(model_impl)

    # ---- High-level step hooks (with full extensibility for future models) ----

    # Note the training start time
    training_start_time = time.time()

    # Fetch the preprocessing transforms, postprocessing transforms, and the 
    # inferer (e.g., sliding window inferer)
    pre_transforms, post_transforms, inferer = model_module.get_bundle_components(params)

    # ---------- Step 1: Prepare training data ----------
    progress_callback_updated_stats(
                progress_callback=progress_callback,
                phase="data prep",
                msg="Step 1/8: Preparing training and validation data...",
                progress=compute_progress_nested([8], [1]),
            )

    # Data Loaders
    # Should return dict: {'train': train_loader, 'val': val_loader}
    loaders = model_module.prepare_training_data(params)         
    train_loader = loaders.get('train')
    val_loader = loaders.get('val')

    # ---------- Step 2: Extract thumbnails for thumbnails (before transforms, delays!) ----------
    progress_callback_updated_stats(
                progress_callback=progress_callback,
                phase="thumbnail_extract",
                msg="Step 2/8: Extracting random visualization samples...",
                progress=compute_progress_nested([8], [2]),
            )

    # Visualization
    samples_train = model_module.extract_visualization_samples_from_loader(train_loader, n=10)
    samples_val = model_module.extract_visualization_samples_from_loader(val_loader, n=10)

    # ---------- Step 3: Thumbnail visualizations ----------
    progress_callback_updated_stats(
                progress_callback=progress_callback,
                phase="thumbnail visualization",
                msg="Step 3/8: Visualizing training/validation thumbnails...",
                progress=compute_progress_nested([8], [3]),
                train_thumbnails=samples_train,
                val_thumbnails=samples_val,
            )

    # Transforms/Pipeline
    # You can optionally store/return transforms for advanced hooks/visualization
    
    # ---------- Step 4: Model/optimizer/etc. ----------
    progress_callback_updated_stats(
                progress_callback=progress_callback,
                phase="model_setup",
                msg="Step 4/8: Initializing network, loss, optimizer...",
                progress=compute_progress_nested([8], [4]),
            )

    # Device, Model, Loss, Optimizer, Metric
    network = model_module.define_training_network(params, device)
    loss_fn = model_module.define_loss_function(params)
    optimizer = model_module.define_training_optimizer(params, network.parameters())
    metric_fn = model_module.define_training_metric(params)

    # Optional: learning rate scheduler
    scheduler = getattr(model_module, 'define_training_scheduler', None)
    if scheduler:
        lr_scheduler = scheduler(params, optimizer)
    else:
        lr_scheduler = None
        warning(TRAIN_CONSOLE_LOG_KEY, 
                f"no learning rate scheduler defined in {model_impl}")

    # ---------- Step 5: Training loop epoch/batch ----------
    train_losses, train_dices = [], []
    val_losses, val_dices = [], []
    accuracies, precisions, lrs = [], [], []
    per_class_metrics = []
    recalls = []
    epoch_times = []
    batch_times = []
    batch_phases = []
    train_batch_times = []
    val_batch_times = []

    for epoch in range(1, max_epochs+1):
        epoch_start = time.time()
        network.train()
        epoch_loss, epoch_metric, step = 0, 0, 0
        # Batch-wise train stats (if computing train Dice per batch)
        batch_train_dices = []

        # --- Training Loop ---
        # --- Per-batch UI update
        for batch_i, batch_data in enumerate(train_loader, 1):
            train_batch_start = time.time()
            step += 1
            batch_loss, batch_dice, batch_metrics, batch_acc, batch_prec, batch_recall = \
                model_module.train_batch(
                    network=network,
                    batch_data=batch_data,
                    loss_fn=loss_fn,
                    optimizer=optimizer,
                    metric_fn=metric_fn,   # <-- accept dice or other metric
                    device=device,
                    params=params,
                    epoch=epoch,
                )
            epoch_loss += batch_loss
            if batch_dice is not None:
                batch_train_dices.append(batch_dice)
            if batch_metrics is not None:
                per_class_metrics.append(batch_metrics)  # Optionally append/average
            if batch_acc is not None:
                accuracies.append(batch_acc)
            if batch_prec is not None:
                precisions.append(batch_prec)
            if batch_recall is not None:
                recalls.append(batch_recall)
            batch_times.append(time.time() - train_batch_start)
            batch_phases.append("Training")
            
            # Update random visualization samples
            samples_train = model_module.extract_visualization_samples_from_loader(train_loader, n=10)
            samples_val = model_module.extract_visualization_samples_from_loader(val_loader, n=10)

            # Fine-grained progress update per training batch
            progress_callback_updated_stats(
                progress_callback=progress_callback,
                phase="train_batch",
                msg=f"Training epoch {epoch}/{max_epochs}  Batch {batch_i}",
                epoch=epoch, 
                batch_times=batch_times,
                batch_phases=batch_phases,
                train_losses=train_losses + [epoch_loss/step],
                train_dices=train_dices + [np.mean(batch_train_dices)],
                lrs=lrs,
                per_class_metrics=per_class_metrics,
                accuracy=accuracies,
                precision=precisions,
                recall=recalls,
                progress=compute_progress_nested(
                    [8, max_epochs+1],  # (max_epochs+1) * len(train_loader)], 
                    [5, epoch]          #,        epoch * batch_i]
                ),
                train_thumbnails=samples_train,
                val_thumbnails=samples_val,
            )

        avg_loss = epoch_loss / step if step > 0 else float('nan')
        train_losses.append(avg_loss)
        if batch_train_dices:
            train_dices.append(np.mean(batch_train_dices))

        # Update learning rate variations if LR scheduler defined
        if lr_scheduler:
            lrs.append(lr_scheduler.get_last_lr()[0])
        else:
            lrs.append(optimizer.param_groups[0]['lr'])

        # ---- SAVE MODEL CHECKPOINT HERE ----
        # model_module.save_trained_model(network, epoch, params)

        # --- Step 6: Per-epoch Validation Loop ---
        val_epoch_loss = 0
        val_metrics, val_epoch_dice = [], []
        val_epoch_acc, val_epoch_prec, val_epoch_recall = [], [], []
        network.eval()
        if val_loader is not None:
            for val_batch in val_loader:
                val_batch_start = time.time()
                val_metric_val, val_loss, val_dice, cls_metrics, val_acc, \
                    val_prec, val_recall, viz = model_module.evaluate_batch(
                        network=network,
                        batch_data=val_batch,
                        loss_fn=loss_fn,
                        metric_fn=metric_fn,
                        device=device,
                        params=params,
                        epoch=epoch,
                    )
                val_epoch_loss += val_loss
                if val_dice is not None:
                    val_epoch_dice.append(val_dice)
                if cls_metrics is not None:
                    per_class_metrics.append(cls_metrics)
                batch_times.append(time.time() - val_batch_start)
                batch_phases.append("Validation")
        
                # ---- Fix: aggregate metrics if dict, else append as-is ----
                def aggregate_metric(val):
                    if isinstance(val, dict):
                        vals = [v for v in val.values() if v is not None]
                        return np.mean(vals) if vals else None
                    return val
        
                if val_acc is not None:
                    val_epoch_acc.append(aggregate_metric(val_acc))
                if val_prec is not None:
                    val_epoch_prec.append(aggregate_metric(val_prec))
                if val_recall is not None:
                    val_epoch_recall.append(aggregate_metric(val_recall))
        
                # Update random visualization samples
                samples_train = model_module.extract_visualization_samples_from_loader(train_loader, n=10)
                samples_val = model_module.extract_visualization_samples_from_loader(val_loader, n=10)

                # Fine-grained progress update per validation batch
                progress_callback_updated_stats(
                    progress_callback=progress_callback,
                    phase="validation",
                    msg=f"Step 6/8: Validating after epoch {epoch}...",
                    batch_times=batch_times,
                    batch_phases=batch_phases,
                    progress=compute_progress_nested(
                        [8, max_epochs+1], [5, epoch]
                    ),
                    train_thumbnails=samples_train,
                    val_thumbnails=samples_val,
                    visualization=viz,
                )
        else:
            warning(TRAIN_CONSOLE_LOG_KEY, 
                    "No validation loader provided by the selected model. Skipping validation.")
        
        mean_val_loss = val_epoch_loss / max(1, len(val_loader))
        val_losses.append(mean_val_loss)

        def flatten_to_mean(val):
            """
            If val is a dict (e.g., per-class metric), return the mean of its numeric values.
            If scalar, return as is. Otherwise, return np.nan.
            """
            if isinstance(val, dict):
                numeric_vals = [x for x in val.values() if isinstance(x, (float, int, np.floating, np.integer))]
                return np.mean(numeric_vals) if numeric_vals else np.nan
            elif isinstance(val, (float, int, np.floating, np.integer)):
                return val
            return np.nan
        
        if val_epoch_dice:
            dices_flat = [flatten_to_mean(v) for v in val_epoch_dice if v is not None]
            val_dices.append(np.mean([v for v in dices_flat if not isinstance(v, dict)] if dices_flat else [np.nan]))
        if val_epoch_acc:
            acc_flat = [flatten_to_mean(v) for v in val_epoch_acc if v is not None]
            accuracies.append(np.mean([v for v in acc_flat if not isinstance(v, dict)] if acc_flat else [np.nan]))
        if val_epoch_prec:
            prec_flat = [flatten_to_mean(v) for v in val_epoch_prec if v is not None]
            precisions.append(np.mean([v for v in prec_flat if not isinstance(v, dict)] if prec_flat else [np.nan]))
        if val_epoch_recall:
            recall_flat = [flatten_to_mean(v) for v in val_epoch_recall if v is not None]
            recalls.append(np.mean([v for v in recall_flat if not isinstance(v, dict)] if recall_flat else [np.nan]))
        epoch_times.append(time.time() - epoch_start)

        # --- Step 7: After epoch: Update with full context (loss/metrics/images)
        progress_callback_updated_stats(
            progress_callback=progress_callback,
            phase="epoch_end",
            msg=f"Completed epoch {epoch}/{max_epochs}",
            epoch=epoch, 
            total_epochs=max_epochs,
            avg_loss=avg_loss,
            mean_dice=np.mean(batch_train_dices) if batch_train_dices else None,
            train_losses=train_losses.copy(),
            val_losses=val_losses.copy(),
            train_dices=train_dices.copy(),
            val_dices=val_dices.copy(),
            lrs=lrs.copy(),
            per_class_metrics=per_class_metrics.copy(),
            accuracy=accuracies.copy(),
            precision=precisions.copy(),
            recall=recalls.copy(),
            epoch_times=epoch_times.copy(),
            train_thumbnails=samples_train,
            val_thumbnails=samples_val,
            progress=compute_progress_nested(
                [8, max_epochs+1],
                [7, epoch,      ],
            ),
        )

        # Step LR, if learning rate scheduler defined
        if lr_scheduler:
            lr_scheduler.step()
    
    # ---------- Step 8: Done! ----------
    training_duration = f"(took {(time.time() - training_start_time):.2f}s)"
    progress_callback_updated_stats(
                progress_callback=progress_callback,
                phase="done",
                msg=f"Step 8/8: Training complete {training_duration}!",
                progress=compute_progress_nested([8], [8]),
                train_losses=train_losses,
                val_dices=val_dices,
                lrs=lrs,
                train_thumbnails=samples_train,
                val_thumbnails=samples_val,
            )

    # Save the trained model as a bundle
    model_key = session_state_get('monai_model_info')['id']
    base_metadata = {
                        "id": model_key,
                        "version": "0.1.0",
                        # ... other metadata fields
                    }
    bundle_dir = save_monai_bundle(
                    network=network,
                    pre_transforms=pre_transforms,
                    post_transforms=post_transforms,
                    inferer=inferer,
                    metadata_base=base_metadata,
    )
    info(TRAIN_CONSOLE_LOG_KEY, 
         f"Trained model saved in bundle format at {bundle_dir}")

def monai_training_driver():
    # ----- Training Button -----
    training_header_placeholder = st.container()
    with training_header_placeholder:
        training_header_row = st.columns([1, 7])

    # Layout the training button and model info
    train_btn_placeholder = training_header_row[0].empty()
    run_btn = train_btn_placeholder.button(TRAIN_BTN_LBL, 
                                           disabled=False, 
                                           key="train_model_btn")
    with training_header_row[1]:
        train_info_placeholder = training_header_row[1].empty()
        # ---- PROGRESS BAR ----
        progress_row = st.empty()
    
    with train_info_placeholder:
        display_metadata()
    progress_indicator = progress_row.progress(0, text="Waiting to initiate model training...")

    # Thumbnail groups across training and validation samples (random)
    def render_group(col, group_name, key_prefix):
        with col:
            # Title bar
            st.markdown(
                TRAINING_RANDOM_SAMPLE_HEADER_HTML.format(
                    group_name=group_name
                ),
                unsafe_allow_html=True
            )
    
            # Create a Streamlit container that visually sits inside the parent box
            inner = st.container(border=True)
            with inner:
                row1 = st.columns(N_VIZ_COLUMNS)
                row1_placeholders = [c.empty() for c in row1]
                row2 = st.columns(N_VIZ_COLUMNS)
                row2_placeholders = [c.empty() for c in row2]
    
            # Close parent box
            st.markdown("</div>", unsafe_allow_html=True)
    
        return row1_placeholders, row2_placeholders
    
    # Split view across random samples and training stats visualization
    sample_viewport, training_viewport = st.columns([2, 1])

    # Layout the training viewport
    training_container = training_viewport.container()
    with training_container:
        training_tabs = st.tabs(["**Prediction**", 
                                 "**Recall**", 
                                 "**Learning Rate**",
                                 "**Timing**",
                                ],
                               )
        tab_placeholder_1 = training_tabs[0].empty()
        tab_placeholder_2 = training_tabs[1].empty()
        tab_placeholder_3 = training_tabs[2].empty()
        tab_placeholder_4 = training_tabs[3].empty()

    # Live" per-epoch batch inference result visuals
    with tab_placeholder_1:
        viz_group = st.container()
        with viz_group:
            viz_intro = st.empty()
            row1 = st.columns(N_VIZ_COLUMNS)
            viz_placeholder_1 = [c.empty() for c in row1]
            row2 = st.columns(N_VIZ_COLUMNS)
            viz_placeholder_2 = [c.empty() for c in row2]
    
    # Recall metric
    with tab_placeholder_2:
        with st.container():
            st.caption("Recall is the fraction of actual positives that are correctly identified by the model.")
            recall_curve = st.empty()

    # Learning Rate
    with tab_placeholder_3:
        with st.container():
            st.caption("Learning Rate is the step size controlling how much model weights are updated during training.")
            lr_curve = st.empty()

    # Timing
    with tab_placeholder_4:
        with st.container():
            st.caption("Breakup of time taken for training and validation per batch.")
            timing_curve = st.empty()

    # Mark validation pending for clarification
    with viz_intro:
        st.caption("Prediction is the model's output label or value for each input, representing its best guess based on learned patterns.")

    # Layout the random samples viewport
    with sample_viewport:
        # Add a neat display to identify the device being used
        device_container = st.container(border=True)
        device_identification = device_container.empty()
        device_id = session_state_get('training_device')
        device_type = session_state_get('training_device_type')
        device_identification.markdown(
                TRAINING_DEVICE_ID_HTML.format(
                    device=device_id,
                    device_type=device_type,
                ),
                unsafe_allow_html=True
        )

        # Layout with 2 columns (Train, Validate)
        random_sample_explanation = st.empty()
        with random_sample_explanation:
            st.caption("Random samples from training & validation dataset...")
        train_col, val_col = st.columns(2)
    train_inputs, train_labels = render_group(train_col, 
                                              "Training Dataset (random samples)", "train")
    val_inputs, val_labels     = render_group(val_col, 
                                              "Validation Dataset (random samples)", "val")

    # Layout the training stats visualization
    stats_viewport = st.container()
    with stats_viewport:
        # Title bar
        st.markdown(
            TRAINING_STATS_HEADER_HTML.format(
                group_name="Training & Validation Metrics"
            ),
            unsafe_allow_html=True
        )
        render_markdown("markdown/monai_training_metrics.md")
        st.markdown("</div>", unsafe_allow_html=True)
        graph_containers = st.columns(4)
        loss_curve = graph_containers[0].empty()
        dice_curve = graph_containers[1].empty()
        metrics_curve = graph_containers[2].empty()

        # Create a special box for displaying the best accuracy
        with graph_containers[3].container(border=True):
            with st.container(border=True):
                best_metric = st.empty()

    def overlay_contour(img, mask, color='red', linewidth=2):
        import matplotlib.pyplot as plt
        import numpy as np
    
        # If image is (C, H, W), convert to (H, W, C)
        if img.ndim == 3 and img.shape[0] in [1, 3]:
            img = np.transpose(img, (1, 2, 0))

        fig, ax = plt.subplots(figsize=(2,2), dpi=80)
        ax.imshow(img, cmap='gray' if img.shape[-1] == 1 else None)

        # Ensure mask is 2D
        if mask.ndim == 3:
            if mask.shape[0] == 1:
                mask = mask[0]
            else:
                mask = mask[..., 0]

        # Only draw contour if mask is at least 2x2
        if mask.ndim == 2 and (mask > 0).any() and mask.shape[0] >= 2 and mask.shape[1] >= 2:
            ax.contour(mask, colors=[color], linewidths=linewidth)
        ax.axis('off')
        fig.tight_layout(pad=0)
    
        # Use buffer_rgba, convert RGBA to RGB
        fig.canvas.draw()
        buf = np.asarray(fig.canvas.buffer_rgba())
        img_array = buf[..., :3]  # Discard alpha channel to get RGB
        plt.close(fig)
        return img_array
    
    def display_epochwise_viz(
            epoch,
            viz_data, 
            viz_placeholder_1, 
            viz_placeholder_2,
            per_epoch=N_VIZ_COLUMNS):
        """
        Display one overlay for label (row 1) and one for prediction (row 2), 
        for each epoch (column) with wraparound.
        viz_data: list of dicts (must include 'input', 'label', 'prediction')
        viz_placeholder_X: list of st.empty() placeholders, len = per_epoch
        """
        for sample in viz_data:
            # Wraparound column calculation
            col = (epoch + per_epoch -1) % per_epoch

            # Overlay label on input
            if all(k in sample for k in ("input", "label")):
                lbl_overlay = overlay_contour(sample["input"], sample["label"], color='lime')
                viz_placeholder_1[col].image(lbl_overlay, 
                                             caption=f"Epoch {epoch}: Ground Truth", 
                                             width="stretch")
            else:
                viz_placeholder_1[col].markdown("<div style='background:#eee;height:80px;'></div>", 
                                                unsafe_allow_html=True)

            # Overlay prediction on input
            if all(k in sample for k in ("input", "prediction")):
                pred_overlay = overlay_contour(sample["input"], sample["prediction"], color='magenta')
                viz_placeholder_2[col].image(pred_overlay, 
                                             caption=f"Epoch {epoch}: Prediction", 
                                             width="stretch")
            else:
                viz_placeholder_2[col].markdown("<div style='background:#eee;height:80px;'></div>", 
                                                unsafe_allow_html=True)

    def compose_callback_for_training():
        def _callback():
            # Fetch the latest training data set 
            stats = session_state_get("training_stats")

            # Update progress indicator
            progress_indicator.progress(stats.get("progress", 0.0), text=stats.get("msg", "..."))
            
            # Prominent device identification
            device_id = session_state_get('training_device')
            device_type = session_state_get('training_device_type')
            device_identification.markdown(
                    TRAINING_DEVICE_ID_HTML.format(
                        device=device_id,
                        device_type=device_type,
                    ),
                    unsafe_allow_html=True
            )

            model_info = session_state_get('monai_model_info')
            model_id = model_info['id']
            with random_sample_explanation:
                explanation = f"markdown/monai_training_random_samples_{model_id}.md"
                render_markdown(explanation)

            # Display visualization samples
            viz = stats.get("visualization", [])
            if viz and isinstance(viz, list):
                with viz_intro:
                    render_markdown("markdown/monai_viz_panel.md")
                epoch = stats.get("epoch", 0)
                display_epochwise_viz(epoch, viz, viz_placeholder_1, viz_placeholder_2)

            # Display compact tile layout in separate "input" and "label" rows for train and val sets
            samples_train = stats.get("train_thumbnails", [])
            samples_val = stats.get("val_thumbnails", [])
            n_display = min(5, len(samples_train), 
                               len(samples_val), 
                               len(train_inputs), 
                               len(train_labels), 
                               len(val_inputs), 
                               len(val_labels))
            
            # Fill placeholders for training samples
            for idx in range(n_display):
                sample = samples_train[idx]

                # Input
                in_img = sample.get("input")
                if in_img is not None:
                    train_inputs[idx].image(in_img, 
                                            caption=f"Input #{idx+1}",
                                            width="stretch")
                else:
                    train_inputs[idx].markdown(TRAINING_SAMPLE_MISSING_HTML, unsafe_allow_html=True)

                # Label (mask)
                label_img = sample.get("label")
                if label_img is not None:
                    train_labels[idx].image(label_img, 
                                            caption=f"Label #{idx+1}", 
                                            width="stretch")
                else:
                    train_labels[idx].markdown(
                            TRAINING_SAMPLE_MISSING_HTML, 
                            unsafe_allow_html=True
                    )
            
            # Fill placeholders for validation samples
            for idx in range(n_display):
                sample = samples_val[idx]

                # Input
                in_img = sample.get("input")
                if in_img is not None:
                    val_inputs[idx].image(in_img, 
                                          caption=f"Input #{idx+1}", 
                                          width="stretch")
                else:
                    val_inputs[idx].markdown(TRAINING_SAMPLE_MISSING_HTML, unsafe_allow_html=True)

                # Label (mask)
                label_img = sample.get("label")
                if label_img is not None:
                    val_labels[idx].image(label_img, 
                                          caption=f"Label #{idx+1}", 
                                          width="stretch")
                else:
                    val_labels[idx].markdown(TRAINING_SAMPLE_MISSING_HTML, unsafe_allow_html=True)
                
            # Display the metrics as grouped graphs
            def pad_columns_to_max_length(columns: dict) -> pd.DataFrame:
                """
                Given a dict of {colname: list of values}, pad all lists with NaN so they are equal length,
                then construct a DataFrame.
                """
                max_len = max((len(lst) if lst is not None else 0) for lst in columns.values())
                padded = {}
                for col, lst in columns.items():
                    values = lst if lst is not None else []
                    values = list(values)  # ensure list
                    values = values + [np.nan] * (max_len - len(values))
                    padded[col] = values
                return pd.DataFrame(padded)
            
            def ensure_min_len(lst, min_len=2):
                return lst if len(lst) >= min_len else [np.nan] + lst
            
            # Loss curves DataFrame
            df_loss = pad_columns_to_max_length({
                "Train Loss": stats.get("train_losses", []),
                "Val Loss": stats.get("val_losses", []),
            })
            
            # Dice curves DataFrame
            df_dice = pad_columns_to_max_length({
                "Train Dice": stats.get("train_dices", []),
                "Val Dice": stats.get("val_dices", []),
            })
            
            # LR DataFrame
            df_lr = pad_columns_to_max_length({
                "Learning Rate": stats.get("lrs", []),
            })
            
            # Additional metrics DataFrame
            df_metrics = pd.DataFrame()
            if "accuracy" in stats and stats["accuracy"]:
                df_metrics["Accuracy"]    = ensure_min_len(stats["accuracy"])
            if "precision" in stats and stats["precision"]:
                df_metrics["Precision"]   = ensure_min_len(stats["precision"])

            # Separate dataframe for recall
            df_recall = pd.DataFrame()
            if "recall" in stats and stats["recall"]:
                df_recall["Recall"]      = ensure_min_len(stats["recall"])

            # Separate out training and validation times
            train_times = [tm for tm, ph in zip(stats["batch_times"], 
                                                stats["batch_phases"]) if ph == 'Training']
            val_times = [tm for tm, ph in zip(stats["batch_times"], 
                                              stats["batch_phases"]) if ph == 'Validation']

            # Pad the shorter list with NaN
            max_len = max(len(train_times), len(val_times))
            train_times += [np.nan] * (max_len - len(train_times))
            val_times += [np.nan] * (max_len - len(val_times))

            # Create the dataframe for the timing line chart
            df_timing = pd.DataFrame()
            df_timing["Training"] = train_times
            df_timing["Validation"] = val_times

            # Display additional details dynamically
            if not df_loss.empty:
               loss_curve.line_chart(df_loss, 
                                     height=200, 
                                     use_container_width=True
               )
            if not df_dice.empty:
                dice_curve.line_chart(df_dice, 
                                      height=200, 
                                      use_container_width=True
                )
            if not df_metrics.empty:
                metrics_curve.line_chart(df_metrics, 
                                         height=200, 
                                         use_container_width=True
                )

            # Chart recall and learning rate separately
            if not df_recall.empty:
                with recall_curve:
                    st.line_chart(df_recall, 
                                  height=200, 
                                  x_label="Recall", 
                                  use_container_width=True
                    )
            if not df_lr.empty:
                with lr_curve:
                    lr_curve.line_chart(df_lr, 
                                        height=200, 
                                        x_label="Learning Rate", 
                                        use_container_width=True
                    )
            if not df_timing.empty:
                timing_curve.line_chart(df_timing,
                                        height=200,
                                        use_container_width=True
                )

            # Display the "live" best accuracy seen so far
            val_dices = stats.get("val_dices", [])
            msg_dice="N/A"
            if val_dices:
                best_epoch_dice = np.argmax(val_dices)
                best_dice = val_dices[best_epoch_dice]
                msg_dice = f"{best_dice:.2f} (at epoch {best_epoch_dice})"

            val_accuracies = stats.get("accuracy", [])
            msg_acc="N/A"
            if val_accuracies:
                # Compute the "best" epoch based on accuracy
                best_epoch_acc = np.argmax(val_accuracies)
                best_acc = val_accuracies[best_epoch_acc]
                msg_acc = f"{best_acc:.2f} (at epoch {best_epoch_acc})"

                best_metric.markdown(
                    BEST_METRIC_HTML.format(
                            metric_name="Accuracy",
                            metric_value=f"{best_acc:.2f}"
                    ),
                    unsafe_allow_html=True
                )

        return _callback

    # ---- Button event: Launch training ----
    if run_btn:
        # Fetch the training device
        training_device = session_state_get("training_device")

        # Gather model info to display some details on the spinner
        model_info = session_state_get('monai_model_info')
        with st.spinner(f"Training {model_info['name']} on {training_device}...", show_time=True):
            # Prepare hyperparameters based on selected model and load
            # corresponding model implementation
            params = {}
            for pname in model_info["params"]:
                params[pname] = session_state_get(f"monai_training_{pname}")
            model_impl = f"products.monai.training.models.{model_info['id']}"
            info(TRAIN_CONSOLE_LOG_KEY, 
                 f"Initiating training {model_info['id']} with {params}")
        
            # Initialize training stats with defaults
            progress_callback_updated_stats(reset=True, max_epochs=params["epochs"])

            # Fetch the latest training data set 
            stats = session_state_get("training_stats")

            # Initiate model training through dynamic loading of
            # implementation, passing in the callback for dynamic UI updates
            run_dynamic_monai_training(
                model_impl=model_impl,
                device=training_device,
                params=params,
                max_epochs=params["epochs"],
                progress_callback=compose_callback_for_training()
            )
            info(TRAIN_CONSOLE_LOG_KEY, 
                 f"Completed training {model_info['id']} on {training_device} with {params}")


