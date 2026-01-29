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
#!/usr/bin/env python3

import os
import shutil
import sys
import time
import numpy as np
import torch
import nibabel as nib
import matplotlib.pyplot as plt
from pathlib import Path
import requests
import tarfile
import tempfile

from monai.data import Dataset, DataLoader, decollate_batch
from monai.inferers import SlidingWindowInferer
from monai.networks.nets import UNet
from monai.transforms import (
    Activationsd,
    AsDiscreted,
    Compose,
    EnsureChannelFirstd,
    EnsureTyped,
    Invertd,
    LoadImaged,
    Orientationd,
    SaveImaged,
    ScaleIntensityRanged,
    Spacingd,
)

def run_bundle_spleen_inference(
    input_path,
    bundle_root,
    output_dir,
    device_id=0,
    model_path=None  # Specific model path from Streamlit UI
):
    """
    Run spleen CT segmentation inference using exact MONAI bundle configuration
    """
    print("=== MONAI BUNDLE SPLEEN CT SEGMENTATION INFERENCE ===")
    
    # Start overall timing
    start_time = time.time()
    timing_info = {}

    # Import path utilities for consistent path handling
    import sys
    from pathlib import Path as PathLib
    current_file = PathLib(__file__).resolve()
    project_root = None
    for parent in current_file.parents:
        if parent.name == "rocm-ls-examples":
            project_root = parent
            break
    if project_root:
        sys.path.insert(0, str(project_root))

    from components.path_utils import resolve_path

    def resolve_path_wrapper(path_str):
        """Convert path to absolute Path object using utility"""
        return Path(resolve_path(path_str))
    
    # Convert paths to absolute Path objects
    input_path = str(resolve_path_wrapper(input_path))
    bundle_root = str(resolve_path_wrapper(bundle_root))
    output_dir = str(resolve_path_wrapper(output_dir))

    # Setup device (supporting both CPU and GPU)
    device_setup_start = time.time()
    if device_id == -1:
        # CPU mode requested
        device = torch.device("cpu")
    else:
        # GPU mode
        if torch.cuda.is_available():
            device = torch.device(f"cuda:{device_id}")
        else:
            print("⚠️ GPU requested but CUDA not available, falling back to CPU")
            device = torch.device("cpu")
    
    timing_info['device_setup_time'] = time.time() - device_setup_start
    print(f"Using device: {device}")
    print(f"Device setup time: {timing_info['device_setup_time']:.3f}s")

    # Check input file
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input file not found: {input_path}")

    print(f"Input file: {input_path}")
    print(f"Bundle root: {bundle_root}")

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Bundle configuration parameters
    image_key = "image"
    
    # Prepare data exactly as in bundle config
    data_prep_start = time.time()
    data = [{"image": input_path}]

    # Define preprocessing transforms (exactly matching bundle config)
    print("Setting up preprocessing transforms...")
    transforms_start = time.time()
    preprocessing = Compose([
        LoadImaged(keys=image_key),
        EnsureChannelFirstd(keys=image_key),
        Orientationd(keys=image_key, axcodes="RAS"),
        Spacingd(
            keys=image_key,
            pixdim=[1.5, 1.5, 2.0],
            mode="bilinear"
        ),
        ScaleIntensityRanged(
            keys=image_key,
            a_min=-57,
            a_max=164,
            b_min=0.0,
            b_max=1.0,
            clip=True,
        ),
        EnsureTyped(keys=image_key),
    ])

    # Define postprocessing transforms (exactly matching bundle config)
    postprocessing = Compose([
        Activationsd(keys="pred", softmax=True),  # Apply softmax first
        Invertd(
            keys="pred",
            transform=preprocessing,
            orig_keys=image_key,
            nearest_interp=False,
            to_tensor=True,
        ),
        AsDiscreted(keys="pred", argmax=True),
        SaveImaged(
            keys="pred",
            output_dir=output_dir,
            output_ext=".nii.gz",
            output_dtype=np.float32,
            output_postfix="seg",
            separate_folder=True,
        ),
    ])
    
    timing_info['transforms_setup_time'] = time.time() - transforms_start
    print(f"Transform setup time: {timing_info['transforms_setup_time']:.3f}s")

    # Create dataset and dataloader (matching bundle config)
    dataset_start = time.time()
    dataset = Dataset(data=data, transform=preprocessing)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=1)
    timing_info['dataset_setup_time'] = time.time() - dataset_start
    print(f"Dataset setup time: {timing_info['dataset_setup_time']:.3f}s")

    # Create network (exactly matching bundle config)
    print("Creating UNet model...")
    model_creation_start = time.time()
    network = UNet(
        spatial_dims=3,
        in_channels=1,
        out_channels=2,
        channels=[16, 32, 64, 128, 256],
        strides=[2, 2, 2, 2],
        num_res_units=2,
        norm="batch",
    ).to(device)
    
    timing_info['model_creation_time'] = time.time() - model_creation_start
    print(f"Model creation time: {timing_info['model_creation_time']:.3f}s")

    # Load model weights - use specific path if provided, otherwise use dynamic discovery
    if model_path is not None:
        # Use specific model path provided (from Streamlit UI)
        final_model_path = str(model_path)
        if not os.path.exists(final_model_path):
            raise FileNotFoundError(f"Specified model file not found: {final_model_path}")
        print(f"Using selected model from UI: {final_model_path}")
    else:
        # Fallback: Use dynamic discovery
        from products.monai.inference.data_manager import get_available_model_files

        # Get all available spleen models
        spleen_models = get_available_model_files('spleen_ct_segmentation')
        if not spleen_models:
            spleen_models = get_available_model_files('spleen_ct_seg')  # Alternative naming

        # Also check bundle location
        bundle_model_path = os.path.join(bundle_root, "models", "model.pt")
        if os.path.exists(bundle_model_path) and bundle_model_path not in spleen_models:
            spleen_models.insert(0, bundle_model_path)  # Prioritize bundle model

        if not spleen_models:
            raise FileNotFoundError("No spleen CT segmentation models found. Please train a model or download from model zoo.")

        # Use the first (highest priority) model
        final_model_path = spleen_models[0]
        print(f"Selected model: {final_model_path}")
        if len(spleen_models) > 1:
            print(f"Other available models: {len(spleen_models) - 1}")
            for i, alt_model in enumerate(spleen_models[1:4], 1):  # Show up to 3 alternatives
                print(f"  Alternative {i}: {os.path.basename(alt_model)}")
    
    # Convert relative path to absolute path
    if not os.path.isabs(final_model_path):
        final_model_path = os.path.abspath(final_model_path)

    print(f"Loading model from: {final_model_path}")
    model_loading_start = time.time()
    checkpoint = torch.load(final_model_path, map_location=device, weights_only=True)
    if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        network.load_state_dict(checkpoint['state_dict'])
    else:
        network.load_state_dict(checkpoint)
    timing_info['model_loading_time'] = time.time() - model_loading_start
    print(f"Model loaded successfully in {timing_info['model_loading_time']:.3f}s")

    # Create inferer (exactly matching bundle config)
    inferer_start = time.time()
    inferer = SlidingWindowInferer(
        roi_size=[96, 96, 96],
        sw_batch_size=4,
        overlap=0.5
    )
    timing_info['inferer_setup_time'] = time.time() - inferer_start
    print(f"Inferer setup time: {timing_info['inferer_setup_time']:.3f}s")

    # Run inference
    print("Starting inference...")
    inference_start = time.time()
    network.eval()
    raw_outputs = []
    processed_inputs = []
    
    data_processing_time = 0
    actual_inference_time = 0
    postprocessing_time = 0
    
    with torch.no_grad():
        for i, batch_data in enumerate(dataloader):
            print(f"Processing image {i+1}/{len(dataloader)}...")
            
            # Time data transfer to device
            data_transfer_start = time.time()
            images = batch_data[image_key].to(device)
            data_processing_time += time.time() - data_transfer_start
            print(f"Input shape: {images.shape}")
            
            # Store processed input for visualization
            processed_inputs.append(images.detach().cpu()[0, 0])  # Remove batch and channel dims
            
            # Run inference using the bundle's sliding window inferer
            model_inference_start = time.time()
            predictions = inferer(images, network)
            actual_inference_time += time.time() - model_inference_start
            print(f"Raw prediction shape: {predictions.shape}")
            
            # Store raw logits for analysis
            raw_outputs.append(predictions.detach().cpu())
            
            # Analyze raw predictions
            bg_logits = predictions[0, 0].detach().cpu()
            spleen_logits = predictions[0, 1].detach().cpu()
            print(f"Background logits - min: {bg_logits.min():.3f}, max: {bg_logits.max():.3f}, mean: {bg_logits.mean():.3f}")
            print(f"Spleen logits - min: {spleen_logits.min():.3f}, max: {spleen_logits.max():.3f}, mean: {spleen_logits.mean():.3f}")
            
            # Apply postprocessing
            postprocess_start = time.time()
            batch_data["pred"] = predictions
            batch_data = [postprocessing(item) for item in decollate_batch(batch_data)]
            postprocessing_time += time.time() - postprocess_start

    timing_info['data_processing_time'] = data_processing_time
    timing_info['actual_inference_time'] = actual_inference_time
    timing_info['postprocessing_time'] = postprocessing_time
    timing_info['total_inference_time'] = time.time() - inference_start
    timing_info['total_time'] = time.time() - start_time
    
    # Print detailed timing information
    print("\n" + "="*60)
    print("PERFORMANCE TIMING BREAKDOWN")
    print("="*60)
    print(f"Device setup time:      {timing_info['device_setup_time']:.3f}s")
    print(f"Transform setup time:   {timing_info['transforms_setup_time']:.3f}s")
    print(f"Dataset setup time:     {timing_info['dataset_setup_time']:.3f}s")
    print(f"Model creation time:    {timing_info['model_creation_time']:.3f}s")
    print(f"Model loading time:     {timing_info['model_loading_time']:.3f}s")
    print(f"Inferer setup time:     {timing_info['inferer_setup_time']:.3f}s")
    print(f"Data processing time:   {timing_info['data_processing_time']:.3f}s")
    print(f"Actual inference time:  {timing_info['actual_inference_time']:.3f}s")
    print(f"Postprocessing time:    {timing_info['postprocessing_time']:.3f}s")
    print(f"Total inference time:   {timing_info['total_inference_time']:.3f}s")
    print(f"TOTAL TIME:             {timing_info['total_time']:.3f}s")
    print("="*60)

    print(f"Bundle inference completed! Results saved to: {output_dir}")
    return raw_outputs, processed_inputs, output_dir, final_model_path, timing_info

def create_bundle_visualization(
    input_path,
    segmentation_dir,
    output_image_path,
    raw_outputs=None,
    processed_inputs=None
):
    """
    Create visualization from bundle inference results
    """
    print("=== CREATING BUNDLE SPLEEN SEGMENTATION VISUALIZATION ===")

    # Convert relative paths to absolute paths using project root detection
    def resolve_path(path_str):
        if os.path.isabs(path_str):
            return path_str

        current_file = Path(__file__).resolve()
        project_root = None
        for parent in current_file.parents:
            if parent.name == "rocm-ls-examples":
                project_root = parent
                break

        if project_root is None:
            project_root = Path(__file__).parent.parent.parent.parent

        return str(project_root / path_str)

    input_path = resolve_path(input_path)
    segmentation_dir = resolve_path(segmentation_dir)
    output_image_path = resolve_path(output_image_path)

    # Load original image
    original_img = nib.load(input_path)
    original_data = original_img.get_fdata()
    print(f"Original image shape: {original_data.shape}")

    # Use raw outputs if provided
    if raw_outputs and len(raw_outputs) > 0:
        raw_output = raw_outputs[0]  # First (and only) image
        print("Using raw model output for visualization")
        
        # Apply softmax to get probabilities
        raw_probs = torch.softmax(raw_output, dim=1)
        predicted_mask = raw_probs[0, 1].numpy()  # Spleen channel
        print(f"Predicted mask shape: {predicted_mask.shape}")
        print(f"Spleen probability - min: {predicted_mask.min():.4f}, max: {predicted_mask.max():.4f}, mean: {predicted_mask.mean():.4f}")
        
        # Use processed input
        if processed_inputs and len(processed_inputs) > 0:
            display_image = processed_inputs[0].numpy()
            print(f"Using processed input for display, shape: {display_image.shape}")
        else:
            display_image = original_data
    else:
        # Load from saved segmentation files
        print("Loading segmentation from saved files...")
        seg_files = list(Path(segmentation_dir).glob("**/*.nii.gz"))
        if not seg_files:
            raise FileNotFoundError(f"No segmentation files found in {segmentation_dir}")
        
        seg_path = seg_files[0]
        print(f"Loading segmentation: {seg_path}")
        seg_img = nib.load(str(seg_path))
        predicted_mask = seg_img.get_fdata()
        display_image = original_data

    # Calculate dimensions
    original_slices = original_data.shape[2]
    processed_slices = predicted_mask.shape[2]
    print(f"Original slices: {original_slices}, Processed slices: {processed_slices}")

    # Calculate corresponding anatomical slices
    slice_ratio = original_slices / processed_slices
    processed_slice_idx = processed_slices // 2
    original_slice_idx = int(processed_slice_idx * slice_ratio)
    
    # Ensure indices are within bounds
    original_slice_idx = max(0, min(original_slice_idx, original_slices - 1))
    processed_slice_idx = max(0, min(processed_slice_idx, processed_slices - 1))

    print(f"Showing slices - Original: {original_slice_idx}/{original_slices}, Processed: {processed_slice_idx}/{processed_slices}")

    # Calculate segmentation statistics
    threshold = 0.5
    thresholded_mask = (predicted_mask > threshold).astype(int)
    total_voxels = np.prod(predicted_mask.shape)
    segmented_voxels = np.sum(thresholded_mask)
    segmentation_percentage = (segmented_voxels / total_voxels) * 100

    # Quality assessment
    quality_status = "✅ Good" if segmentation_percentage < 5.0 else "⚠️ High (check for over-segmentation)"

    # Create visualization
    plt.figure("bundle_spleen_segmentation", (20, 6))

    # Panel 1: Original image
    plt.subplot(1, 4, 1)
    plt.title(f"Original CT\n(Slice {original_slice_idx}/{original_slices})")
    plt.imshow(original_data[:, :, original_slice_idx], cmap="gray", aspect='equal')
    plt.axis('off')

    # Panel 2: Processed input
    plt.subplot(1, 4, 2)
    plt.title(f"Processed Input\n(Slice {processed_slice_idx}/{processed_slices})")
    if processed_inputs and len(processed_inputs) > 0:
        plt.imshow(display_image[:, :, processed_slice_idx], cmap="gray", aspect='equal')
    else:
        plt.imshow(original_data[:, :, original_slice_idx], cmap="gray", aspect='equal')
    plt.axis('off')

    # Panel 3: Probability map
    plt.subplot(1, 4, 3)
    plt.title(f"Spleen Probability\n(Range: {predicted_mask.min():.3f} - {predicted_mask.max():.3f})")
    im = plt.imshow(predicted_mask[:, :, processed_slice_idx], cmap="Reds", vmin=0, vmax=1, aspect='equal')
    plt.colorbar(im, shrink=0.8)
    plt.axis('off')

    # Panel 4: Final segmentation overlay
    plt.subplot(1, 4, 4)
    plt.title(f"Segmentation Overlay\n(Threshold: {threshold})")
    if processed_inputs and len(processed_inputs) > 0:
        plt.imshow(display_image[:, :, processed_slice_idx], cmap="gray", aspect='equal')
    else:
        plt.imshow(original_data[:, :, original_slice_idx], cmap="gray", aspect='equal')
    
    # Overlay thresholded segmentation
    mask_overlay = np.ma.masked_where(
        predicted_mask[:, :, processed_slice_idx] <= threshold, 
        predicted_mask[:, :, processed_slice_idx]
    )
    plt.imshow(mask_overlay, cmap='Reds', alpha=0.7, aspect='equal')
    plt.axis('off')

    plt.suptitle(f'MONAI Bundle Spleen CT Segmentation Results\n'
                f'Input: {os.path.basename(input_path)} | '
                f'Segmented: {segmented_voxels} voxels ({segmentation_percentage:.2f}%) | '
                f'Status: {quality_status}',
                fontsize=14)

    plt.tight_layout()

    # Save visualization
    os.makedirs(os.path.dirname(output_image_path), exist_ok=True)
    plt.savefig(output_image_path, dpi=150, bbox_inches='tight')
    print(f"Visualization saved to: {output_image_path}")
    plt.close()

    return {
        'output_image_path': output_image_path,
        'total_voxels': total_voxels,
        'segmented_voxels': segmented_voxels,
        'segmentation_percentage': segmentation_percentage,
        'original_slice_used': original_slice_idx,
        'processed_slice_used': processed_slice_idx,
        'original_shape': original_data.shape,
        'segmentation_shape': predicted_mask.shape,
        'quality_status': quality_status
    }

def run_spleen_segmentation(input_path, model_path=None, model_dir=None, device=None):
    """
    Streamlit-compatible wrapper for spleen CT segmentation with CPU/GPU support
    
    Args:
        input_path (str): Path to input NIfTI file
        model_dir (str): Path to model directory (unused, kept for compatibility)
        device (torch.device): PyTorch device object (CPU or GPU)
        
    Returns:
        dict: Results dictionary with visualization, metrics, and performance data
    """
    import time
    
    try:
        start_time = time.time()
        
        # Handle device selection
        if device is None:
            device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        
        device_name = f"{device.type.upper()}"
        if device.type == 'cuda' and device.index is not None:
            device_name += f":{device.index}"
        
        print(f"Using device: {device} ({device_name})")
        
        # Extract device_id for compatibility with run_bundle_spleen_inference
        if device.type == 'cuda':
            device_id = device.index if device.index is not None else 0
        else:
            device_id = -1  # Will be handled as CPU in the inference function
        
        # Use the model-zoo path for consistent results (relative to project root)
        bundle_root = "model-zoo/models/spleen_ct_segmentation"
        output_dir = "demo/output/streamlit_spleen_segmentation"
        visualization_path = "demo/output/streamlit_spleen_visualization.png"
        
        # Convert relative paths to absolute paths using resolve_path
        current_file = Path(__file__).resolve()
        project_root = None
        for parent in current_file.parents:
            if parent.name == "rocm-ls-examples":
                project_root = parent
                break

        if project_root is None:
            # Fallback - assume we're in the right place
            project_root = Path(__file__).parent.parent.parent.parent

        bundle_root = str(project_root / bundle_root)
        output_dir = str(project_root / output_dir)
        visualization_path = str(project_root / visualization_path)
        
        # Measure model loading time
        model_load_start = time.time()
        print(f"Loading spleen CT model...")
        
        # Run bundle-based inference (this includes model loading)
        inference_start = time.time()
        raw_outputs, processed_inputs, seg_dir, used_model_path, timing_info = run_bundle_spleen_inference(
            input_path=input_path,
            bundle_root=bundle_root,
            output_dir=output_dir,
            device_id=device_id,
            model_path=model_path  # Pass the selected model path
        )
        inference_time = time.time() - inference_start
        
        # Use actual timing from detailed measurements
        model_load_time = timing_info.get('model_loading_time', 0.0)
        actual_inference_time = timing_info.get('actual_inference_time', inference_time)
        
        # Measure visualization time
        viz_start = time.time()
        viz_results = create_bundle_visualization(
            input_path=input_path,
            segmentation_dir=seg_dir,
            output_image_path=visualization_path,
            raw_outputs=raw_outputs,
            processed_inputs=processed_inputs
        )
        viz_time = time.time() - viz_start
        
        total_time = time.time() - start_time
        
        # Calculate processing metrics
        total_voxels = viz_results['total_voxels']
        voxels_per_second = total_voxels / inference_time if inference_time > 0 else 0
        
        print(f"Performance Summary:")
        print(f"  Model loading time: {model_load_time:.2f}s")
        print(f"  Inference time: {inference_time:.2f}s") 
        print(f"  Visualization time: {viz_time:.2f}s")
        print(f"  Total time: {total_time:.2f}s")
        print(f"  Processing speed: {voxels_per_second:.0f} voxels/sec")
        
        # Return results in Streamlit-compatible format with all required fields
        results = {
            'success': True,
            'plot_image': visualization_path,
            'model_path': used_model_path,  # Required by Streamlit driver
            'input_path': input_path,  # Required by Streamlit driver
            'segmentation_dir': seg_dir,  # Required by Streamlit driver
            'segmentation_percentage': viz_results['segmentation_percentage'],
            'segmented_voxels': viz_results['segmented_voxels'],
            'total_voxels': viz_results['total_voxels'],
            'original_shape': viz_results['original_shape'],
            'segmentation_shape': viz_results['segmentation_shape'],
            'quality_status': viz_results['quality_status'],
            'output_directory': seg_dir,
            'visualization_path': visualization_path,
            'original_slice_used': viz_results['original_slice_used'],
            'processed_slice_used': viz_results['processed_slice_used'],
            
            # Performance measurements (matching pathology_inference format)
            'model_load_time': model_load_time,
            'inference_time': inference_time,  # Required by Streamlit driver
            'processing_time': inference_time,
            'visualization_time': viz_time,
            'total_time': total_time,
            'voxels_per_second': voxels_per_second,
            'input_file': os.path.basename(input_path),
            'input_size_mb': os.path.getsize(input_path) / (1024 * 1024) if os.path.exists(input_path) else 0,
            
            # Additional metrics for display
            'volume_percentage': viz_results['segmentation_percentage'],
            'processing_speed_desc': f"{voxels_per_second:.0f} voxels/sec",
            'quality_assessment': viz_results['quality_status'],
            'device_used': str(device),
            'device_type': device.type.upper(),
            'device_name': device_name
        }
        
        return results
        
    except Exception as e:
        import traceback
        error_msg = f"Spleen CT inference error: {str(e)}"
        print(f"❌ {error_msg}")
        traceback.print_exc()
        
        return {
            'success': False,
            'error': error_msg,
            'plot_image': None,
            'model_path': '',  # Required by Streamlit driver
            'input_path': input_path if 'input_path' in locals() else '',  # Required by Streamlit driver
            'segmentation_dir': '',  # Required by Streamlit driver
            'segmentation_percentage': 0.0,
            'model_load_time': 0.0,
            'inference_time': 0.0,  # Required by Streamlit driver
            'processing_time': 0.0,
            'total_time': 0.0,
            'voxels_per_second': 0.0,
            'device_used': str(device) if 'device' in locals() else 'unknown',
            'device_type': device.type.upper() if 'device' in locals() else 'unknown'
        }

def main():
    """
    Main function to run MONAI bundle spleen CT segmentation
    """
    # Use new demo structure paths
    input_path = "demo/data/spleen_ct_segmentation/spleen_1.nii.gz"
    # check if file exists if not download it
    if os.path.exists(input_path) is False:
        os.makedirs(os.path.dirname(input_path), exist_ok=True)
        url = "https://msd-for-monai.s3-us-west-2.amazonaws.com/Task09_Spleen.tar"
        # Download in a temp dir and extract to input_path
        with tempfile.TemporaryDirectory() as tmpdir:
            tar_path = os.path.join(tmpdir, "Task09_Spleen.tar")
            print(f"Downloading sample data from {url}...")
            response = requests.get(url, stream=True)
            with open(tar_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            print("Extracting data...")
            with tarfile.open(tar_path, 'r') as tar:
                tar.extract("Task09_Spleen/imagesTs/._spleen_1.nii.gz", path=tmpdir)
            # Move the specific file to input_path
            extracted_file = os.path.join(tmpdir, "Task09_Spleen/imagesTs/spleen_1.nii.gz")
            if os.path.exists(extracted_file):
                shutil.move(extracted_file, input_path)
            else:
                raise FileNotFoundError("Downloaded data does not contain expected file.")
    bundle_root = "model-zoo/models/spleen_ct_segmentation"
    output_dir = "demo/output/bundle_spleen_segmentation"
    visualization_path = "demo/output/bundle_spleen_visualization.png"

    # Convert relative paths to absolute paths using project root
    current_file = Path(__file__).resolve()
    project_root = None
    for parent in current_file.parents:
        if parent.name == "rocm-ls-examples":
            project_root = parent
            break

    if project_root is None:
        # Fallback - assume we're in the right place
        project_root = Path(__file__).parent.parent.parent.parent

    input_path = str(project_root / input_path)
    bundle_root = str(project_root / bundle_root)
    output_dir = str(project_root / output_dir)
    visualization_path = str(project_root / visualization_path)

    try:
        # Phase 1: Run bundle-based inference
        print("Phase 1: Running MONAI bundle spleen segmentation inference...")
        raw_outputs, processed_inputs, seg_dir, model_path, timing_info = run_bundle_spleen_inference(
            input_path=input_path,
            bundle_root=bundle_root,
            output_dir=output_dir,
            device_id=0
        )

        print("✅ Bundle inference completed successfully!")

        # Phase 2: Create visualization
        print("\nPhase 2: Creating bundle visualization...")
        viz_results = create_bundle_visualization(
            input_path=input_path,
            segmentation_dir=seg_dir,
            output_image_path=visualization_path,
            raw_outputs=raw_outputs,
            processed_inputs=processed_inputs
        )

        print("✅ Bundle visualization completed successfully!")

        # Print comprehensive summary
        print(f"\n" + "="*80)
        print("MONAI BUNDLE SPLEEN CT SEGMENTATION RESULTS")
        print("="*80)
        print(f"Input file: {os.path.basename(input_path)}")
        print(f"Bundle config: {bundle_root}")
        print(f"Original shape: {viz_results['original_shape']}")
        print(f"Processed shape: {viz_results['segmentation_shape']}")
        print(f"Segmented voxels: {viz_results['segmented_voxels']}")
        print(f"Segmentation percentage: {viz_results['segmentation_percentage']:.2f}%")
        print(f"Quality assessment: {viz_results['quality_status']}")
        print(f"Visualization slices - Original: {viz_results['original_slice_used']}, Processed: {viz_results['processed_slice_used']}")
        print(f"Output directory: {seg_dir}")
        print(f"Visualization: {visualization_path}")
        print("="*80)

        # Provide recommendations based on results
        if viz_results['segmentation_percentage'] > 8.0:
            print("\n🔍 RECOMMENDATIONS:")
            print("• High segmentation percentage suggests potential over-segmentation")
            print("• Consider adjusting probability threshold or checking model weights")
            print("• Verify preprocessing pipeline matches training configuration")
        elif viz_results['segmentation_percentage'] < 0.5:
            print("\n🔍 RECOMMENDATIONS:")
            print("• Low segmentation percentage may indicate under-segmentation")
            print("• Check if spleen is actually present in the image")
            print("• Consider lowering probability threshold")
        else:
            print("\n✅ Results look reasonable for spleen segmentation")

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
