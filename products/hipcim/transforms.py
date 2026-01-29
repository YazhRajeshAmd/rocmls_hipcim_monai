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

# GPU-related imports
import cupy as cp
from cupyx.scipy.ndimage import sobel as gpu_sobel
from cupyx.scipy import ndimage as gpu_ndi
from cucim.skimage import color as gpu_color
from cucim.skimage.util import img_as_float32 as gpu_img_as_float32
from cucim.skimage.filters import gabor_kernel as gpu_gabor_kernel
from cucim.skimage.filters import threshold_otsu as gpu_threshold_otsu
from cucim.skimage.morphology import binary_dilation as gpu_binary_dilation, disk as gpu_disk
from cucim.skimage.morphology import remove_small_objects as gpu_remove_small_objects
from cucim.skimage.measure import label as gpu_label
from cucim.skimage.transform import (
        rotate as gpu_rotate, 
        warp as gpu_warp, 
        AffineTransform as gpu_AffineTransform
)

# CPU-related imports
import numpy as np
from scipy.ndimage import sobel as cpu_sobel
from scipy import ndimage as cpu_ndi
from skimage import color as cpu_color
from skimage.util import img_as_float32 as cpu_img_as_float32
from skimage.filters import gabor_kernel as cpu_gabor_kernel
from skimage.filters import threshold_otsu as cpu_threshold_otsu
from skimage.morphology import binary_dilation as cpu_binary_dilation, disk as cpu_disk
from skimage.morphology import remove_small_objects as cpu_remove_small_objects
from skimage.measure import label as cpu_label
from skimage.transform import (
        rotate as cpu_rotate, 
        warp as cpu_warp, 
        AffineTransform as cpu_AffineTransform
)

# Local imports
from components.diagnostics import error, exception, info, warning, note
from components.state import session_state_get
from products.hipcim.console import CONSOLE_LOG_KEY

# Convert an image to grayscale only if it is not already grayscale
# (common for both CPU and GPU)
def ensure_grayscale(img_array):
    # If image is already 2D (grayscale), return as is
    if img_array.ndim == 2:
        return img_array

    # If image is 3D with 3 or more channels (RGB or RGBA)
    elif img_array.ndim == 3 and img_array.shape[2] >= 3:
        # Use only the first three channels (R, G, B)
        return img_array[..., :3].mean(axis=-1)

    else:
        raise ValueError("Input image must be either grayscale or RGB.")

# Stain separation
def stain_separation(tile, device, stain="hematoxylin"):
    # Device stack selection
    if device == "cuda":
        xp = cp
        color = gpu_color
    else:
        xp = np
        color = cpu_color

    # Convert to array
    ihc_input = xp.array(tile)

    # Convert 4-channel (RGBA) regions to RGB
    if ihc_input.shape[-1] == 4:
        ihc_rgb = color.rgba2rgb(ihc_input)
    else:
        ihc_rgb = ihc_input

    # Transform to colorspace where the stains are separated
    ihc_hed = color.rgb2hed(ihc_rgb)

    # Create an RGB image for visualizing each of the stains
    null = xp.zeros_like(ihc_hed[:, :, 0])

    # Separate out selected stain
    if stain == "hematoxylin":
        ihc_h = color.hed2rgb(xp.stack((ihc_hed[:, :, 0], null, null), axis=-1))
        tile = ihc_h

    if stain == "eosin":
        ihc_e = color.hed2rgb(xp.stack((null, ihc_hed[:, :, 1], null), axis=-1))
        tile = ihc_e

    if stain == "dab":
        ihc_d = color.hed2rgb(xp.stack((null, null, ihc_hed[:, :, 2]), axis=-1))
        tile = ihc_d

    return tile

# Generate a Gabor filter using the parameters and return the filtered tile
# sigma: Controls the width of the Gaussian envelope. A larger
#        sigma will result in more smoothing; smaller details
#        may be suppressed, emphasizing larger structures.
# theta: Defines the orientation of the Gabor filter; helps in
#        detecting features oriented in different directions.
# lambda: Determines the frequency of the sinusoidal wave. A
#         smaller lambda emphasizes finer textures, while a
#         larger lambda highlights coarser patterns.
# psi: Changing psi can shift the sinusoidal component along
#      the spatial domain, which might reveal different
#      features depending on the phase shift.
# gamma: Controls the ellipticity of the Gaussian. A gamma less
#        than 1 creates elongated filters, which might be useful
#        for emphasizing features that appear more linear at
#        certain orientations.
def gabor_filter(tile, device, frequency=0.05, theta=0.0, sigma=1):
    if device == "cuda":
        xp = cp
        img_as_float32 = gpu_img_as_float32
        gabor_kernel = gpu_gabor_kernel
        ndi = gpu_ndi
    else:
        xp = np
        img_as_float32 = cpu_img_as_float32
        gabor_kernel = cpu_gabor_kernel
        ndi = cpu_ndi

    # Convert input to float32 and grayscale
    tile = img_as_float32(xp.asarray(tile))
    tile = ensure_grayscale(tile)

    # Create a single Gabor kernel with input parameters
    kernel = gabor_kernel(frequency, theta=theta, sigma_x=sigma, sigma_y=sigma)

    # Apply the Gabor filter (real part only, as in your original code)
    filtered = ndi.convolve(tile, xp.asarray(kernel.real), mode='wrap')

    # Compute mean and variance as features
    mean = filtered.mean()
    var = filtered.var()

    # Optionally, compute the "power" image (magnitude response)
    tile_norm = (tile - tile.mean()) / tile.std()
    power_img = xp.sqrt(
        ndi.convolve(tile_norm, xp.asarray(kernel.real), mode='wrap')**2 +
        ndi.convolve(tile_norm, xp.asarray(kernel.imag), mode='wrap')**2
    )

    ## Name	        Type	    Description	                                    Typical Use
    ## ----         ----        -----------                                     -----------
    ## filtered	    2D array	Filtered image (real part)	                    Feature map, analysis
    ## mean	        Scalar	    Mean of filtered image	                        Compact feature
    ## var	        Scalar	    Variance of filtered image	                    Compact feature
    ## power_img	2D array	Magnitude of filter response (real + imag)	    Visualization, analysis

    return power_img

# Apply Sobel filter to detect edges
def sobel_edge(tile, device):
    if device == "cuda":
        xp = cp
        sobel = gpu_sobel
    else:
        xp = np
        sobel = cpu_sobel

    # Convert the image to a numpy array
    img_array = xp.array(tile)

    # Convert to grayscale for simplicity
    gray_img = ensure_grayscale(img_array)

    # Normalize to [0, 1]
    gray_img = (gray_img - gray_img.min()) / (gray_img.max() - gray_img.min() + 1e-8)

    # Apply sobel filter to detect edges
    sobel_x = sobel(gray_img, axis=0)
    sobel_y = sobel(gray_img, axis=1)
    sobel_edges = xp.sqrt(sobel_x**2 + sobel_y**2)

    # Return as NumPy array
    return sobel_edges

# Apply binary dilation
def dilation(tile, device, iterations=3):
    if device == "cuda":
        xp = cp
        threshold_otsu = gpu_threshold_otsu
        disk = gpu_disk
        binary_dilation = gpu_binary_dilation
    else:
        xp = np
        threshold_otsu = cpu_threshold_otsu
        disk = cpu_disk
        binary_dilation = cpu_binary_dilation

    # Convert the image to a numpy array
    img_array = xp.array(tile)

    # Convert to grayscale for simplicity
    gray_img = ensure_grayscale(img_array)

    # Apply binary threshold
    thresh = threshold_otsu(gray_img)
    binary = gray_img > thresh

    # Apply dilation
    selem = disk(iterations)  # structuring element
    dilated = binary_dilation(binary, selem)

    # Convert the boolean dilated mask to a displayable format
    tile = dilated.astype(xp.uint8) * 255

    return tile

# Remove small objects
def erase_small_objects(tile, device, min_size=100):
    if device == "cuda":
        xp = cp
        threshold_otsu = gpu_threshold_otsu
        label = gpu_label
        remove_small_objects = gpu_remove_small_objects
    else:
        xp = np
        threshold_otsu = cpu_threshold_otsu
        label = cpu_label
        remove_small_objects = cpu_remove_small_objects

    # Convert RGBA to RGB
    if tile.shape[-1] == 4:
        tile = xp.array(tile)[:, :, :3]

    # Handle grayscale or color tiles
    if tile.ndim == 3 and tile.shape[2] >= 3:
        # Convert RGB to grayscale
        gray_img = (
            0.2989 * tile[:, :, 0]
            + 0.5870 * tile[:, :, 1]
            + 0.1140 * tile[:, :, 2]
        )
    else:
        # Already grayscale (H, W)
        gray_img = tile

    # Apply binary threshold
    thresh = threshold_otsu(gray_img)
    binary = gray_img > thresh

    # Label connected components
    labeled = label(binary)

    # Remove small objects (e.g., smaller than 100 pixels)
    cleaned = remove_small_objects(labeled, min_size=min_size)

    # Convert to binary mask for display
    result = cleaned > 0
    tile = result.astype(xp.uint8) * 255

    return tile

# Rotate patch by 45 degrees
def rotate_patch(tile, device, angle=45):
    if device == "cuda":
        xp = cp
    else:
        xp = np

    # Ensure tile is a (H, W, C) CuPy array
    if tile.shape[-1] == 4:
        tile = xp.asarray(tile)[:, :, :3]  # Remove alpha channel if present

    # Rotate image by 45 degrees
    rotated = rotate(
        tile,
        angle=angle,              # degrees counter-clockwise
        resize=False,             # keep original shape; set True to expand output
        center=None,              # rotate around image center
        preserve_range=True,      # maintain pixel value range
        order=1,                  # bilinear interpolation
        mode='constant',          # fill border with constant value (default=0)
        cval=0                    # value used for padding outside input bounds
    )

    return rotated

# Apply affine transformations to warp the tile
def affine_warp(tile, device, scale=(0.8, 0.8), rotation=0.2, shear=0.1, translation=(10, -5)):
    if device == "cuda":
        xp = cp
        AffineTransform = gpu_AffineTransform
        warp = gpu_warp
    else:
        xp = np
        AffineTransform = cpu_AffineTransform
        warp = cpu_warp

    # Remove alpha if present
    if tile.shape[-1] == 4:
        tile = xp.asarray(tile)[:, :, :3]

    # Define an affine transformation: scale, rotate, shear, translate
    tform = AffineTransform(
        scale=scale,                # shrink image
        rotation=rotation,          # ~11 degrees
        shear=shear,                # slight shear
        translation=translation     # x and y shift
    )

    # Apply warp using inverse mapping
    warped = warp(
        tile,
        inverse_map=tform.inverse,   # required
        preserve_range=True,
        output_shape=tile.shape[:2],  # same shape
        order=1,                      # bilinear interpolation
        mode='constant',
        cval=0
    )

    return warped

# Map the pipeline transforms to implementations
TRANSFORM_FN_MAP = {
    "stain_separation": stain_separation,
    "gabor_filter": gabor_filter,
    "sobel_edges": sobel_edge,
    "binary_dilation": dilation,
    "remove_small_objects": erase_small_objects,
    "rotate": rotate_patch,
    "warp_affine": affine_warp,
}

def apply_pipeline(tile, device=None):
    pipeline = session_state_get('pipeline')
    if pipeline:
        n_tr = len(pipeline)
        if device == "cuda":
            d_name = "GPU"
        else:
            d_name = "CPU"

        # Iterate through the transforms in the pipeline along with 
        # the parameters
        for i, transform in enumerate(pipeline):
            op = transform["op"]
            params = transform.get("params", {})
            fn = TRANSFORM_FN_MAP[op]
            try:
                info(CONSOLE_LOG_KEY, 
                     f"[{d_name}:{i+1}/{n_tr}] Applying {op}({params})")
                tile = fn(tile, device, **params)
            except Exception as e:
                exception(CONSOLE_LOG_KEY, 
                          f"[{d_name}:{i+1}/{n_tr}] Failed applying {op}({params})")
                continue
    return tile
