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
import cupy as cp
import numpy as np
import time
import base64
from io import BytesIO
from PIL import Image

from components.diagnostics import error, exception, warning, note, info
from components.state import (
    session_state_get,
    session_state_set,
    session_state_append_times,
)
from components.styles import (
    TILE_PLACEHOLDER_CSS,
    TILE_PLACEHOLDER_HTML,
)
from components.utility import rescale_image

from products.hipcim.performance import display_performance_graph
from products.hipcim.transforms import apply_pipeline
from products.hipcim.console import CONSOLE_LOG_KEY

def display_single_tile_scaled(np_img, caption, display_size=400):
    """
    Convert the NumPy array into an image and rescale it to the specified
    (square) dimensions preserving the aspect ratio.
    Display the rescaled image using st.image.
    Returns the rescaled image.
    Now safely handles both float and uint8, as well as GPU arrays.
    """
    # If input is cupy (GPU), convert to numpy
    if hasattr(np_img, "get"):
        np_img = np_img.get()

    # Handle float images (e.g., skimage/cucim: [0,1], float32/64)
    if np.issubdtype(np_img.dtype, np.floating):
        np_img = np.clip(np_img, 0, 1)
        np_img = (np_img * 255).astype(np.uint8)
    elif np_img.dtype != np.uint8:
        np_img = np_img.astype(np.uint8)

    display_size=session_state_get('tile_display_size')
    img = Image.fromarray(np_img)
    img = rescale_image(img, display_size, display_size)
    st.image(img, caption=caption, width='content', clamp=True)
    return img

# Display WSI tiles side-by-side
def display_tiles():
    # Retrieve the saved CuImage
    cuImage = session_state_get('cuImage')
    x = session_state_get('position_x')
    y = session_state_get('position_y')
    tile_size = session_state_get('tile_size')

    # Read the tile on GPU
    try:
        # Track GPU timing
        tstart = time.time()

        # Read tile into GPU
        gpu_tile = cuImage.read_region((x, y),
                                       (tile_size, tile_size),
                                       device="cuda")

        # Run transformations on GPU tile
        transformed_gpu_tile = apply_pipeline(gpu_tile, "cuda")

        # Track GPU time
        gpu_time = time.time() - tstart
        session_state_append_times('gpu_times', gpu_time)

    # except Exception as e:
    #     gpu_tile = cp.random.rand(tile_size, tile_size)
    #     transformed_gpu_tile = cp.random.rand(tile_size, tile_size)
    #     exception(CONSOLE_LOG_KEY, 
    #               f"failed to load tile from {selected_wsi} on GPU: {e}")
    #     note(CONSOLE_LOG_KEY, f"using random CuPy noise as placeholder")
    except Exception as e:
        gpu_tile = cp.random.rand(tile_size, tile_size)
        transformed_gpu_tile = cp.random.rand(tile_size, tile_size)
        wsi_name = session_state_get('selected_wsi', 'unknown')
        exception(CONSOLE_LOG_KEY,
                  f"failed to load tile from {wsi_name} on GPU: {e}")
        note(CONSOLE_LOG_KEY, f"using random CuPy noise as placeholder")


    # Read the tile on CPU
    try:
        # Track CPU timing
        tstart = time.time()

        # Read the tile into CPU
        cpu_tile = cuImage.read_region((x, y), (tile_size, tile_size))

        # Run transformations on CPU tile
        transformed_cpu_tile = apply_pipeline(cpu_tile)

        # Track CPU time
        cpu_time = time.time() - tstart
        session_state_append_times('cpu_times', cpu_time)

    except Exception as e:
        cpu_tile = np.random.rand(tile_size, tile_size)
        transformed_cpu_tile = np.random.rand(tile_size, tile_size)
        wsi_name = session_state_get('selected_wsi', 'unknown')
        exception(CONSOLE_LOG_KEY,
                  f"failed to load tile from {wsi_name} on GPU: {e}")
        note(CONSOLE_LOG_KEY, f"using random CuPy noise as placeholder")

    # Squirrel away the tiles as persistent
    session_state_set('gpu_tile', gpu_tile)
    session_state_set('cpu_tile', cpu_tile)

    # Convert to numpy array for display
    gpu_tile_np = cp.asnumpy(gpu_tile)
    cpu_tile_np = np.asarray(cpu_tile)
    transformed_gpu_tile_np = cp.asnumpy(transformed_gpu_tile)
    transformed_cpu_tile_np = np.asarray(transformed_cpu_tile)

    # Layout the display with improved styling
    wsi_tile, cpu_tile, gpu_tile, perf_tile = st.columns([1, 1, 1, 1.2], gap="medium")
    
    with wsi_tile:
        with st.container():
            display_single_tile_scaled(cpu_tile_np, "📷 Reference")

    with cpu_tile:
        with st.container():
            display_single_tile_scaled(transformed_cpu_tile_np, "🖥️ CPU Transform")

    with gpu_tile:
        with st.container():
            display_single_tile_scaled(transformed_gpu_tile_np, "⚡ GPU Transform")

    with perf_tile:
        with st.container():
            st.markdown("###### 📊 Performance")
            display_performance_graph()
