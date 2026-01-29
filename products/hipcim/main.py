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
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import io
import os
from components.io import render_markdown
from components.styles import (
    THUMB_PLACEHOLDER_CSS,
    THUMB_PLACEHOLDER_HTML,
    TILE_PLACEHOLDER_CSS,
    TILE_PLACEHOLDER_HTML,
)
from components.utility import (
    generate_wsi_thumbnail,
    rescale_image,
)
from components.state import session_state_get

from products.hipcim.metadata import hipcim_metadata
from products.hipcim.tile import display_tiles

# Disable PIL.Image.DecompressionBombError: decompression bomb DOS attack.
Image.MAX_IMAGE_PIXELS = None

def display_thumbnail_with_grid(thumb_path):
    # Load downsampled JPEG thumbnail
    thumb_np = np.array(Image.open(thumb_path).convert("RGB"))
    thumb_height, thumb_width = thumb_np.shape[:2]

    # Compute applicable scaling
    scale_x = thumb_width / session_state_get('wsi_width')
    scale_y = thumb_height / session_state_get('wsi_height')

    # Create matplotlib figure
    fig, ax = plt.subplots(figsize=(8, 2.9), dpi=96)
    # This yields image of 800x278 px (width x height)

    # Display thumbnail with ticks
    ax.imshow(thumb_np)
    ax.set_xticks(np.arange(0, thumb_width, step=100))
    ax.set_yticks(np.arange(0, thumb_height, step=100))
    ax.grid(color='gray', linestyle='--', linewidth=0.5)

    # Add labels for the ticks
    for i in ax.get_xticks():
        ax.text(i, -15, str(int(i / scale_x)), ha='center', va='center', fontsize=6)
    for j in ax.get_yticks():
        ax.text(-15, j, str(int(j / scale_y)), ha='right', va='center', fontsize=6)

    # Highlight RoI rectangle
    x = session_state_get('position_x')
    y = session_state_get('position_y')
    tile_size = session_state_get('tile_size')
    rect = plt.Rectangle((x * scale_x, y * scale_y),
                         tile_size * scale_x, tile_size * scale_y,
                         linewidth=2, edgecolor='red', facecolor='none')
    ax.add_patch(rect)

    ax.text(-120, thumb_height / 2, "Thumbnail Map", va='center', ha='right',
            rotation='vertical', fontsize=12)
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    plt.axis('on')

    # Save figure to PNG buffer
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=150)
    buf.seek(0)
    plt.close(fig)

    # Rescale the image using PIL to fit the placeholder without distortion
    image = Image.open(buf)
    image = rescale_image(image, max_width=800, max_height=250)

    # Render the thumbnail
    st.image(image) 

# Custom CSS for main panel tiles
MAIN_PANEL_CSS = """
<style>
.tile-container {
    background: #ffffff;
    border-radius: 12px;
    box-shadow: 0 2px 12px rgba(0,0,0,0.06);
    padding: 12px;
    border: 1px solid #e2e8f0;
    text-align: center;
}
.tile-label {
    font-size: 13px;
    font-weight: 600;
    color: #475569;
    margin-top: 8px;
    padding: 4px 8px;
    background: #f1f5f9;
    border-radius: 4px;
    display: inline-block;
}
.thumbnail-container {
    background: #ffffff;
    border-radius: 12px;
    box-shadow: 0 2px 12px rgba(0,0,0,0.06);
    padding: 16px;
    border: 1px solid #e2e8f0;
}
.metadata-container {
    background: linear-gradient(135deg, #fafbfc 0%, #f0f4f8 100%);
    border-radius: 12px;
    padding: 16px;
    border: 1px solid #e2e8f0;
    height: 100%;
}
</style>
"""

def hipcim_main():
    # Apply main panel CSS
    st.markdown(MAIN_PANEL_CSS, unsafe_allow_html=True)
    
    # hipCIM_r1: Thumbnail map with metadata panel
    st.markdown("##### 🔬 Whole Slide Image Overview")
    r1c1, r1c2 = st.columns([2.5, 1], gap="medium")

    with r1c1:
        current_wsi = session_state_get('selected_wsi')
        if current_wsi:
            # Generate the thumbnail from the WSI
            thumb_path = f".generated_thumbnails/{current_wsi}_thumb.jpg"
            with st.spinner('🔄 Generating thumbnail...'):
                generate_wsi_thumbnail(current_wsi, thumb_path, width=800)
            display_thumbnail_with_grid(thumb_path)
        else:
            st.markdown(THUMB_PLACEHOLDER_CSS, unsafe_allow_html=True)
            st.markdown(THUMB_PLACEHOLDER_HTML, unsafe_allow_html=True)

    with r1c2:
        hipcim_metadata()

    st.markdown("---")
    
    # hipCIM_r2: Display tiles side-by-side with section header
    st.markdown("##### 🖼️ Tile Comparison: Reference vs Transformed")
    display_tiles()

