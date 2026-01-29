"""
AMD ROCm-LS Demo Application
---------------------------
Purpose: Launches interactive demo panels for ROCm-accelerated Life Sciences workflows,
         including hipCIM (for image analytics) and MONAI (for medical AI/deep learning).
"""
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

import torch
import streamlit as st

from components.io import render_markdown
from components.diagnostics import error, exception
from components.state import session_state_init

from products.hipcim.layout import hipcim_layout
from products.monai.layout import monai_layout

st.set_page_config(
    page_title="AMD ROCm-LS: Accelerated Imaging & AI for Life Sciences",
    layout="wide"
)

# Remove default Streamlit top padding
st.markdown("""
<style>
    .block-container { padding-top: 0 !important; }
    header[data-testid="stHeader"] { display: none; }
    #MainMenu { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# Render header with embedded logo
import base64
from pathlib import Path

def get_base64_image(image_path):
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode()

logo_base64 = get_base64_image("assets/amd-logo.png")

st.markdown(f"""
<div style="background: linear-gradient(90deg, #00A3E0, #00B4D8); padding: 12px 24px; display: flex; align-items: center; gap: 16px; margin: 0;">
    <img src="data:image/png;base64,{logo_base64}" alt="AMD" style="height: 28px; width: auto;">
    <span style="font-family: 'Segoe UI', sans-serif; font-size: 16px; font-weight: 600; color: #000;">ROCm Life Sciences: GPU-Accelerated Imaging & AI for Healthcare</span>
</div>
""", unsafe_allow_html=True)

# Initialize session state management
session_state_init()

# Create tabs for each product to be demo'ed
tabs = st.tabs(["**hipCIM**", "**MONAI**"])

with tabs[0]:
    hipcim_layout()

with tabs[1]:
    monai_layout()

# Display persistent contact and copyright info as footer
render_markdown("markdown/rocm-ls_footer.md")
