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
import torch
import plotly.graph_objs as go
import importlib

from components.io import file_upload_widget, render_markdown
from components.tooltips import tooltips
from components.utility import get_image_shape
from components.console_log import append_console_log
from components.diagnostics import error, warning, note, info
from components.state import (
    session_state_get,
    session_state_set,
)

from products.monai.console import TRAIN_CONSOLE_LOG_KEY
from products.monai.training.models.zoo import MODEL_ZOO

# Layout the MONAI sidebar control panel
def monai_training_sidebar():
    render_markdown("markdown/monai_training_sidebar_header.md")
    
    # Model Selection with enhanced display
    st.markdown("""
    <style>
    .model-info-box {
        background: #f8fafc;
        border-radius: 8px;
        padding: 12px;
        margin: 8px 0;
        border-left: 3px solid #00A3E0;
        font-size: 12px;
    }
    .model-category {
        font-size: 10px;
        color: #00A3E0;
        font-weight: 600;
        letter-spacing: 0.5px;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Create display names with icons
    model_display_names = {
        name: f"{info.get('icon', '🔬')} {name}" 
        for name, info in MODEL_ZOO.items()
    }
    
    selected_model = st.selectbox(
        "**Select Model**", 
        list(MODEL_ZOO.keys()),
        format_func=lambda x: model_display_names[x]
    )
    model_info = MODEL_ZOO[selected_model]
    model_info['name'] = selected_model
    
    # Display model context
    with st.expander("ℹ️ Model Details", expanded=False):
        st.markdown(f"**Category:** {model_info.get('category', 'Medical Imaging')}")
        st.markdown(f"**Description:** {model_info['description']}")
        st.info(f"💼 **Business Use:** {model_info.get('business_use', 'Medical image analysis')}")
        st.caption(f"🔧 *Technical:* {model_info.get('technical_desc', '')}")
        
        if 'clinical_applications' in model_info:
            st.markdown("**Clinical Applications:**")
            for app in model_info['clinical_applications']:
                st.markdown(f"• {app}")

    # Device to be used for training
    # (disable if no GPU detected)
    device_selection_disabled=True
    device_index=2
    if torch.cuda.is_available():
        device_selection_disabled=False
        device_index=0
    else:
        warning(TRAIN_CONSOLE_LOG_KEY,
                "No GPU detected! Device selection disabled")
    selected_device = st.radio(label="**Select device**", 
                               options=['Auto', 'GPU', 'CPU'],
                               index=device_index,
                               horizontal=True,
                               help="Choose device for current workload",
                               disabled=device_selection_disabled,
    )

    # Setup the training device based on user selection
    if selected_device == "Auto":
        # Auto select the training device based on device availability
        selected_device = "GPU" if torch.cuda.is_available() else "CPU"
    if selected_device == "GPU":
        training_device = torch.device("cuda")
    else:
        training_device = torch.device("cpu")
    session_state_set("training_device", training_device)
    session_state_set("training_device_type", selected_device)

    # Hyperparameter tuning with enhanced UI
    with st.expander("**⚙️ Hyperparameters**", expanded=True):
        params = {}
        for pname, pdef in model_info["params"].items():
            help_text = pdef.get("help", "")
            if pdef["type"] == "slider":
                params[pname] = st.slider(
                    pname.replace("_", " ").title(),
                    min_value=pdef["min"], 
                    max_value=pdef["max"],
                    step=pdef["step"], 
                    value=pdef["default"], 
                    format="%.5f" if "rate" in pname else "%d",
                    help=help_text
                )
            elif pdef["type"] == "selectbox":
                params[pname] = st.selectbox(
                    pname.replace("_", " ").title(),
                    options=pdef["options"],
                    index=pdef["options"].index(pdef["default"]) if pdef["default"] in pdef["options"] else 0,
                    help=help_text
                )

    # Squirrel away the selected model and hyperparameters
    session_state_set('monai_model', selected_model)
    session_state_set('monai_model_info', model_info)
    for k, v in params.items():
        session_state_set(f"monai_training_{k}", v)

