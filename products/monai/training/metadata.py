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
import pandas as pd
from components.diagnostics import warning
from components.io import render_markdown
from components.utility import filesize_h
from components.diagnostics import error, warning, note, info
from components.state import session_state_get
from products.monai.styles import TRAINING_METADATA_TAB_CSS, TRAINING_METADATA_TAB_HTML

# Display model metadata
def display_metadata():
    # Extract metadata on the selected model from the session state
    selected_model = session_state_get('monai_model')
    model_info = session_state_get('monai_model_info')
    model_id = model_info["id"]  # e.g. spleen_ct_seg
    model_url = model_info.get("url", "#")
    model_description = model_info.get("description", 
                                       "N/A")
    model_dataset = model_info.get("dataset", "N/A")
    model_dataurl = model_info.get("dataurl", "#")

    params = {}
    for pname in model_info["params"]:
        params[pname] = session_state_get(f"monai_training_{pname}")

    # Display formatted metadata
    st.markdown(TRAINING_METADATA_TAB_CSS, unsafe_allow_html=True)
    st.markdown(
        TRAINING_METADATA_TAB_HTML.format(
            model_name = f"{selected_model}",
            description = f"{model_description}",
            url = f"{model_url}",
            dataset = f"{model_dataset}",
            dataurl = f"{model_dataurl}",
        ),
        unsafe_allow_html=True
    )

