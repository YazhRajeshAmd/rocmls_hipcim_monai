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
import os

from components.diagnostics import exception, error, warning, info, note
from components.state import session_state_get

# Render the specified markdown file
def render_markdown(md_file=None):
    if not md_file:
        md_file = "markdown/default.md"

    # Render the contents of the markdown
    try:
        with open(md_file, 'r') as md_handle:
            content = md_handle.read()
        st.markdown(content, unsafe_allow_html=True)
    except Exception as e:
        exception(session_state_get('current_console_log_key'), 
                  f"failed to render {md_file}: {e}")

def file_upload_widget(caption="Select image", 
                       allowed_types=["svs", "tif", "tiff"], 
                       wsi_dir="sample_images"):
    if not os.path.exists(wsi_dir):
        warning(session_state_get('current_console_log_key'), 
                f"Directory '{wsi_dir}' not found!")
        return None, None

    files = sorted([f for f in os.listdir(wsi_dir) if f.lower().endswith(tuple(allowed_types))])
    if not files:
        warning(session_state_get('current_console_log_key'), 
                f"No supported images found in '{wsi_dir}'.")
        return None, None

    selected_file = st.selectbox(caption, files)
    if selected_file:
        file_path = os.path.join(wsi_dir, selected_file)
        return selected_file, file_path
    else:
        return None, None

def save_uploaded_file(uploaded_file, directory):
    if not os.path.exists(directory):
        os.makedirs(directory)
    file_path = os.path.join(directory, uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return file_path
