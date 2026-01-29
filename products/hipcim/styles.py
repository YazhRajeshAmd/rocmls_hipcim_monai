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
# Style strings for various hipCIM customizations
#

# Compact sidebar with enhanced styling
COMPACT_SIDEBAR_CSS = """
<style>
    /* Expander header styling */
    .streamlit-expanderHeader {
        font-size: 14px;
        font-weight: 600;
        background: linear-gradient(90deg, #f8f9fa 0%, #ffffff 100%);
        border-radius: 8px;
        padding: 8px 12px !important;
    }
    
    /* Slider improvements */
    .compact-slider label {
        font-size: 13px;
        color: #374151;
    }
    .compact-slider .stSlider {
        padding-top: 4px !important;
        padding-bottom: 4px !important;
    }
    
    /* Control panel section */
    .control-section {
        background: #f8fafc;
        border-radius: 10px;
        padding: 12px;
        margin-bottom: 12px;
        border: 1px solid #e2e8f0;
    }
    
    /* Slider track styling - AMD Teal */
    .stSlider > div > div > div {
        background: linear-gradient(90deg, #00A3E0, #00c4ff) !important;
    }
    
    /* Button improvements */
    .stButton > button {
        border-radius: 8px;
        font-weight: 500;
        transition: all 0.2s ease;
    }
    .stButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(0, 163, 224, 0.3);
    }
    
    /* Selectbox styling */
    .stSelectbox > div > div {
        border-radius: 8px;
    }
    
    /* Info/caption text */
    .stCaption {
        font-size: 11px;
        color: #6b7280;
        background: #f1f5f9;
        padding: 4px 8px;
        border-radius: 4px;
        margin-top: 8px;
    }
</style>
"""

