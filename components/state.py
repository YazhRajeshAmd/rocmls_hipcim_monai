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
# 
# Centralized session state management
#

#
# Example Usage:
#
#   from components.state import (
#       session_state_init,
#       session_state_get,
#       session_state_set,
#       session_state_append_times,
#       session_state_reset_tiles,
#       session_state_set_spacing_units,
#       session_state_increment_op_count
#   )
#   
#   # Initialize all state variables (call early, e.g., in main or at top of tab)
#   session_state_init()
#   
#   # Set and get specific state variables:
#   session_state_set('selected_wsi', selected_wsi_filepath)
#   current_wsi = session_state_get('selected_wsi')
#   
#   # Timing/operation tracking:
#   session_state_append_times('gpu_times', gpu_time)
#   session_state_increment_op_count()
#   
#   # Metadata updating:
#   session_state_set_spacing_units(wsi.metadata, utility.unit_map)
#   
#   # Tile handling reset:
#   session_state_reset_tiles()

import streamlit as st
from components.console_log import console_log_init

# Centralized session state defaults for the AMD ROCm-LS demo.
_SESSION_STATE_DEFAULTS = {
    'gpu_times': [],
    'cpu_times': [],
    'spacing_x': None,
    'spacing_y': None,
    'unit_x': None,
    'unit_y': None,
    'gpu_tile': None,
    'cpu_tile': None,
    'selected_wsi': None,           # Path of currently selected WSI
    'cuImage': None,                # Currently selected WSI as CuImage
    'wsi_width': None,
    'wsi_height': None,
    'tile_size': 128,
    'position_x': 0,
    'position_y': 0,
    'tile_max_dimension': 16384,    # Maximum allowed tile dimension
    'tile_display_size': 400,       # Dimension of individual tiles to be rendered
    'pipeline': [],               # Pipeline for hipCIM # TODO Fix namespace to be hipCIM specific
    # Add other keys here as needed for future scalability
}

def session_state_init(**overrides):
    """
    Initialize all required session state variables with sensible defaults.

    Args:
        overrides: Any key-value pairs to override initial values.
    Call this at the start of your script or before you need state variables.
    """
    for key, default in _SESSION_STATE_DEFAULTS.items():
        if key not in st.session_state:
            st.session_state[key] = overrides.get(key, default)

def session_state_get(key, default=None):
    """
    Safely get a value from session state.

    Args:
        key: The session state variable to get.
        default: Value to return if key is missing.
    """
    return st.session_state.get(key, default)

def session_state_set(key, value):
    """
    Safely set a value in session state.

    Args:
        key: The session state variable to set.
        value: The value to set.
    """
    st.session_state[key] = value

def session_state_append_times(timer_key, timer_value):
    """
    Append timing value (e.g., for GPU, CPU op durations) to the appropriate session list.

    Args:
        timer_key: 'gpu_times' or 'cpu_times'
        timer_value: Numeric duration (float) to append.
    """
    if timer_key not in st.session_state:
        st.session_state[timer_key] = []
    st.session_state[timer_key].append(timer_value)

def session_state_reset_tiles():
    """
    Reset GPU/CPU tile references (if switching slides, for example).
    """
    st.session_state['gpu_tile'] = None
    st.session_state['cpu_tile'] = None

def session_state_set_spacing_units(metadata, unit_map):
    """
    Set image spacing and units in session state based on WSI metadata.

    Args:
        metadata: The metadata dictionary from WSI (CuImage, etc.)
        unit_map: A mapping dict from unit names to display strings.
    """
    st.session_state['spacing_x'] = metadata['cucim']['spacing'][1]   # spacing[1] is X
    st.session_state['spacing_y'] = metadata['cucim']['spacing'][0]   # spacing[0] is Y
    st.session_state['unit_x'] = unit_map.get(metadata['cucim']['spacing_units'][1])
    st.session_state['unit_y'] = unit_map.get(metadata['cucim']['spacing_units'][0])

