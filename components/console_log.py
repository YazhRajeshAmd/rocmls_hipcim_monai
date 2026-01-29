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
# Centralized console logging mechanism
#

import streamlit as st
import re

from components.styles import CONSOLE_LOG_CSS, CONSOLE_LOG_HTML

def console_log_init(console_log_key):
    st.session_state['current_console_log_key'] = console_log_key
    if console_log_key not in st.session_state:
        st.session_state[console_log_key] = []

def strip_span_tags(text):
    # Remove <span ...> and </span>
    text = re.sub(r"<span[^>]*>", "", text)
    text = re.sub(r"</span>", "", text)

    # Compact multiple spaces to a single space
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def append_console_log(console_log_key, line):
    """
    Appends a log line with the given text (should already contain any tags/timestamp).
    """
    if console_log_key not in st.session_state:
        st.session_state[console_log_key] = []
    st.session_state[console_log_key].append(line)
    print(strip_span_tags(line))

def clear_console_log(console_log_key):
    """
    Clears the log.
    """
    st.session_state[console_log_key] = []

def console_log_view(console_log_key, log_lines=None):
    st.markdown(CONSOLE_LOG_CSS, unsafe_allow_html=True)

    if log_lines is None:
        log_lines = st.session_state.get(console_log_key, [])

    if not log_lines:
        log_lines = ["(console logs will appear here...)"]

    log_lines = [str(l) for l in log_lines]
    log_content = "<br>".join(log_lines)
    autoscroll_script = """
    <script>
    setTimeout(function() {
        var logBox = document.querySelector('.console-output');
        if (logBox) {
            logBox.scrollTop = logBox.scrollHeight;
        }
    }, 50);
    </script>
    """
    st.markdown(
        CONSOLE_LOG_HTML.format(content=log_content) + autoscroll_script,
        unsafe_allow_html=True
    )

    # Buttons to download and clear the log
    st.markdown('<div class="console-buttons">', unsafe_allow_html=True)
    col1, col2 = st.columns([1, 1])
    with col1:
        if st.button("Clear", key=console_log_key+"_clear"):
            st.session_state[console_log_key] = []
    with col2:
        log_text = "\n".join(log_lines)

        # Unicode for download symbol (U+2B07)
        # You can also use: "\u2B07" or "&#11015;"
        DOWNLOAD_ICON = "\u2B07"

        # Download button with custom icon
        st.download_button(
            key=console_log_key+"_download",
            label=f"{DOWNLOAD_ICON}",
            data=log_text,
            file_name="console_log.txt",
            mime="text/plain",
            width="stretch",
            help="Download Console Log",      # Optional extra tooltip for accessibility
            disabled=False,
        )
    st.markdown('</div>', unsafe_allow_html=True)

