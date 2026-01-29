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
import datetime
import traceback
import inspect
import os

from components.console_log import append_console_log

# Debug flag: set to True to enable debug messages globally
DEBUG_ENABLED = False  # Set to True for verbose diagnostics

# Mapping from log level to color class for Streamlit HTML
LEVEL_TO_SPAN = {
    "ERROR": "<span class='log-error'>error:</span>",
    "EXCEPTION": "<span class='log-error'>exception:</span>",
    "WARNING": "<span class='log-warning'>warning:</span>",
    "INFO": "<span class='log-info'>info:</span>",
    "NOTE": "<span class='log-note'>note:</span>",
    "DEBUG": "<span class='log-debug'>debug:</span>",
}

def _now_str():
    """Return the current timestamp as a string."""
    return datetime.datetime.now().strftime("%H:%M:%S")

def _caller_context(skip=3):
    """
    Return a string with (file basename):(line no.) (function name)
    skip=2 means: skip get_caller_info + the logging function (error, warning, etc.) to get real caller.
    """
    stack = inspect.stack()
    if len(stack) > skip:
        frame = stack[skip]
        filename = os.path.basename(frame.filename)
        return f"{filename}:{frame.lineno} {frame.function}"
    return "<unknown file>:0 <unknown function>"

def _format_message(level, msg):
    """
    Format the main log line for all diagnostics.
    Structure: (timestamp): (diagnostic class -- colorized): (file):(line) (function): (diagnostic message)
    """
    color_tag = LEVEL_TO_SPAN.get(level, "")
    # Tag is included inline, with actual level name (e.g., "error:") colorized
    return f"{_now_str()} {color_tag} {_caller_context()} : {msg}"

def exception(console_log_key, e):
    """
    Log an exception with colorized tag, plus stack trace as plain text on the next lines.
    """
    level = "EXCEPTION"
    msg = f"{e!r}"
    header = _format_message(level, msg)
    # Get current full exception traceback
    tb = traceback.format_exc().strip()
    if not tb or tb == "NoneType: None":
        tb = ''.join(traceback.format_stack())
    log_entry = f"{header}\n{tb}"
    append_console_log(console_log_key, log_entry)

def error(console_log_key, msg):
    """
    Log an error with colorized tag and append stack trace as plain text.
    """
    level = "ERROR"
    header = _format_message(level, msg)
    stack = ''.join(traceback.format_stack()[:-1]).strip()
    log_entry = f"{header}\n{stack if stack else '(stack trace not available)'}"
    append_console_log(console_log_key, log_entry)

def warning(console_log_key, msg):
    """
    Log a warning with colorized label.
    """
    level = "WARNING"
    log_entry = _format_message(level, msg)
    append_console_log(console_log_key, log_entry)

def info(console_log_key, msg):
    """
    Log info with colorized label.
    """
    level = "INFO"
    log_entry = _format_message(level, msg)
    append_console_log(console_log_key, log_entry)

def note(console_log_key, msg):
    """
    Log a note with colorized label.
    """
    level = "NOTE"
    log_entry = _format_message(level, msg)
    append_console_log(console_log_key, log_entry)

def debug(console_log_key, msg):
    """
    Log a debug message only if debug is enabled, with colorized label.
    """
    if DEBUG_ENABLED:
        level = "DEBUG"
        log_entry = _format_message(level, msg)
        append_console_log(console_log_key, log_entry)

# Example usage:
# error("MY_LOG", "Something went wrong!")
# warning("MY_LOG", "This might be a problem.")
# info("MY_LOG", "Process started.")
# note("MY_LOG", "Just a heads up.")
# debug("MY_LOG", "This is verbose debug info.")
# try:
#     1 / 0
# except Exception as e:
#     exception("MY_LOG", e)
