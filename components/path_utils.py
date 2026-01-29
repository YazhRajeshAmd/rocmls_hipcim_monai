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
"""
Path utilities for handling relative and absolute paths in ROCm LS examples
"""

import os
from pathlib import Path


def get_project_root() -> Path:
    """Get the project root directory (rocm-ls-examples)"""
    current_file = Path(__file__).resolve()

    # Look for the rocm-ls-examples directory
    for parent in current_file.parents:
        if parent.name == "rocm-ls-examples":
            return parent

    # Fallback - assume we're in the right place
    return Path(__file__).parent.parent


def resolve_path(path_str: str) -> str:
    """
    Convert relative path to absolute path, or return absolute path as-is

    Args:
        path_str: Path string (relative or absolute)

    Returns:
        Absolute path string
    """
    if os.path.isabs(path_str):
        return path_str

    project_root = get_project_root()
    absolute_path = project_root / path_str
    return str(absolute_path)


def to_relative_path(path_str: str) -> str:
    """
    Convert absolute path to relative path from project root

    Args:
        path_str: Path string (absolute or relative)

    Returns:
        Relative path string from project root
    """
    path_obj = Path(path_str)

    if not path_obj.is_absolute():
        return path_str  # Already relative

    project_root = get_project_root()

    try:
        relative_path = path_obj.relative_to(project_root)
        return str(relative_path)
    except ValueError:
        # Path is not relative to project root, return as-is
        return path_str


def ensure_absolute_path(path_str: str) -> str:
    """
    Ensure path is absolute, converting from relative if needed

    Args:
        path_str: Path string

    Returns:
        Absolute path string
    """
    return resolve_path(path_str)


def normalize_path_for_display(path_str: str) -> str:
    """
    Normalize path for display in UI (show relative to project root if possible)

    Args:
        path_str: Path string

    Returns:
        Normalized path for display
    """
    relative = to_relative_path(path_str)

    # If it's a relative path, show it as rocm-ls-examples/...
    if not os.path.isabs(relative):
        return f"rocm-ls-examples/{relative}"

    return path_str