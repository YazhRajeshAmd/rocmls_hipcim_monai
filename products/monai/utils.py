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

import os
import json
import torch
from datetime import datetime
from components.state import session_state_get

def save_model_bundle_config(data, path, fmt=None, save_both=False):
    """
    Save config dict to JSON or YAML file(s), auto-creating directories.
    If fmt is not specified, it's determined by file extension.
    If save_both=True, saves to both JSON and YAML.
    Requires pyyaml if using YAML.
    """
    def save_json(d, filename):
        # Auto-create directories if they don't exist
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        with open(filename, "w") as f:
            json.dump(d, f, indent=2)

    def save_yaml(d, filename):
        try:
            import yaml
        except ImportError:
            raise ImportError("pyyaml is required for saving YAML files. Install via 'pip install pyyaml'.")
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        with open(filename, "w") as f:
            yaml.safe_dump(d, f, sort_keys=False)

    base, ext = os.path.splitext(path)
    ext = ext.lower()
    files_saved = []

    if save_both:
        json_path = base + ".json"
        yaml_path = base + ".yaml"
        save_json(data, json_path)
        save_yaml(data, yaml_path)
        files_saved = [json_path, yaml_path]
        return files_saved

    if fmt:
        fmt = fmt.lower()
        if fmt == "json":
            save_json(data, path)
            files_saved = [path]
        elif fmt in ("yaml", "yml"):
            save_yaml(data, path)
            files_saved = [path]
        else:
            raise ValueError(f"Unknown fmt: {fmt}")
    else:
        if ext == ".json":
            save_json(data, path)
            files_saved = [path]
        elif ext in (".yaml", ".yml"):
            save_yaml(data, path)
            files_saved = [path]
        else:
            raise ValueError("Unknown format: use .json, .yaml, .yml, or provide fmt='json' or 'yaml'.")

    return files_saved

# Example usage:
# config = {"a": 1, "b": 2}
# save_config(config, "foo/bar/baz.json")
# save_config(config, "foo/bar/config.yaml")
# save_config(config, "foo/bar/conf", save_both=True)

def load_model_bundle_config(path):
    """
    Load a config dictionary from a JSON or YAML file.
    Detects format by extension. Requires pyyaml for YAML files.
    """
    ext = os.path.splitext(path)[1].lower()
    if ext == ".json":
        with open(path, "r") as f:
            return json.load(f)
    elif ext in (".yaml", ".yml"):
        try:
            import yaml
        except ImportError:
            raise ImportError("pyyaml is required for loading YAML files. Install via 'pip install pyyaml'.")
        with open(path, "r") as f:
            return yaml.safe_load(f)
    else:
        raise ValueError("Unknown extension: use .json, .yaml, or .yml.")

# Example usage:
# config = load_config("foo/bar/baz.json")
# config = load_config("foo/bar/baz.yaml")

def save_monai_bundle(
    network, pre_transforms, post_transforms, inferer, metadata_base,
):
    """
    Save MONAI model and inference bundle after training.
    """
    # Get the model id from session state
    model_key = session_state_get('monai_model_info')['id']
    device_type = session_state_get('training_device_type')
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    bundle_base = os.path.join("trained_models", 
                               model_key, 
                               device_type.lower(), 
                               timestamp)
    os.makedirs(os.path.join(bundle_base, "configs"), exist_ok=True)
    os.makedirs(os.path.join(bundle_base, "models"), exist_ok=True)

    # 1. Save model weights
    torch.save(network.state_dict(), os.path.join(bundle_base, "models", "model.pt"))

    # 2. Save inference config (network class, transforms, inferer)
    inference_config = {
        "network": network.__class__.__name__,
        "preprocessing": str(pre_transforms),
        "postprocessing": str(post_transforms),
        "inferer": str(inferer)
    }

    # Save config as JSON
    save_model_bundle_config(inference_config, os.path.join(bundle_base, "configs", "inference.json"))

    # 3. Save metadata (extend with run info as needed)
    save_model_bundle_config(metadata_base, os.path.join(bundle_base, "metadata.json"))

    return bundle_base  # For UI/logging/reference

