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
# usage: python3 example/monai_hipcim_test.py

import torch
import torchvision.transforms as T
import matplotlib.pyplot as plt
import torch.profiler
from PIL import Image
import numpy as np
from monai.networks.nets import DenseNet121
from hipcim import CuImage
import cupy as cp

wsi_path = "sample_images/sample_image.tif"  # replace with your WSI path
img = CuImage(wsi_path)

height, width = img.shape[:2]
tile = img.read_region(location=(0, 0), size=(img.shape[1], img.shape[0]), level=0, device="cuda")  # Extract the whole image or a region

tile_np = cp.asnumpy(tile)
print("shape=",img.shape)
print("height: ", height)
print("width: ", width)


# # Convert to PIL Image for torchvision transforms and ensure RGB
tile_pil = Image.fromarray(tile_np).convert("RGB")

transform = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=[0.5]*3, std=[0.5]*3)
])

input_tensor = transform(tile_pil).unsqueeze(0)  
print(input_tensor.shape)
print(input_tensor.dtype)
print(type(input_tensor))


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
model = DenseNet121(spatial_dims=2, in_channels=3, out_channels=2)  # Binary classification
model.to(device)
model.eval()
with torch.profiler.profile(
    activities=[
        torch.profiler.ProfilerActivity.CPU,
        torch.profiler.ProfilerActivity.CUDA,
    ],
    schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=1),
    on_trace_ready=torch.profiler.tensorboard_trace_handler('./profiler_logs'),
    record_shapes=True,
    profile_memory=True,
    with_stack=True
) as prof:

    # -----------------------------------------------------------
    # Run Inference
    # -----------------------------------------------------------
    with torch.no_grad():
        for _ in range(5): 
            output = model(input_tensor.to(device))
            pred_class = torch.argmax(output, dim=1).item()
            prof.step()
print(f"Predicted class: {pred_class}")
print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
