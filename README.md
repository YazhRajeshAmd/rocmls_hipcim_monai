# AMD ROCm-LS Demo Application

___Accelerated Imaging & AI Workflows for Life Sciences___
___Featuring hipCIM and MONAI on AMD Instinct GPUs___


## Overview
This interactive Streamlit application showcases the capabilities of 
**hipCIM** and **MONAI** - two ROCm-LS-optimized life sciences platforms 
for AMD Instinct GPUs. The demo is modular and easily extensible. 

### hipCIM

**Note**
_hipCIM is in an early access state. Running production workloads is not recommended._

hipCIM (HIP Computational Imaging and Microscopy) is an advanced GPU-accelerated 
library designed to unleash the full power of AMD Instinct GPUs for large-scale 
biomedical image analysis. As part of the ROCm-LS Life Sciences software suite, 
hipCIM brings highly-optimized implementations of core imaging algorithms, such 
as denoising, stain separation, registration, segmentation, and geometric 
transformations, into everyday workflows for digital pathology, microscopy, 
and medical imaging research. By leveraging the HIP programming model, hipCIM 
ensures seamless performance, scalability, and integration for Python-based 
life sciences toolchains, eliminating bottlenecks and enabling rapid, 
interactive exploration of massive datasets.

What sets hipCIM apart is its commitment to open standards, modularity, and 
scientific transparency. Researchers and developers can easily extend hipCIM 
with their own custom algorithms, build robust pipelines, and benchmark GPU 
performance side-by-side with CPU implementations, all within a streamlined, 
container-friendly workflow. With hipCIM, AMD delivers not just speed and 
accessibility to life sciences innovators, but also a collaborative foundation 
for reproducible, next-generation biomedical imaging solutions.

### MONAI

**Note**
_MONAI is in an early access state. Running production workloads is not recommended._

MONAI (Medical Open Network for AI) is the leading open-source, PyTorch-based 
framework purpose-built for accelerating deep learning innovation in healthcare 
imaging. Designed through close collaboration with clinicians, researchers, and 
the medical AI community, MONAI offers robust tools for training, validating, 
and deploying state-of-the-art models across domains such as radiology, pathology, 
and digital microscopy. Its domain-specialized workflows, extensive model zoo, 
and rich support for imaging data formats empower rapid prototyping and reproducible 
research, breaking down barriers between AI development and real clinical impact.

With native support for AMD ROCm and Instinct GPUs available as part of the ROCm-LS 
Life Sciences suite, MONAI now enables fully open, high-performance AI pipelines on 
the latest AMD hardware. Users can seamlessly leverage advanced GPU acceleration, 
distributed training, and a thriving community ecosystem, all while avoiding vendor 
lock-in and proprietary constraints. MONAI provides not just the building blocks for 
tomorrow's medical AI breakthroughs, but also the foundation for transparent, 
collaborative, and scalable healthcare innovation.

___New ROCm-LS products/projects will be added to the demo as they become 
available.___

___Current Product Coverage:___

- **hipCIM**: GPU-accelerated computational imaging and microscopy
- **MONAI**: Medical AI model training (inference coming soon)

## Key Features
### ___Modular Panel-Based UI:___

Select and experiment with hipCIM and MONAI workflows.

### ___Session State, Logging, and Diagnostics:___

Robust experiment state, color-coded logs, and error reporting.

### ___Ready for Contributors:___

All code is structured for easy extension. Templates guide you in adding new models, metrics, or product panels.

___Early Access Note:___

Both hipCIM and MONAI are early access releases. The demo and supported features evolve as ROCm-LS progresses.

## Demo Architecture

```
.
|-- supervisord.conf            # Process management config
|-- run_amd                     # Utility script for setup/downloads
|-- setup.sh                    # Environment setup script
|-- components/                 # Shared UI and utility components
|   |-- [state management, logging, diagnostics, styling, utilities]
|-- markdown/                   # Text guides and contextual markdown
|-- products/                   # Product-specific implementations
|   |-- hipcim/                # hipCIM module
|   |-- monai/                 # MONAI module
|-- configs/                    # Configuration files
|-- demo/                       # Demo data and resources
|   |-- custom_trained/        # Pre-trained models
|   |-- data/                  # Training/validation datasets
|-- sample_images/              # Sample image files
|   |-- camelyon/              # Camelyon dataset images
|-- trained_models/             # Output directory for trained models
|-- use-cases/                  # Jupyter notebooks and examples
    |-- hipcim/                # hipCIM use case notebooks
    |-- monai/                 # MONAI use case notebooks
```

### Architecture Principles

- **Modularity**: Each product (hipCIM, MONAI) is self-contained and extensible
- **Separation of Concerns**: UI components, business logic, and data are cleanly separated
- **Extensibility**: New products can be added by creating subfolders under `products/`
- **State Management**: Centralized session state handling via `components/state.py`
- **Reproducibility**: Model bundles and configs ensure reproducible workflows

## Getting Started

### Prerequisites
- AMD Instinct GPUs (MI300 and above)
- ROCm 6.4+ (see ROCm-LS GitHub for installation guides)
- Docker (recommended) with CLI access to ROCm
- Python 3.10+

### Dataset preparation

#### Pathology Tumor detection dataset
To download the camelyon data, follow the instructions in this link https://registry.opendata.aws/camelyon/ and place tumor_001.tif and tumor_101.tif in sample_images/camelyon folder. This dataset is needed to train the pathology_tumor_detection model. To run inference on the pathology tumor detection model, copy tumor_001.tif to demo/data/pathology_tumor_detection/

#### Spleen CT Segmentation

Download the Spleen CT dataset:

```bash
# Create the demo data directory
mkdir -p demo/data/spleen_ct_segmentation/

# Download and extract only spleen_*.nii.gz files directly to destination
wget -qO- https://msd-for-monai.s3-us-west-2.amazonaws.com/Task09_Spleen.tar | \
tar -xf - --strip-components=2 --wildcards -C demo/data/spleen_ct_segmentation/ 'Task09_Spleen/imagesTs/spleen_*.nii.gz'
```

This dataset is needed for inference with the spleen CT segmentation model.

#### Data for the notebook
Run ./run_amd download_testdata to download the data in the correct folder

### Example code
An example with hipCIM and MONAI for AMD on AMD GPU can be found under the /use-cases/monai/sample folder

### Installation

```
git clone https://github.com/ROCm-LS/examples
cd examples
```

#### Bare metal

```
export HIP_PATH=/opt/rocm
export PATH=$HIP_PATH/bin:$PATH
export ROCM_PATH=/opt/rocm
export LD_LIBRARY_PATH=$HIP_PATH/lib:$LD_LIBRARY_PATH
export ROCM_HOME=/opt/rocm

apt update && \
    apt install -y software-properties-common lsb-release gnupg && \
    apt-key adv --fetch-keys https://apt.kitware.com/keys/kitware-archive-latest.asc && \
    add-apt-repository -y "deb https://apt.kitware.com/ubuntu/ $(lsb_release -cs) main" && \
    apt update && \
    apt install -y git wget gcc g++ ninja-build git-lfs       \
                    yasm libopenslide-dev python3.10-venv        \
                    cmake rocjpeg rocjpeg-dev rocthrust-dev      \
                    hipcub hipblas hipblas-dev hipfft hipsparse  \
                    hiprand rocsolver rocrand-dev rocm-hip-sdk libvips supervisor

python -m pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
streamlit run rocm-ls_demo.py
```

#### Docker

##### `Dockerfile`
- **Purpose:** Automates the creation of a reproducible, containerized environment for the ROCm-LS demo.
- **Features:**
  - Uses an appropriate Ubuntu/ROCm development image.
  - Installs all system and Python dependencies.
  - Secures source code checkout using an SSH private key ARG.
  - Sets up a Python virtual environment for isolation.
  - Uses `supervisord` to launch the Streamlit dashboard and ensure automatic restarts for reliability.

##### `supervisord.conf`
- **Purpose:** Process manager configuration to keep Streamlit running inside the container.
- **Features:**
  - Launches Streamlit in the correct working directory for the app.
  - Auto-restarts the app on any failure.
  - Directs all logs to STDOUT/STDERR for easy access with `docker logs`.

##### Usage

###### 1. **Build the Docker Image**

```bash
docker build --no-cache -t examples-docker .
```

###### 2. Run the Docker Container
```bash
docker run --cap-add=SYS_PTRACE --ipc=host --privileged=true --shm-size=128GB --network=host --device=/dev/kfd --device=/dev/dri --group-add video -it examples-docker
```

Adjust device flags as needed for your system.
This command enables persistent and robust serving of the demo over the local network.

###### 3. View Application Logs

```bash
docker logs -f examples-docker
```

This will stream the supervisord/Streamlit logs for monitoring and debugging.

###### 4. Stopping and Removing the Demo

```bash
docker stop examples-docker
docker rm examples-docker
```

##### Access

By default, the Streamlit app will be available via http://localhost:8501 on your server (or [hostname]:8501 across your network).

##### Notes

- **Security:** Do not leave your SSH private key in the build context or image. The recommended `Dockerfile` clears it after use.
- **Persistence:** For persistent demo use, the container is configured to auto-restart on failure or system reboot.
- **GPU Support:** Both `/dev/kfd` and `/dev/dri` are passed for full AMD GPU access and graphics.

##### Troubleshooting

- Check logs (`docker logs -f examples-docker`) for any errors or crash reports.
- Ensure your system has ROCm-compatible AMD GPUs, proper drivers, and Docker configured to support device and group flags.
- If the app does not start, double-check the device permissions and network/port availability.

### GPU Requirements

Requires AMD Instinct MI300 or newer. For details, see ROCm-LS Hardware Requirements.

## Usage Guide

### General Flow

1. Start the demo and select between hipCIM and MONAI tabs.
2. Each tab provides:
  - **Sidebar:** main controls (image/model selection, parameters)
  - **Main panel:** workflow visualization, model stats or tile displays
  - **Right sidebar:** context, log panel, hardware/package status

### hipCIM Workflows

- Select or upload a WSI image
- Extract tiles at configurable size/XY position
- Build transformation pipeline
- Add/reorder/remove steps (stain separation, edge detection, morph operations, rotation, warp)
- View reference, CPU, and GPU results side-by-side
- Analyze live performance benchmarking

### MONAI Workflows

Note: _Only training demo is implemented as of now. MONAI inference panel is a work in progress._

- **Model Selection:** Pick from Model Zoo (e.g., Spleen CT Segmentation)
  - Model zoo metadata is displayed on selection.
- **Hyperparameter Tuning:** Use sidebar sliders for learning rate, batch size, epochs, etc.
- **Training Driver:** Launch training and view real-time progress with live graphs
- **Result Visualization:** Contour overlays, loss curves, dice, and sample images per epoch
- **Placeholder Tab:** Reserved for future visualizations (see "Next Steps" below)

### MONAI Model Bundle Saving and Reproducible Inference
The demo supports saving each trained MONAI model as a standardized MONAI
bundle under `trained_models/{model_key}/{device_type}/{timestamp}`. 

Each bundle encapsulates:

- The trained model's weights (`model.pt`)
- The complete preprocessing and postprocessing transformation pipelines
- The inference handler (e.g., sliding window inferer settings)
- Human-readable metadata, including model version, authorship, label map, and dependencies
- All configurations are exported in both JSON and YAML for compatibility

This bundle format ensures that models trained within the demo can be easily
loaded and used for future inference demonstrations in a fully reproducible
way, even across upgrades or different environments.

Implementation guidelines for contributors:

- Every new MONAI training script should implement a `get_bundle_components(params)` function.
  - This function must return (`pre_transforms`, `post_transforms`, `inferer`) describing how to preprocess inputs, postprocess outputs, and perform inference with the
saved model.
- The driver automates bundle export using these components post-training, ensuring consistency and extensibility.
- When adding new models, refer to existing templates for the required structure.

How the bundle is used:

- The inference demo can load the saved bundle, rehydrate the model and transforms, and run predictions on new data, enabling one-click transition from training to deployment.

## Extending the Demo (Contributors Guide)

### Adding New ROCm-LS Products/Projects

- Create a new subfolder under `products/`
- Follow the separation of logic/console/layout/sidebar/main/metadata modules
- Register a new tab in `rocm-ls_demo.py` for your product

### Adding/Extending hipCIM Transforms

- Modify `products/hipcim/transforms.py` to add new transformation functions
- Update `OP_CATALOG` and `ICON_MAP` in `products/hipcim/sidebar.py` for new sidebar controls
- Ensure new transforms follow device-agnostic structure (support both GPU and CPU as templates)

### Adding MONAI Models

- Create a new script in `products/monai/training/models/` following the template of `spleen_ct_seg.py`
- Register the model and its parameter set in `products/monai/training/models/zoo.py`
- Ensure API compatibility with training driver (`prepare_training_data`, `train_batch`, `evaluate_batch`, etc.)
- Ensure model works with device set to either `torch("cuda")` or `torch("cpu")`
- Metadata, visualization sample extraction, and loss/metric computation must be handled to populate the dashboard graphs and overlays

### Adding Inference Workflows (see Next Steps)

- Mirror the current MONAI training sidebar structure for model selection and hyperparameters
- Implement inference main logic in a file, in `products/monai/inference/` directory.
- Ensure the interface returns predictions in the format (input, prediction, possibly label) for visualization

### Adding Metrics/Graphs

- Add your graph code into the specific panel (MONAI, hipCIM, etc.)
- For MONAI, you can use the "Placeholder" tab to add system performance graphs (e.g., GPU usage, system stats)

## Known Bugs & Limitations
- Only training is implemented in MONAI panel. Inference is a work in progress.
- The fourth metric graph (rightmost panel in training stats) currently only displays epoch/time; other stats (accuracy, precision, recall) are missing.
- On some systems, MONAI model implementation may fail to recognize the training device as "cuda" even when running on AMD MI300 GPUs.
- UI alignment issue: "Clear" and "Download" buttons under the console log are not properly aligned.
- The console log fails to automatically scroll to the latest entry; manual scrolling is required.

## Performance Insights
- Side-by-side bar graphs show comparative speedups for each hipCIM transformation.
- Hardware/software version and package details are always visible in the sidebar footer.

## References

- **ROCm-LS:** [Intro & Documentation](https://rocm.docs.amd.com/projects/rocm-ls/en/latest/index.html)
- **hipCIM:** [AMD Computational Imaging and Microscopy](https://github.com/ROCm-LS/hipCIM)
- **MONAI:** [Medical Open Network for AI](https://github.com/ROCm-LS/monai)
- **ROCm-LS GitHub Organization:** [https://github.com/ROCm-LS](https://github.com/ROCm-LS)

## Contributors

- Soumitra Chatterjee
- Anik Chaudhuri
- Chandan Sharma
- Prateek Chokse

## License

`Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.`

Licensed under the Apache License, Version 2.0.
