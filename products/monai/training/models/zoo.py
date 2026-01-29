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
# List of MONAI models from the Model Zoo supported for demo
# Enhanced with business context and technical explanations
#

MODEL_ZOO = {
    "Spleen CT Segmentation": {
        "id": "spleen_ct_seg",
        "category": "3D Medical Imaging",
        "icon": "🫁",
        "description": "A 3D segmentation model for spleen delineation in CT images. The model processes 96x96x96 pixel patches and provides segmentation masks for spleen tissue.",
        "business_use": "Automate organ segmentation for radiation therapy planning, surgical navigation, and volumetric analysis. Reduces manual contouring time from hours to seconds.",
        "technical_desc": "U-Net based 3D CNN architecture with residual connections. Trained on volumetric CT data using Dice loss optimization for precise boundary delineation.",
        "clinical_applications": [
            "Radiation oncology treatment planning",
            "Surgical pre-operative assessment",
            "Organ volumetry for disease monitoring",
            "Trauma assessment in emergency radiology"
        ],
        "url": "https://monai.io/model-zoo.html#/model/spleen_ct_segmentation",
        "dataset": "Medical Segmentation Decathlon (MSD) - Spleen",
        "dataurl": "https://medicaldecathlon.com/dataaws/",
        "params": {
            "learning_rate": {
                "type": "slider", "min": 1e-5, "max": 3e-3, "step": 1e-5, "default": 1e-4,
                "help": "Controls model weight updates. Lower values (1e-5) for fine-tuning, higher (1e-3) for faster initial training."
            },
            "batch_size": {
                "type": "slider", "min": 1, "max": 8, "step": 1, "default": 2,
                "help": "Number of 3D volumes processed per iteration. Limited by GPU memory for volumetric data."
            },
            "epochs": {
                "type": "slider", "min": 1, "max": 100, "step": 1, "default": 10,
                "help": "Complete passes through training data. More epochs improve accuracy but increase training time."
            },
        },
    },

    "Pathology Tumor Detection": {
        "id": "pathology_tumor_detection",
        "category": "Digital Pathology",
        "icon": "🔬",
        "description": "A deep learning model for detecting metastatic tissue in whole-slide pathology images. The model processes 224x224 pixel RGB patches and provides probability scores for metastasis detection. Trained on the Camelyon16 dataset.",
        "business_use": "Assist pathologists in detecting cancer metastases in lymph node biopsies. Improves diagnostic accuracy and reduces review time for high-volume pathology labs.",
        "technical_desc": "ResNet-based classification network with transfer learning. Processes gigapixel WSIs using patch-based inference with probability heatmap generation.",
        "clinical_applications": [
            "Breast cancer lymph node metastasis detection",
            "Pathologist workflow prioritization",
            "Quality assurance for missed diagnoses",
            "Research cohort screening"
        ],
        "url": "https://monai.io/model-zoo.html#/model/pathology_tumor_detection",
        "dataset": "CAncer MEtastases in LYmph nOdes challeNge (CAMELYON)",
        "dataurl": "https://registry.opendata.aws/camelyon/",
        "params": {  
            "learning_rate": {
                "type": "slider", "min": 1e-5, "max": 3e-3, "step": 1e-5, "default": 1e-4,
                "help": "Optimizer step size. Use smaller values when fine-tuning pre-trained weights."
            },
            "batch_size": {
                "type": "slider", "min": 1, "max": 100, "step": 1, "default": 32,
                "help": "Patches per GPU iteration. Larger batches improve training stability for classification tasks."
            },
            "epochs": {
                "type": "slider", "min": 1, "max": 100, "step": 1, "default": 2,
                "help": "Training iterations over full dataset. Pathology models often converge quickly with transfer learning."
            },
            "backend": {
                "type": "selectbox", "options": ["cucim", "numpy"], "default": "cucim",
                "help": "Image loading backend. cuCIM provides GPU-accelerated WSI reading for faster data loading."
            },
            "grid_shape": {
                "type": "slider", "min": 1, "max": 8, "step": 1, "default": 3,
                "help": "Patch extraction grid dimensions. Higher values extract more patches per slide region."
            },
            "patch_size": {
                "type":"slider", "min": 200, "max": 224, "step": 1, "default": 224,
                "help": "Input patch dimensions (pixels). 224x224 matches standard ImageNet pre-training."
            },
            "prob": {
                "type":"slider", "min": 0.1, "max": 1.0, "step": .01, "default": 0.5,
                "help": "Random patch sampling probability. Balances coverage vs. training speed."
            },
            "gpu": {
                "type": "slider", "min": 0, "max": 7, "step": 1, "default": 0,
                "help": "Target GPU device index for multi-GPU systems."
            },
        },
    },

    # Add more models here
}
