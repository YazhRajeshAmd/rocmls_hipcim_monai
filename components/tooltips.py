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
Professional tooltips for hipCIM UI elements.
Each tooltip provides business context followed by technical details.
"""

tooltips = {
    # ============================================
    # TILE CONTROLS
    # ============================================
    'tile_size': (
        "**Region of Interest Size**\n\n"
        "Controls the dimensions of the tissue region extracted for analysis. "
        "Larger tiles (256-512px) capture more tissue context for architectural analysis, "
        "while smaller tiles (128px) focus on cellular-level details.\n\n"
        "*Technical: Defines the width and height in pixels of the extracted tile from the WSI.*"
    ),

    'x_position': (
        "**Horizontal Position**\n\n"
        "Navigate to specific tissue regions horizontally across the slide. "
        "Use this to locate areas of diagnostic interest such as tumor margins or lymph nodes.\n\n"
        "*Technical: X-coordinate (pixels) for the top-left corner of the extracted tile.*"
    ),

    'y_position': (
        "**Vertical Position**\n\n"
        "Navigate to specific tissue regions vertically across the slide. "
        "Combine with X position to precisely target regions for quality control or analysis.\n\n"
        "*Technical: Y-coordinate (pixels) for the top-left corner of the extracted tile.*"
    ),

    # ============================================
    # STAIN SEPARATION
    # ============================================
    'stain_option': (
        "**Stain Channel Isolation**\n\n"
        "Separate individual stain contributions for quantitative analysis:\n\n"
        "• **Hematoxylin** - Isolates nuclear staining (blue/purple) for cell counting, "
        "mitotic index calculation, and nuclear morphometry in tumor grading.\n\n"
        "• **Eosin** - Isolates cytoplasmic/stromal staining (pink) for tissue architecture "
        "analysis and fibrosis quantification.\n\n"
        "• **DAB** - Isolates immunohistochemistry chromogen (brown) for biomarker "
        "quantification such as Ki-67, HER2, or PD-L1 scoring.\n\n"
        "*Technical: Uses color deconvolution in HED color space to unmix overlapping stain signals.*"
    ),

    # ============================================
    # FEATURE EXTRACTION
    # ============================================
    'gabor': (
        "**Texture Pattern Analysis**\n\n"
        "Quantifies tissue texture patterns for automated classification. Applications include:\n"
        "• Fibrosis staging in liver biopsies\n"
        "• Collagen fiber orientation analysis\n"
        "• Tumor stroma characterization\n\n"
        "*Technical: Applies Gabor wavelets - orientation-sensitive bandpass filters that "
        "respond to specific spatial frequencies and directions in the image.*"
    ),

    'sobel': (
        "**Boundary Detection**\n\n"
        "Highlights tissue boundaries and structural edges for segmentation workflows:\n"
        "• Cell membrane delineation\n"
        "• Gland lumen detection\n"
        "• Tumor margin identification\n\n"
        "*Technical: Computes first-order intensity gradients using Sobel operators, "
        "producing an edge magnitude map highlighting rapid intensity transitions.*"
    ),

    # ============================================
    # MORPHOLOGICAL OPERATIONS
    # ============================================
    'dilation': (
        "**Region Expansion**\n\n"
        "Expands detected regions to connect fragmented structures:\n"
        "• Bridge gaps between disconnected cell clusters\n"
        "• Create safety margins around detected lesions\n"
        "• Fill small holes in binary masks\n\n"
        "*Technical: Applies morphological dilation with a disk structuring element "
        "after Otsu thresholding to expand foreground regions.*"
    ),

    'smallobjs': (
        "**Artifact Removal**\n\n"
        "Eliminates small spurious detections to reduce false positives:\n"
        "• Remove dust particles and debris\n"
        "• Filter out staining artifacts\n"
        "• Clean up noisy segmentation results\n\n"
        "*Technical: Labels connected components and removes objects below a specified "
        "pixel area threshold, preserving only significant structures.*"
    ),

    # ============================================
    # GEOMETRIC TRANSFORMATIONS
    # ============================================
    'rotate': (
        "**Orientation Correction**\n\n"
        "Rotate tissue images for standardization or augmentation:\n"
        "• Correct specimen orientation for consistent analysis\n"
        "• Generate rotated variants for AI model training\n"
        "• Align tissue sections for registration\n\n"
        "*Technical: Performs rotation around the image center using bilinear "
        "interpolation with zero-padding at boundaries.*"
    ),

    'warp': (
        "**Geometric Transformation**\n\n"
        "Apply complex spatial transformations for advanced processing:\n"
        "• Correct lens distortion and scanning artifacts\n"
        "• Register serial tissue sections\n"
        "• Generate augmented training data for deep learning\n\n"
        "*Technical: Applies affine transformation matrix combining scale, rotation, "
        "shear, and translation operations in a single pass.*"
    ),
}
