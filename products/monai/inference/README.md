# MONAI Inference UI

This folder contains the MONAI inference interface for medical image analysis, with specialized support for pathology tumor detection and other medical imaging tasks.

## Overview

The inference UI provides a Streamlit-based interface for running inference on trained MONAI models, with particular focus on:
- **Pathology tumor detection** on Whole Slide Images (WSI)
- **CT segmentation** tasks
- **Custom trained models** from the training pipeline

## File Structure

```
products/monai/inference/
├── README.md              # This documentation
├── driver.py              # Main inference UI driver
├── sidebar.py             # Sidebar controls and model selection
├── pathology_inference.py # Specialized pathology WSI inference
└── data_manager.py        # Data download and management
```

## Required Folder Structure

For the inference system to work correctly, ensure the following folder structure exists in your project root:

### Data Directory
```
data/
├── pathology_tumor_detection/     # Pathology image data
│   ├── tumor_001.tif             # WSI files (.tif, .png, .jpg)
│   ├── tumor_002.tif
│   └── ...
└── spleen_ct_segmentation/        # CT scan data
    ├── spleen_11.nii.gz          # NIfTI files (.nii, .nii.gz)
    ├── spleen_12.nii.gz
    └── ...
```

### Trained Models Directory
```
trained_models/
├── pathology_tumor_detection/     # Pathology models
│   ├── models/                   # Model files directory
│   │   ├── model.pth            # PyTorch model files
│   │   ├── best_model.pth
│   │   └── ...
│   └── validation.csv            # Validation results (optional)
└── spleen_ct_segmentation/        # CT segmentation models
    ├── models/
    │   ├── model.pth
    │   └── ...
    └── validation.csv
```

### Model Zoo Directory (Optional)
```
model-zoo-models/
├── pathology_tumor_detection/     # Downloaded zoo models
│   └── model_*.pth
└── spleen_ct_seg/
    └── model_*.pth
```

## How to Use

### 1. Access the Interface

Run the main demo application:
```bash
cd /path/to/rocm-ls-examples
python rocm-ls_demo.py
```

Navigate to the **MONAI → Inference** tab.

### 2. Model Selection

**Option A: Model Zoo**
- Select "Model Zoo" as model source
- Choose from pre-configured models
- Download data and models using the provided buttons

**Option B: Custom Trained Models**
- Select "Custom Trained" as model source
- Choose from models in your `trained_models/` directory
- Models are automatically detected from the folder structure

### 3. Data Input

**Option A: Local Data**
- Uses data from your `data/` directory
- Automatically matches data to selected model type

**Option B: Upload File**
- Upload medical images directly (.nii, .nii.gz, .tif, .png, .jpg)

**Option C: Paste Path**
- Enter full path to your image file
- Supports all medical image formats

### 4. Pathology Inference Options

For pathology tumor detection models, you get specialized options:

**Quick Analysis (4 patches)**
- Fast analysis using 4 representative patches
- Good for initial screening

**Custom Patch Count (1-20 patches)**
- Adjustable number of patches
- Balance between speed and coverage

**Full WSI Sliding Window**
- Comprehensive whole slide analysis
- Options: 50, 100, 200, 500, 1000, or All possible patches
- Uses 224×224 pixel patches with no overlap
- Generates CSV results similar to validation.csv

### 5. Device Selection

- **Auto**: Automatically selects GPU if available, otherwise CPU
- **GPU**: Force GPU usage (requires CUDA)
- **CPU**: Force CPU usage

## Output

The inference system provides:

### Visualization Tabs
1. **Overview**: Summary metrics and original image
2. **Patch Analysis**: Visual analysis of 224×224 patches
3. **CSV Results**: Downloadable results in CSV format
4. **Technical**: Model and system information

### CSV Output Format
Results are saved to `inference_outputs/` with validation.csv format:
```csv
patch_x,patch_y,patch_w,patch_h,prediction,probability,tumor_detected
0,0,224,224,0,0.234,False
224,0,224,224,1,0.891,True
...
```

## System Requirements

### For Small Images (< 1000×1000 pixels)
- 4GB RAM minimum
- Any modern CPU or GPU

### For Large WSI Files (> 50,000×50,000 pixels)
- 16GB+ RAM recommended
- GPU highly recommended for faster processing
- Processing time: 1-10 hours depending on image size and patch count

## Supported File Formats

**Medical Images:**
- NIfTI: `.nii`, `.nii.gz` (CT scans)
- TIFF: `.tif`, `.tiff` (WSI pathology)
- Standard: `.png`, `.jpg`, `.jpeg`
- DICOM: `.dcm`

**Models:**
- PyTorch: `.pth`, `.pt`
- Pickle: `.pkl`

## Troubleshooting

### "No trained models found"
- Ensure `trained_models/` directory exists with proper structure
- Check that model files are in the correct subdirectories
- Verify file extensions are `.pth`, `.pt`, or `.pkl`

### "Image size exceeds limit"
- Large WSI files are automatically handled
- If issues persist, use "Paste Path" option instead of upload

### "Process getting killed"
- Reduce patch count for very large images
- Use GPU if available
- Increase system memory if possible

### "Only 1 patch generated"
- Check image size - very small images may only fit one patch
- Ensure image is larger than 224×224 pixels
- Try "Custom Patch Count" mode instead

## Advanced Configuration

### Memory Management
The system automatically:
- Removes PIL image size limits for WSI
- Limits visualization patches to prevent memory issues
- Caps "All possible patches" at 50,000 for stability

### Performance Tips
- Use GPU when available
- Start with smaller patch counts for testing
- Use "Local Data" option for faster file access
- Close other applications when processing large WSI files

## Integration with Training

Models trained using the MONAI training pipeline are automatically compatible:
1. Train model using the Training tab
2. Models are saved to `trained_models/[model_type]/models/`
3. Switch to Inference tab and select "Custom Trained"
4. Your trained model appears in the dropdown

## File Paths

All paths in the system use relative references from the project root:
- Data: `./data/[model_type]/`
- Models: `./trained_models/[model_type]/models/`
- Output: `./inference_outputs/`

This ensures portability across different systems and users.