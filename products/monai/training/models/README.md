# AMD ROCm-LS Modular Model Integration Template

## Overview
This project enables a plug-and-play, dynamic demo and training experience for MONAI Model Zoo and custom PyTorch models. Each model is implemented as its own Python module, using a standardized set of APIs, so that the Streamlit UI frontend (driver) can orchestrate data loading, training, visualization, and evaluation without knowing any model-specific details.

Any new model you addsegmentation, classification, detection, etc.should follow the interface and conventions detailed below to ensure seamless UI integration and metric/visualization support.

# Directory Structure Example
```
products/
  monai/
    training/
      models/
        spleen_ct_seg.py
        my_custom_model.py       # <--- your new model goes here
      driver.py
    layout.py
```

## Required Model APIs (Functions)

Each model implementation must export the following functions (signatures and docstrings should match):

### Data Preparation
```
def prepare_training_data(params: dict, val_frac: float = 0.2, random_seed: int = 42) -> dict:
    """
    Download/process and split data. Return DataLoaders for train and val sets.

    Returns:
        loaders: dict with keys 'train' and 'val', each a DataLoader.
    """
```
- Should perform all necessary preprocessing, optional splits, and augmentations.
- Return a dictionary: {'train': train_loader, 'val': val_loader}

### Visualization Sample Extraction

```
def extract_visualization_samples_from_loader(
    loader, n: int = 10, with_label: bool = True
) -> list:
    """
    Extract and return n random visualization samples for thumbnail display.

    Returns:
        List of dicts, each with at least 'input', and optionally 'label'.
    """
```

- Must return random different samples each call (not fixed slice).
- Each dict typically has: `{'input': <2D numpy>, 'label': <2D numpy>}`.

### Device Selection

```
def identify_training_device(params: dict):
    """
    Return the torch.device for current training (e.g., ROCm GPU or CPU).
    """
```

E.g., `return torch.device("cuda" if torch.cuda.is_available() else "cpu")`

### Network Constructor

```
def define_training_network(params: dict, device) -> torch.nn.Module:
    """
    Build and return model (e.g., UNet) on correct device.
    """
```

- All arguments (input/output channels, etc.) can be read from params as needed.

### Loss, Optimizer, Metric Setup

```
def define_loss_function(params: dict):
    """Return the loss criterion, e.g., DiceLoss."""
def define_training_optimizer(params: dict, model_parameters):
    """Return optimizer, e.g., Adam or SGD."""
def define_training_metric(params: dict):
    """Return metric function, e.g., DiceMetric."""
```

### Batch Training Step

```
def train_batch(
    network, batch_data, loss_fn, optimizer, metric_fn, device, params, epoch
) -> tuple:
    """
    Single training step. Must return:
      - loss (float)
      - mean Dice (float), (or None)
      - per-class metrics dict (optional)
      - accuracy (optional)
      - precision (optional)
      - recall (optional)
    """
```

- Must always return a tuple of 6 values
- In order to maintain compatibility with the driver, return `None` for any unimplemented metric

### Batch Evaluation Step

```
def evaluate_batch(
    network, batch_data, loss_fn, metric_fn, device, params, epoch
) -> tuple:
    """
    Validation step. Same convention for return as train_batch, but can include
    visualizations: e.g. (dice, val_loss, val_dice, per_class_metrics, acc, prec, recall, visualization[dict/list])
    """
```

## API Flexibility & Best Practices

- If your model does not support certain metrics (e.g., multiclass), return None for those slots.
- Per-class metrics for multi-class problems (e.g., each organ) should be returned as dicts.
- Random sample extraction should use dataset/batch indices shuffled each call.
- All data returned should be compatible with numpy and st.image/st.line_chart or pandas DataFrames.

## How the Driver Leverages This API

- The Streamlit driver (see `driver.py`) will call these functions, passing current user-selected parameters.
- All UI rendering, progress, and analytics are orchestrated through a callback pattern: at each major phase (or batch/epoch), your training logic calls the driver's callback, which updates the relevant UI panels (loss curves, Dice, sample images, status bars, device info, etc.).
- The driver ensures that training and visualization are decoupled, so your model implementation never needs to know about Streamlit or UI logic.

## Adding a New Model Step by Step

1. *Copy and rename* `spleen_ct_seg.py` as a new file in `models/`, e.g., `brain_tumor_seg.py`.
2. *Update docstrings* cross-function to reflect your model's purpose.
3. *Implement functions above*, customizing transforms, splits, model, and metrics as needed.
4. *Test by adding your model to the Model Zoo menu/registry* in the driver and running the Streamlit app.
5. *Iterate*: As new metrics or visualizations are added, adapt your batch/eval step return signatures.

## Example Snippet for a Minimal Model

```
def prepare_training_data(params, val_frac=0.2, random_seed=42): ...
def extract_visualization_samples_from_loader(loader, n=10, with_label=True): ...
def identify_training_device(params): ...
def define_training_network(params, device): ...
def define_loss_function(params): ...
def define_training_optimizer(params, model_parameters): ...
def define_training_metric(params): ...
def train_batch(...): return loss, dice, None, None, None, None
def evaluate_batch(...): return dice, val_loss, dice, None, None, None, []
```

## Troubleshooting

Ensure all APIs always return the expected tuple, filling with None as needed.
For new metrics, document what is returned at each step to keep driver-side DataFrame handling robust.
Use default/baseline values (like `np.nan` for missing floats) to keep curves working with incomplete data.

