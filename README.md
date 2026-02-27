# DCC - DaCapo CellMap Config

Export PyTorch models to multiple portable formats for inference and finetuning.

## Installation

```bash
pip install -e .
```

## Export Formats

Each exported model directory contains:

| File | Format | Description |
|---|---|---|
| `model.pt` | PyTorch pickle | Full model object (`torch.save`) |
| `model.pt2` | torch.export | `ExportedProgram` — supports `unflatten` for finetuning |
| `model.ts` | TorchScript | Traced model (`torch.jit.trace`) |
| `model.onnx` | ONNX | For cross-framework inference |
| `metadata.json` | JSON | Model metadata (shapes, voxel sizes, channels, etc.) |
| `README.md` | Markdown | Auto-generated model card |

## Usage

### 1. Export a DaCapo model

```python
import dcc.model_export.config as c
c.DCC_EXPORT_FOLDER = "/path/to/export/folder"

from dcc.model_export.dacapo_model import export_dacapo_model

export_dacapo_model("my_run_name", iteration=100000)
```

### 2. Export any PyTorch model

```python
import torch
import dcc.model_export.config as c
from dcc.model_export.generate_metadata import ModelMetadata, get_export_folder
from dcc.model_export.export_model import export_torch_model
import os

c.DCC_EXPORT_FOLDER = "/path/to/export/folder"

# Load your model
model = ...  # any torch.nn.Module
model.eval()

# Define metadata
metadata = ModelMetadata(
    model_name="my_model",
    model_type="UNet",
    framework="torch",
    spatial_dims=3,
    in_channels=1,
    out_channels=2,
    channels_names=["mito", "er"],
    input_shape=[1, 1, 96, 96, 96],
    output_shape=[1, 2, 96, 96, 96],
    inference_input_shape=[96, 96, 96],
    inference_output_shape=[96, 96, 96],
    input_voxel_size=[8, 8, 8],
    output_voxel_size=[8, 8, 8],
    author="Your Name",
    description="My segmentation model",
)

# Export model to all formats (pt, pt2, ts, onnx) + metadata and README
input_shape = (1, 1, 96, 96, 96)
export_torch_model(model, input_shape, os.path.join(get_export_folder(), "my_model"), metadata=metadata)
```

### 3. Load an exported model for inference

```python
from dcc.model_export.cellmap_model import CellmapModel

model = CellmapModel("/path/to/export/folder/my_model")

# Access metadata
print(model.metadata.channels_names)

# Use any format for inference
onnx_session = model.onnx_model        # ONNX Runtime session
ts_model = model.ts_model              # TorchScript model
pt_model = model.pytorch_model         # PyTorch pickle model
exported = model.exported_model         # torch.export ExportedProgram
```

### 4. Load an exported model for finetuning

```python
from dcc.model_export.cellmap_model import CellmapModel

cellmap_model = CellmapModel("/path/to/export/folder/my_model")
model = cellmap_model.train()
# Returns an nn.Module in train mode
# Tries torch.export (model.pt2) + unflatten first, falls back to TorchScript
```
