

















# PyTorch VGG19 Feature Extraction and Loss Modules

This project contains implementations of a VGG19 feature extractor and custom loss modules for content and style loss using PyTorch.

## Files

- `import_torch.py`: Contains the implementation of the VGG19 feature extractor, `ContentLoss`, and `StyleLoss` modules.

## Classes

### VGG19

A class that extracts features from the VGG19 network.

#### Methods

- `__init__()`: Initializes the VGG19 model and extracts the first 21 layers.
- `forward(x)`: Passes the input `x` through the VGG19 layers and returns the features.

### ContentLoss

A class that computes the content loss between the target and the input.

#### Methods

- `__init__(target, idx)`: Initializes the content loss with the target feature map and index.
- `forward(x)`: Computes the mean squared error loss between the input `x` and the target.

### StyleLoss

A class that computes the style loss between the target and the input using the Gram matrix.

#### Methods

- `__init__(target, idx)`: Initializes the style loss with the target feature map and index.
- `forward(x)`: Computes the mean squared error loss between the Gram matrix of the input `x` and the target.
- `gram_matrix(x)`: Computes the Gram matrix of the input `x`.

## Usage

To use the classes, import them from the `import_torch.py` file:

```python
from import_torch import VGG19, ContentLoss, StyleLoss
