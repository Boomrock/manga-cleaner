# Manga Text Cleaner

This project provides a tool to automatically remove text (e.g., speech bubbles, subtitles) from manga images using deep learning techniques.

## Features

- Uses **U-Net** or **U-Net++** architectures for text detection.
- Employs **LaMa inpainting model** to clean detected text regions.
- Supports both CPU and GPU inference.

## Requirements

- Python 3.12
- PyTorch
- PIL / Pillow
- segmentation_models_pytorch
- simple_lama_inpainting

Install dependencies via pip:

```bash
pip install -r requirements.txt
```

## Usage

### Example usage:

```python
from Cleaner.Cleaner import Cleaner
from PIL import Image

cleaner = Cleaner()  # You can specify detector type: DetectorType.UNetPlusPlus
image = Image.open("path_to_manga_image.jpg").convert("RGB")
cleaned_image = cleaner.clean(image)
cleaned_image.save("cleaned_image.jpg")
```

## Modules

- `Cleaner`: Main class for cleaning text from manga images.
- `Preprocessor / Postprocessor`: Handle mask preprocessing and postprocessing.
- `ModelLoader`: Loads trained weights for the detector model.
- `DetectorType`: Enum to choose between U-Net and U-Net++.

## Supported Detectors

- `UNet`
- `UNetPlusPlus`

