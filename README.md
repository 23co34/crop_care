# crop_care


Dataset : https://data.mendeley.com/datasets/p997fvf526/2

## Dataset
This project uses the **Mendeley BDPapayaLeaf** dataset. 
*   **Classes:** Anthracnose, BacterialSpot, Curl, Healthy, and RingSpot.
*   **Setup:** Only the `Original Images` folder is required. The data must be split into an 80/20 training and validation distribution prior to execution.

# Software Requirements
* **Python:** 3.10 or higher
* **Core Frameworks:** PyTorch, Torchvision (CUDA-enabled build recommended)
* **Libraries:**
  * `timm` (PyTorch Image Models for EfficientNet/ConvNeXt)
  * `albumentations` (Heavy field-condition data augmentation)
  * `opencv-python-headless` (Image processing)
  * `split-folders` (Dataset formatting)
  * `Pillow` (Image loading)
