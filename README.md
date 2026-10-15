# CropCare 🌿

## Project Overview

**CropCare** is an AI-based plant disease detection system designed to identify diseases in papaya leaves from images. The user can capture or upload a leaf image, and the trained deep learning model analyzes the image and predicts the disease.

The system supports five classes: **Anthracnose, Bacterial Spot, Curl, Healthy, and Ring Spot**. The project uses **PyTorch** and pretrained deep learning models such as **EfficientNet** and **ConvNeXt** for image classification.

The main goal of CropCare is to provide a simple and accessible way to detect papaya leaf diseases at an early stage and provide useful information to the user.

### System Flow

**Leaf Image → Preprocessing → Deep Learning Model → Disease Prediction → Result**

### Key Features

- 🌿 Papaya leaf disease detection
- 📷 Image-based prediction
- 🤖 Deep learning-based classification
- 🧠 EfficientNet and ConvNeXt models
- 📊 Training and validation using real-world papaya leaf images
- 💡 Disease prediction with confidence score

---

## Dataset
Dataset : https://data.mendeley.com/datasets/p997fvf526/2

This project uses the **Mendeley BDPapayaLeaf** dataset.

* **Classes:** Anthracnose, BacterialSpot, Curl, Healthy, and RingSpot.
* **Setup:** Only the `Original Images` folder is required. The data must be split into an 80/20 training and validation distribution prior to execution.

# Software Requirements
* **Python:** 3.10 or higher
* **Core Frameworks:** PyTorch, Torchvision (CUDA-enabled build recommended)
* **Libraries:**
  * `timm` (PyTorch Image Models for EfficientNet/ConvNeXt)
  * `albumentations` (Heavy field-condition data augmentation)
  * `opencv-python-headless` (Image processing)
  * `split-folders` (Dataset formatting)
  * `Pillow` (Image loading)
