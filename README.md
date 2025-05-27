# Practical Course: Image Classification with Neural Networks (CIFAR-10)

This repository contains a Jupyter notebook for a practical course on image classification using neural networks, specifically focusing on the CIFAR-10 dataset. The course covers various neural network architectures including MLPs and CNNs.

## Project Structure

```
.
├── Practical_Course_Cifar_10.ipynb  # Main notebook file
├── ActMax/                          # Submodule for activation maximization
├── data/                           # Directory for CIFAR-10 dataset
├── myLogs/                         # Directory for your training logs
├── myModels/                       # Directory for your trained models
├── lecture_logs/                   # Pre-trained model logs
└── lecture_models/                 # Pre-trained models
```

## Prerequisites

- Python 3.8 or higher
- CUDA-capable GPU (optional but recommended)
- Jupyter Notebook/Lab environment

## Installation

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd <repository-name>
   ```

2. Initialize and update the submodule:
   ```bash
   git submodule init
   git submodule update
   ```

3. Create and activate a virtual environment (recommended):
   ```bash
   conda create -n torch python=3.8
   conda activate torch
   ```

4. Install the required packages:
   ```bash
   # Install PyTorch and torchvision
   # Visit https://pytorch.org for the appropriate command for your system
   # Example for CPU-only installation:
   pip install torch torchvision

   # Install other dependencies
   pip install numpy matplotlib seaborn jupyter tensorboard
   ```

## Usage

1. Start Jupyter Notebook:
   ```bash
   jupyter notebook
   ```

2. Open `Practical_Course_Cifar_10.ipynb` in your browser.

3. To view training logs using TensorBoard:
   ```bash
   # For viewing lecture logs
   tensorboard --logdir lecture_logs

   # For viewing your own experiment logs
   tensorboard --logdir myLogs
   ```
   If you encounter issues, try:
   ```bash
   tensorboard --load_fast=false --logdir lecture_logs
   ```

## Dataset

The notebook uses the CIFAR-10 dataset, which contains:
- 60,000 32x32 color images
- 10 different classes
- 50,000 training images and 10,000 test images
- 6,000 images per class

The dataset will be automatically downloaded when running the notebook.

## Training Options

The notebook provides two ways to work with the models:

1. **Using Pre-trained Models**: Set `TRAIN_IN_NOTEBOOK = False` to load and experiment with pre-trained models.
2. **Training from Scratch**: Set `TRAIN_IN_NOTEBOOK = True` to train the models yourself (GPU recommended).

## Google Colab Support

If you don't have access to a GPU, you can run this notebook on Google Colab:
1. Upload the notebook to Google Colab
2. Select Runtime > Change runtime type > GPU
3. Follow the notebook instructions

Note: Some adjustments might be needed for the Colab environment.

## Notes

- Training on CPU is possible but will be significantly slower
- The notebook includes experiments with both MLP and CNN architectures
- Custom logging directories are created automatically when needed
- The implementation includes visualization tools for understanding model behavior

## Disclaimer

The notebook version may have slight differences from the lecture version, particularly in the MLP model architecture. These changes generally result in better performance while maintaining the same educational objectives. 