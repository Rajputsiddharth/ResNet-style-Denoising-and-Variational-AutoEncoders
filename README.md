# Altered MNIST Image Reconstruction using Autoencoders

This script implements different variants of autoencoders (AE) to reconstruct altered MNIST images.
The altered MNIST dataset consists of images where each clean image has been augmented to create several variations.
The autoencoders are trained to reconstruct the clean version of the augmented images.
Two variants of autoencoders, namely Vanilla Autoencoder (VAE) and Conditional Variational Autoencoder (CVAE), are implemented and compared in terms of reconstruction quality.

## Dataset
The dataset used in this project consists of two folders:
- **clean**: Contains the clean MNIST images.
- **aug**: Contains the augmented versions of the clean MNIST images.

## Requirements
- Python 3.x
- PyTorch
- NumPy
- Scikit-learn
- Matplotlib
- Scikit-image

## Usage
1. Run this script.
2. Modify the parameters such as training epochs, batch size, learning rate, etc., as per your requirement.
3. The script will train both Vanilla Autoencoder (AE) and Variational Autoencoder (VAE).
4. Once trained, the script will evaluate the models based on specified evaluation metrics.
5. You can adjust the evaluation parameters and choose the type of evaluation metric (`SSIM` or `PSNR`).

## Conditional Variational Autoencoder (CVAE)
The implementation of Conditional Variational Autoencoder (CVAE) is not available in this script.
However, it can be added by extending the VAE architecture to include label conditioning in both the encoder and decoder.

## Results
- The trained models achieve high reconstruction quality, as measured by Structural Similarity Index (SSIM) and Peak Signal-to-Noise Ratio (PSNR).

## Future Work
- Implement the Conditional Variational Autoencoder (CVAE) for conditional image generation.
- Explore different augmentation techniques for improved model performance.

## Acknowledgments
This project is inspired by the Altered MNIST dataset and builds upon existing autoencoder architectures.
