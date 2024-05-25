# ResNet-style-Denoising-and-Variational-AutoEncoders
This project implements different variants of autoencoders (AE) to reconstruct altered MNIST images. The altered MNIST dataset consists of images where each clean image has been augmented to create several variations. The autoencoders are trained to reconstruct the clean version of the augmented images. Two variants of autoencoders, namely Vanilla Autoencoder (VAE) and Conditional Variational Autoencoder (CVAE), are implemented and compared in terms of reconstruction quality.

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

### Training
1. Run `train_ae.py` to train a Vanilla Autoencoder (AE) or `train_vae.py` to train a Variational Autoencoder (VAE).
2. The trained models will be saved in the current directory.

### Evaluation
1. To evaluate the trained AE/VAE models, run `evaluate_ae.py` or `evaluate_vae.py`, respectively.
2. Provide the paths to the sample and original images for evaluation.
3. Choose the type of evaluation metric (`SSIM` or `PSNR`).

### Conditional Variational Autoencoder (CVAE)
The Conditional Variational Autoencoder (CVAE) implementation is not available in the current README, but it can be added by extending the VAE architecture to include label conditioning in both the encoder and decoder. The training and evaluation process would be similar to the VAE.

## Files
- `train_ae.py`: Script for training the Vanilla Autoencoder.
- `train_vae.py`: Script for training the Variational Autoencoder.
- `evaluate_ae.py`: Script for evaluating the Vanilla Autoencoder.
- `evaluate_vae.py`: Script for evaluating the Variational Autoencoder.
- `best_modelAE_ssim.pth`: Saved model weights for the best Vanilla Autoencoder based on SSIM.
- `best_modelVAE_ssim.pth`: Saved model weights for the best Variational Autoencoder based on SSIM.
- `README.md`: Project README markdown file.

## Results
- The trained models achieve high reconstruction quality, as measured by Structural Similarity Index (SSIM) and Peak Signal-to-Noise Ratio (PSNR).

## Future Work
- Implement the Conditional Variational Autoencoder (CVAE) for conditional image generation.
- Explore different augmentation techniques for improved model performance.

## Acknowledgments
This project is inspired by the Altered MNIST dataset and builds upon existing autoencoder architectures.


