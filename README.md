# Altered MNIST Image Reconstruction using Autoencoders
This Python script implements and trains three types of autoencoders: regular autoencoders, variational autoencoders (VAEs), and conditional variational autoencoders 
The altered MNIST dataset consists of images where each clean image has been augmented to create several variations.
The autoencoders are trained to reconstruct the clean version of the augmented images.
Both the encoder and decoder architectures follow the ResNet style, with residual connections after 2 convolution / 2 convolution-batchnorm layers.

## Dataset
The dataset used in this project consists of two folders:
- **clean**: Contains the clean MNIST images.
- **aug**: Contains the augmented versions of the clean MNIST images.
Download the dataset from the following link:
[Download Dataset](https://drive.google.com/file/d/1cCPO0A4YB0oGcuzmw3S7OTkpHKfUFkzR/view?usp=sharing)

## Requirements
- Python 3.x
- PyTorch
- NumPy
- Scikit-learn
- Matplotlib
- Scikit-image

## Usage
1. Run this script for the file 2021102A3.
2. No need for user input variables; everything is handled internally.
3. The script will train denoising autoencoders, VAEs, and CVAEs on the provided dataset.
4. After training, the script will generate 3D TSNE embedding plots for logits/embeddings of the whole dataset after every 10 epochs.
5. Checkpoints are saved as required.

## Implementation
1. **Denoising Autoencoder (AE)**:
   - Encoder and decoder follow ResNet style with residual connections.
   - Design choices are flexible except for the specified residual connections.

2. **Denoising Variational Autoencoder (VAE)**:
   - Similar architecture to the AE but with VAE-specific modifications.
   - Encoder outputs logits/embeddings, which are then sampled for the VAE loss calculation.
   - Additional TSNE plots are generated for sampled logits/embeddings.

3. **Conditional Variational Autoencoder (CVAE)** (Bonus Question):
   - Implements a CVAE to generate one of the classes of the MNIST dataset at inference time, given the class label.
   - Architecture includes label conditioning in both the encoder and decoder.
     
## Future Work
- Explore different augmentation techniques for improved model performance.
