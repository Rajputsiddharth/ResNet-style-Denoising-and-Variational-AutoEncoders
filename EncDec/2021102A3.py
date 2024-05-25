import os
import torch
import random
import argparse
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, Dataset
import torchvision.io as io
import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
from sklearn.metrics import pairwise_distances_argmin_min
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import cosine_similarity
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
import torchvision.utils as vutils

torch.manual_seed(19)

if torch.cuda.is_available():
    # Set random seed for CUDA
    torch.cuda.manual_seed(19)
    torch.cuda.manual_seed_all(19)

class AlteredMNIST(Dataset):
    def __init__(self):
        # Define paths to clean and aug folders
        self.clean_folder = "./Data/clean"
        self.aug_folder = "./Data/aug"
        self.mapping, self.num_clusters = self.create_mapping()

    def __len__(self):
        return len(self.mapping)

    def __getitem__(self, idx):
        aug_image_path, clean_image_path = self.mapping[idx]
        aug_image = io.read_image(aug_image_path)
        clean_image = io.read_image(clean_image_path)

        # Convert to greyscale
        aug_image = transforms.functional.rgb_to_grayscale(aug_image)
        clean_image = transforms.functional.rgb_to_grayscale(clean_image)

        # Convert to NumPy arrays
        aug_image = transforms.functional.to_pil_image(aug_image)
        clean_image = transforms.functional.to_pil_image(clean_image)

        transform = transforms.Compose([
            transforms.ToTensor(),
        ])
        aug_image = transform(aug_image)
        clean_image = transform(clean_image)

        # Extract label from the filename
        label = self.extract_label(os.path.basename(clean_image_path))

        return aug_image, clean_image, label

    def extract_label(self, filename):
        return int(filename.split("_")[-1].split(".")[0])  # Convert label to integer

    def create_mapping(self):
        label_dict_clean = {}
        label_dict_aug = {}
        num_clusters = {}

        for clean_image in os.listdir(self.clean_folder):
            label = self.extract_label(clean_image)
            if label not in label_dict_clean:
                label_dict_clean[label] = []
            label_dict_clean[label].append(clean_image)

        for aug_image in os.listdir(self.aug_folder):
            label = self.extract_label(aug_image)
            if label not in label_dict_aug:
                label_dict_aug[label] = []
            label_dict_aug[label].append(aug_image)

        mapping = []

        for label in label_dict_clean.keys():  # Dynamically iterate over labels
            clean_images = label_dict_clean[label]
            aug_images = label_dict_aug[label]

            # Apply PCA with 95% variance retention to clean images
            pca = PCA(0.95)
            clean_data = []
            for clean_image in clean_images:
                clean_img = io.read_image(os.path.join(self.clean_folder, clean_image))
                clean_img = clean_img.numpy()
                clean_img = transforms.functional.rgb_to_grayscale(torch.from_numpy(clean_img))  # Convert to grayscale
                clean_data.append(clean_img.flatten())
            clean_data = np.array(clean_data)
            pca.fit(clean_data)

            # Fit GMM to PCA-transformed clean images
            gmm = GaussianMixture(n_components=45)
            gmm.fit(pca.transform(clean_data))

            unique_clusters = set()
            num_unique_mapped_clean= set()

            # Precompute PCA-transformed representations and cluster predictions for clean images
            clean_data_pca = pca.transform(clean_data)
            clean_cluster_predictions = gmm.predict(clean_data_pca)

            for aug_image in aug_images:
                aug_img = io.read_image(os.path.join(self.aug_folder, aug_image))
                aug_img = aug_img.numpy()
                aug_img = transforms.functional.rgb_to_grayscale(torch.from_numpy(aug_img))  # Convert to grayscale
                aug_img = aug_img.flatten()
                aug_img_pca = pca.transform(aug_img.reshape(1, -1))
                label_pred = gmm.predict(aug_img_pca)

                # Filter clean images belonging to the same cluster as label_pred
                clean_images_cluster_indices = np.where(clean_cluster_predictions == label_pred)[0]
                clean_images_cluster = [clean_images[i] for i in clean_images_cluster_indices]

                # Calculate distances to find closest clean image
                distances = [np.linalg.norm(aug_img_pca - clean_data_pca[i]) for i in clean_images_cluster_indices]
                closest_clean_image_index = clean_images_cluster_indices[np.argmin(distances)]
                closest_clean_image = clean_images[closest_clean_image_index]

                mapping.append((os.path.join(self.aug_folder, aug_image), os.path.join(self.clean_folder, closest_clean_image)))
            #     unique_clusters.add(label_pred[0])
            #     num_unique_mapped_clean.add(closest_clean_image_index)

            # num_clusters[label] = len(unique_clusters)
            # print(f"Label {label}: {len(unique_clusters)} unique clusters, {len(num_unique_mapped_clean)} unique clean images mapped")


        random.shuffle(mapping)
        return mapping, num_clusters

# Define ResNet Block with residual connection
class ResNetBlockEncoder(nn.Module):
    def __init__(self, in_channels, out_channels, stride=2, downsample=None):
        super(ResNetBlockEncoder, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.in_channels = 1
        self.layer1 = self.make_layer(ResNetBlockEncoder, 32, 1, stride=1)
        self.layer2 = self.make_layer(ResNetBlockEncoder, 64, 1, stride=2)
        self.layer3 = self.make_layer(ResNetBlockEncoder, 128, 1, stride=2)
        self.layer4 = self.make_layer(ResNetBlockEncoder, 256, 1, stride=2)
        self.fcvae_mu = nn.Linear(4096, 64)
        self.fcvae_logvar = nn.Linear(4096, 64)
        self.fc= nn.Linear(64, 4096)
        self.num_classes = 10
        self.fccvae = nn.Linear(4096, 64)
        self.fccvae_logvar = nn.Linear(4096, 64)
        
        # Add a linear layer for the class label
        self.label_projector = nn.Sequential(
            nn.Linear(self.num_classes, 64),
            nn.ReLU(),
        )

    def condition_on_label(self, z, y):
        projected_label = self.label_projector(y.float())
        return z + projected_label


    def make_layer(self, block, out_channels, blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        layers = []
        layers.append(block(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels
        for _ in range(1, blocks):
            layers.append(block(self.in_channels, out_channels))
        return nn.Sequential(*layers)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def forward(self, x, flag=0, labels=None):
        if flag==0:
            mu = None
            logvar = None
            #print(x.shape)
            x = self.layer1(x)
            #print(x.shape)
            x = self.layer2(x)
            #print(x.shape)
            x = self.layer3(x)
            #print(x.shape)
            x = self.layer4(x)
            #print(x.shape)

        if flag==1:
            x = self.layer1(x)
            #print(x.shape)
            x = self.layer2(x)
            #print(x.shape)
            x = self.layer3(x)
            #print(x.shape)
            x = self.layer4(x)
            #print(x.shape)
            x = torch.flatten(x, start_dim=1)
            mu = self.fcvae_mu(x)
            logvar = self.fcvae_logvar(x)
            z = self.reparameterize(mu, logvar)
            z = self.fc(z)
            x = z.view(-1, 256, 4, 4)

        if flag==2:
            #print(x.shape)
            x = self.layer1(x)
            #print(x.shape)
            x = self.layer2(x)
            #print(x.shape)
            x = self.layer3(x)
            #print(x.shape)
            x = self.layer4(x)
            #print(x.shape)
            x = torch.flatten(x, start_dim=1)
            mu = self.fccvae(x)
            logvar = self.fccvae_logvar(x)
            z = self.reparameterize(mu, logvar)
            #One hot encoding of labels
            y = F.one_hot(labels, num_classes=self.num_classes)
            z = self.condition_on_label(z, y)
            #print(z.shape)
            z = self.fc(z)
            x = z.view(-1, 256, 4, 4)

        return x, mu, logvar

class ResNetBlockDecoder(nn.Module):
    def __init__(self, in_channels, out_channels, stride=2, flag=0, upsample=None):
        super(ResNetBlockDecoder, self).__init__()
        self.conv2 = nn.ConvTranspose2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1,bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        if flag==0:
          self.conv1 = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        else:
          self.conv1 = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1,output_padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.upsample = upsample

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.upsample is not None:
            identity = self.upsample(x)

        out += identity
        out = self.relu(out)

        return out


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.in_channels = 256
        self.layer4 = self.make_layer(ResNetBlockDecoder, 128, 1, stride=2, flag=0, upsample=nn.ConvTranspose2d(256, 128, kernel_size=1, stride=2))
        self.layer3 = self.make_layer(ResNetBlockDecoder, 64, 1, stride=2, flag=1, upsample=nn.ConvTranspose2d(128, 64, kernel_size=1, stride=2, output_padding=1))
        self.layer2 = self.make_layer(ResNetBlockDecoder, 32, 1, stride=2, flag=1, upsample=nn.ConvTranspose2d(64, 32, kernel_size=1, stride=2, output_padding=1))
        self.layer1 = self.make_layer(ResNetBlockDecoder, 1, 1, stride=1, flag=0, upsample=nn.ConvTranspose2d(32, 1, kernel_size=1, stride=1))

    def make_layer(self, block, out_channels, blocks, stride=1, flag=0, upsample=None):
        layers = []
        layers.append(block(self.in_channels, out_channels, stride, flag, upsample))
        self.in_channels = out_channels
        for _ in range(1, blocks):
            layers.append(block(self.in_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x, flag=0):
      if flag==0:
        #print(x.shape)
        x = self.layer4(x)
        #print(x.shape)
        x = self.layer3(x)
        #print(x.shape)
        x = self.layer2(x)
        #print(x.shape)
        x = self.layer1(x)
        #print(x.shape)

      if flag==1:
        #print(x.shape)
        x = self.layer4(x)
        #print(x.shape)
        x = self.layer3(x)
        #print(x.shape)
        x = self.layer2(x)
        #print(x.shape)
        x = self.layer1(x)
        #print(x.shape)

      if flag==2:
        #print(x.shape)
        x = self.layer4(x)
        #print(x.shape)
        x = self.layer3(x)
        #print(x.shape)
        x = self.layer2(x)
        #print(x.shape)
        x = self.layer1(x)
        #print(x.shape)
      return x

class AELossFn:
    def __init__(self):
        self.criterion = nn.MSELoss()

    def __call__(self, x, y):
        return self.criterion(x, y)

class VAELossFn:
    def __init__(self):
        pass

    def __call__(self, recon_x, x, mu, logvar):
        # Reconstruction loss (MSE)
        recon_loss = F.mse_loss(recon_x, x, reduction='sum')

        # KL divergence loss
        kl_loss = torch.mean(-0.5 * torch.sum(1 + logvar - mu ** 2 - logvar.exp(), dim=1), dim=0)
        kl_weight = 0.00025

        return recon_loss + kl_loss * kl_weight


def ParameterSelector(E, D):
    return list(E.parameters()) + list(D.parameters())


class AETrainer:
    def __init__(self, dataloader, encoder, decoder, loss_fn, optimizer, device='T'):
        self.dataloader = dataloader
        self.encoder = encoder
        self.decoder = decoder
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        if device == 'T':
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = 'cpu'
        self.encoder.to(self.device)
        self.decoder.to(self.device)

        self.best_ssim = 0.0  # Initialize best SSIM
        self.best_model_state = None  # Initialize best model state dictionary
        self.train()

    def train(self, num_epochs=50):
        for epoch in range(num_epochs):
            self.encoder.train()
            self.decoder.train()
            running_loss = 0.0
            running_ssim = 0.0

            for i, (inputs, targets, labels) in enumerate(self.dataloader):
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)

                self.optimizer.zero_grad()

                encoded, _, _ = self.encoder(inputs, flag=0)
                outputs = self.decoder(encoded, flag=0)

                loss = self.loss_fn(outputs, targets)
                loss.backward()
                self.optimizer.step()

                running_loss += loss.item()
                running_ssim += self.calculate_ssim(outputs, targets)

                if (i + 1) % 10 == 0:
                    print(">>>>> Epoch:{}, Minibatch:{}, Loss:{:.4f}, Similarity:{:.4f}".format(
                        epoch + 1, i + 1, loss.item(), running_ssim / (i + 1)))

            epoch_loss = running_loss / len(self.dataloader)
            epoch_ssim = running_ssim / len(self.dataloader)

            print("----- Epoch:{}, Loss:{:.4f}, Similarity:{:.4f}".format(epoch + 1, epoch_loss, epoch_ssim))

            if epoch_ssim > self.best_ssim:
                self.best_ssim = epoch_ssim
                self.best_model_state = {
                    'encoder_state_dict': self.encoder.state_dict(),
                    'decoder_state_dict': self.decoder.state_dict()
                }

            if (epoch + 1) % 10 == 0:
                self.plot_tsne_embeddings(epoch + 1)

        # Save the best model based on SSIM
        torch.save(self.best_model_state, 'best_modelAE_ssim.pth')

    def calculate_ssim(self, outputs, targets):
        outputs_np = outputs.detach().cpu().numpy().squeeze()
        targets_np = targets.detach().cpu().numpy().squeeze()
        batch_ssim = np.mean([ssim(output, target, data_range=target.max() - target.min()) for output, target in zip(outputs_np, targets_np)])
        return batch_ssim

    def plot_tsne_embeddings(self, epoch):
        embeddings = []
        labels = []
        self.encoder.eval()
        with torch.no_grad():
            # Collect all embeddings and labels
            for inputs, targets, labels_batch in self.dataloader:
                inputs = inputs.to(self.device)
                encoded, _, _ = self.encoder(inputs, flag=0)
                embeddings.append(encoded.view(encoded.size(0), -1).cpu().detach().numpy())
                labels.append(labels_batch.cpu().detach().numpy())

        embeddings = np.concatenate(embeddings, axis=0)
        labels = np.concatenate(labels, axis=0)


        # Reduce the number of data points for TSNE computation
        num_samples = min(5000, len(embeddings))  # Adjust the number of samples as needed
        indices = np.random.choice(len(embeddings), num_samples, replace=False)
        embeddings_subset = embeddings[indices]
        labels_subset = labels[indices]

        tsne = TSNE(n_components=3, perplexity=30, learning_rate=200, metric='euclidean', random_state=42)
        embeddings_3d = tsne.fit_transform(embeddings_subset)

        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')

        scatter = ax.scatter(embeddings_3d[:, 0], embeddings_3d[:, 1], embeddings_3d[:, 2], c=labels_subset, cmap='viridis', s=20)

        ax.set_xlabel('Dimension 1')
        ax.set_ylabel('Dimension 2')
        ax.set_zlabel('Dimension 3')
        ax.set_title('3D t-SNE Embeddings')

        legend1 = ax.legend(*scatter.legend_elements(), title="Labels")
        ax.add_artist(legend1)

        plt.show()
        plt.savefig("AE_epoch_{}.png".format(epoch))
        plt.close()


class VAETrainer:

    def __init__(self, dataloader, encoder, decoder, loss_fn, optimizer, device='T'):
        self.dataloader = dataloader
        self.encoder = encoder
        self.decoder = decoder
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        if device == 'T':
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = 'cpu'

        # Move encoder, decoder, and fc_mu, fc_logvar layers to the appropriate device
        self.encoder.to(self.device)
        self.decoder.to(self.device)

        self.best_ssim = 0.0  # Initialize best SSIM
        self.best_model_state = None  # Initialize best model state dictionary
        self.train()


    def train(self, num_epochs=50):
        for epoch in range(num_epochs):
            self.encoder.train()
            self.decoder.train()
            running_loss = 0.0
            running_ssim = 0.0

            for i, (inputs, targets, labels) in enumerate(self.dataloader):
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)

                self.optimizer.zero_grad()

                encoded, mu, logvar = self.encoder(inputs, flag=1)
                outputs = self.decoder(encoded, flag=1)

                loss = self.loss_fn(outputs, targets, mu, logvar)
                loss.backward()
                self.optimizer.step()

                running_loss += loss.item()
                running_ssim += self.calculate_ssim(outputs, targets)

                if (i + 1) % 10 == 0:
                    print(">>>>> Epoch:{}, Minibatch:{}, Loss:{:.4f}, Similarity:{:.4f}".format(
                        epoch + 1, i + 1, loss.item(), running_ssim / (i + 1)))

            epoch_loss = running_loss / len(self.dataloader)
            epoch_ssim = running_ssim / len(self.dataloader)

            print("----- Epoch:{}, Loss:{:.4f}, Similarity:{:.4f}".format(epoch + 1, epoch_loss, epoch_ssim))

            if epoch_ssim > self.best_ssim:
                self.best_ssim = epoch_ssim
                self.best_model_state = {
                    'encoder_state_dict': self.encoder.state_dict(),
                    'decoder_state_dict': self.decoder.state_dict()
                }

            if (epoch + 1) % 10 == 0:
                self.plot_tsne_embeddings(epoch + 1)

        # Save the best model based on SSIM
        torch.save(self.best_model_state, 'best_modelVAE_ssim.pth')

    def calculate_ssim(self, outputs, targets):
        outputs_np = outputs.detach().cpu().numpy().squeeze()
        targets_np = targets.detach().cpu().numpy().squeeze()
        batch_ssim = np.mean([ssim(output, target, data_range=target.max() - target.min()) for output, target in zip(outputs_np, targets_np)])
        return batch_ssim

    def plot_tsne_embeddings(self, epoch):
        embeddings = []
        labels = []
        self.encoder.eval()
        with torch.no_grad():
            # Collect all embeddings and labels
            for inputs, targets, labels_batch in self.dataloader:
                inputs = inputs.to(self.device)
                encoded, _, _ = self.encoder(inputs, flag=1)
                embeddings.append(encoded.view(encoded.size(0), -1).cpu().detach().numpy())
                labels.append(labels_batch.cpu().detach().numpy())

        embeddings = np.concatenate(embeddings, axis=0)
        labels = np.concatenate(labels, axis=0)


        # Reduce the number of data points for TSNE computation
        num_samples = min(5000, len(embeddings))  # Adjust the number of samples as needed
        indices = np.random.choice(len(embeddings), num_samples, replace=False)
        embeddings_subset = embeddings[indices]
        labels_subset = labels[indices]

        tsne = TSNE(n_components=3, perplexity=30, learning_rate=200, metric='euclidean', random_state=42)
        embeddings_3d = tsne.fit_transform(embeddings_subset)

        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')

        scatter = ax.scatter(embeddings_3d[:, 0], embeddings_3d[:, 1], embeddings_3d[:, 2], c=labels_subset, cmap='viridis', s=20)

        ax.set_xlabel('Dimension 1')
        ax.set_ylabel('Dimension 2')
        ax.set_zlabel('Dimension 3')
        ax.set_title('3D t-SNE Embeddings')

        legend1 = ax.legend(*scatter.legend_elements(), title="Labels")
        ax.add_artist(legend1)

        plt.show()
        plt.savefig("VAE_epoch_{}.png".format(epoch))
        plt.close()


class AE_TRAINED:

    def __init__(self, gpu='T'):
        self.gpu = gpu
        self.encoder = Encoder()
        self.decoder = Decoder()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        if self.gpu:
            self.encoder.cuda()
            self.decoder.cuda()
        self.from_checkpoint('best_modelAE_ssim.pth')

    def from_checkpoint(self, checkpoint_path):
        state_dict = torch.load(checkpoint_path)
        self.encoder.load_state_dict(state_dict['encoder_state_dict'])
        self.decoder.load_state_dict(state_dict['decoder_state_dict'])

    def from_path(self, sample, original, type):
        # Load sample and original images
        sample = io.read_image(sample)
        original = io.read_image(original)

        # Convert to greyscale
        sample = transforms.functional.rgb_to_grayscale(sample)
        original = transforms.functional.rgb_to_grayscale(original)

        # Convert to NumPy arrays
        sample = transforms.functional.to_pil_image(sample)
        original = transforms.functional.to_pil_image(original)

        transform = transforms.Compose([
            transforms.ToTensor(),
        ])

        sample = transform(sample)
        original = transform(original)

        # Forward pass
        output = self.forward_pass(sample)

        if type == "SSIM":
            return self.compute_ssim(output, original)
        elif type == "PSNR":
            return self.compute_psnr(output, original)

    def forward_pass(self, sample):
        self.encoder.eval()
        self.decoder.eval()
        if self.gpu:
            sample = sample.cuda()
        with torch.no_grad():
            encoded, _, _= self.encoder(sample.unsqueeze(0), flag=0)  # Add batch dimension
            output = self.decoder(encoded, flag=0)
        return output.squeeze(0)  # Remove batch dimension

    @staticmethod
    def compute_ssim(sample, original):
        # Convert tensors to numpy arrays
        sample_np = sample.detach().cpu().numpy().squeeze()
        original_np = original.detach().cpu().numpy().squeeze()
        # Compute SSIM score
        ssim_score = ssim(sample_np, original_np, data_range=original_np.max() - original_np.min())
        return ssim_score

    @staticmethod
    def compute_psnr(sample, original):
        # Convert tensors to numpy arrays
        sample_np = sample.detach().cpu().numpy().squeeze()
        original_np = original.detach().cpu().numpy().squeeze()
        # Compute PSNR score
        psnr_score = psnr(original_np, sample_np, data_range=original_np.max() - original_np.min())
        return psnr_score

class VAE_TRAINED:
    def __init__(self, gpu='T'):
        self.gpu = gpu
        self.encoder = Encoder()  # Initialize your encoder architecture
        self.decoder = Decoder()  # Initialize your decoder architecture
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        if self.gpu:
            self.encoder.cuda()
            self.decoder.cuda()


        self.from_checkpoint('best_modelVAE_ssim.pth')

    def from_checkpoint(self, checkpoint_path):
        state_dict = torch.load(checkpoint_path)
        self.encoder.load_state_dict(state_dict['encoder_state_dict'])
        self.decoder.load_state_dict(state_dict['decoder_state_dict'])

    def from_path(self, sample, original, type):
        # Load sample and original images
        sample = io.read_image(sample)
        original = io.read_image(original)

        # Convert to greyscale
        sample = transforms.functional.rgb_to_grayscale(sample)
        original = transforms.functional.rgb_to_grayscale(original)

        # Convert to NumPy arrays
        sample = transforms.functional.to_pil_image(sample)
        original = transforms.functional.to_pil_image(original)

        transform = transforms.Compose([
            transforms.ToTensor(),
        ])

        sample = transform(sample)
        original = transform(original)

        # Forward pass
        output = self.forward_pass(sample)

        if type == "SSIM":
            return self.compute_ssim(output, original)
        elif type == "PSNR":
            return self.compute_psnr(output, original)

    def forward_pass(self, sample):
        self.encoder.eval()
        self.decoder.eval()
        if self.gpu:
            sample = sample.cuda()
        with torch.no_grad():
            encoded, _, _ = self.encoder(sample.unsqueeze(0), flag=1)
            output = self.decoder(encoded, flag=1)
        return output.squeeze(0)

    @staticmethod
    def compute_ssim(sample, original):
        # Convert tensors to numpy arrays
        sample_np = sample.detach().cpu().numpy().squeeze()
        original_np = original.detach().cpu().numpy().squeeze()
        # Compute SSIM score
        ssim_score = ssim(sample_np, original_np, data_range=original_np.max() - original_np.min())
        return ssim_score

    @staticmethod
    def compute_psnr(sample, original):
        # Convert tensors to numpy arrays
        sample_np = sample.detach().cpu().numpy().squeeze()
        original_np = original.detach().cpu().numpy().squeeze()
        # Compute PSNR score
        psnr_score = psnr(original_np, sample_np, data_range=original_np.max() - original_np.min())
        return psnr_score


class CVAELossFn():
    def __init__(self):
        pass

    def __call__(self, recon_x, x, mu, logvar):
        # Reconstruction loss (MSE)
        recon_loss = F.mse_loss(recon_x, x, reduction='sum')

        # KL divergence loss
        kl_loss = torch.mean(-0.5 * torch.sum(1 + logvar - mu ** 2 - logvar.exp(), dim=1), dim=0)
        kl_weight = 0.00025

        return recon_loss + kl_loss * kl_weight

class CVAE_Trainer:
    def __init__(self, dataloader, encoder, decoder, loss_fn, optimizer):
        self.dataloader = dataloader
        self.encoder = encoder
        self.decoder = decoder
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # Move encoder, decoder, and fc_mu, fc_logvar layers to the appropriate device
        self.encoder.to(self.device)
        self.decoder.to(self.device)

        self.best_ssim = 0.0  # Initialize best SSIM
        self.best_model_state = None  # Initialize best model state dictionary
        self.train()

    def train(self, num_epochs=50):
        for epoch in range(num_epochs):
            self.encoder.train()
            self.decoder.train()
            running_loss = 0.0
            running_ssim = 0.0

            for i, (inputs, targets, labels) in enumerate(self.dataloader):
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                labels = labels.to(self.device)

                self.optimizer.zero_grad()

                encoded, mu, logvar = self.encoder(inputs, flag=2, labels=labels)
                outputs = self.decoder(encoded, flag=2)

                loss = self.loss_fn(outputs, targets, mu, logvar)
                loss.backward()
                self.optimizer.step()

                running_loss += loss.item()
                running_ssim += self.calculate_ssim(outputs, targets)

                if (i + 1) % 10 == 0:
                    print(">>>>> Epoch:{}, Minibatch:{}, Loss:{:.4f}, Similarity:{:.4f}".format(
                        epoch + 1, i + 1, loss.item(), running_ssim / (i + 1)))

            epoch_loss = running_loss / len(self.dataloader)
            epoch_ssim = running_ssim / len(self.dataloader)

            print("----- Epoch:{}, Loss:{:.4f}, Similarity:{:.4f}".format(epoch + 1, epoch_loss, epoch_ssim))

            if epoch_ssim > self.best_ssim:
                self.best_ssim = epoch_ssim
                self.best_model_state = {
                    'encoder_state_dict': self.encoder.state_dict(),
                    'decoder_state_dict': self.decoder.state_dict()
                }

            if (epoch + 1) % 10 == 0:
                self.plot_tsne_embeddings(epoch + 1)

        # Save the best model based on SSIM
        torch.save(self.best_model_state, 'best_modelCVAE_ssim.pth')

    def calculate_ssim(self, outputs, targets):
        outputs_np = outputs.detach().cpu().numpy().squeeze()
        targets_np = targets.detach().cpu().numpy().squeeze()
        batch_ssim = np.mean([ssim(output, target, data_range=target.max() - target.min()) for output, target in zip(outputs_np, targets_np)])
        return batch_ssim

    def plot_tsne_embeddings(self, epoch):
        embeddings = []
        labels = []
        self.encoder.eval()
        with torch.no_grad():
            # Collect all embeddings and labels
            for inputs, targets, labels_batch in self.dataloader:
                inputs = inputs.to(self.device)
                labels_batch = labels_batch.to(self.device)
                encoded, _, _ = self.encoder(inputs, flag=2, labels=labels_batch)
                embeddings.append(encoded.view(encoded.size(0), -1).cpu().detach().numpy())
                labels.append(labels_batch.cpu().detach().numpy())

        embeddings = np.concatenate(embeddings, axis=0)
        labels = np.concatenate(labels, axis=0)


        # Reduce the number of data points for TSNE computation
        num_samples = min(5000, len(embeddings))  # Adjust the number of samples as needed
        indices = np.random.choice(len(embeddings), num_samples, replace=False)
        embeddings_subset = embeddings[indices]
        labels_subset = labels[indices]

        tsne = TSNE(n_components=3, perplexity=30, learning_rate=200, metric='euclidean', random_state=42)
        embeddings_3d = tsne.fit_transform(embeddings_subset)

        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')

        scatter = ax.scatter(embeddings_3d[:, 0], embeddings_3d[:, 1], embeddings_3d[:, 2], c=labels_subset, cmap='viridis', s=20)

        ax.set_xlabel('Dimension 1')
        ax.set_ylabel('Dimension 2')
        ax.set_zlabel('Dimension 3')
        ax.set_title('3D t-SNE Embeddings')

        legend1 = ax.legend(*scatter.legend_elements(), title="Labels")
        ax.add_artist(legend1)

        plt.show()
        plt.savefig("CVAE_epoch_{}.png".format(epoch))
        plt.close()

class CVAE_Generator:
    def __init__(self):
        self.encoder = Encoder()
        self.decoder = Decoder()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.encoder.to(self.device)
        self.decoder.to(self.device)

        self.from_checkpoint('best_modelCVAE_ssim.pth')

    def from_checkpoint(self, checkpoint_path):
        state_dict = torch.load(checkpoint_path)
        self.encoder.load_state_dict(state_dict['encoder_state_dict'])
        self.decoder.load_state_dict(state_dict['decoder_state_dict'])


    def save_image(self, digit, save_path):
        #given digit is an integer and we need to use it to generate an image
        # Convert digit to one-hot encoding
        label = torch.tensor(int(digit)).unsqueeze(0)
        if self.device == 'cuda':
            label = label.cuda()

        # Generate image
        self.encoder.eval()
        self.decoder.eval()
        with torch.no_grad():
            encoded = torch.randn(1, 64).to(self.device)
            #pass one hot encoded label
            label = F.one_hot(label, num_classes=10)
            encoded = self.encoder.condition_on_label(encoded, label)
            encoded = self.encoder.fc(encoded) 
            encoded = encoded.view(-1, 256, 4, 4)
            output = self.decoder(encoded, flag=2)
        # Save image
        vutils.save_image(output, save_path, normalize=True)

def peak_signal_to_noise_ratio(img1, img2):
    if img1.shape[0] != 1: raise Exception("Image of shape [1,H,W] required.")
    img1, img2 = img1.to(torch.float64), img2.to(torch.float64)
    mse = img1.sub(img2).pow(2).mean()
    if mse == 0: return float("inf")
    else: return 20 * torch.log10(255.0/torch.sqrt(mse)).item()

def structure_similarity_index(img1, img2):
    if img1.shape[0] != 1: raise Exception("Image of shape [1,H,W] required.")
    # Constants
    window_size, channels = 11, 1
    K1, K2, DR = 0.01, 0.03, 255
    C1, C2 = (K1*DR)**2, (K2*DR)**2

    window = torch.randn(11)
    window = window.div(window.sum())
    window = window.unsqueeze(1).mul(window.unsqueeze(0)).unsqueeze(0).unsqueeze(0)

    mu1 = F.conv2d(img1, window, padding=window_size//2, groups=channels)
    mu2 = F.conv2d(img2, window, padding=window_size//2, groups=channels)
    mu12 = mu1.pow(2).mul(mu2.pow(2))

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size//2, groups=channels) - mu1.pow(2)
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size//2, groups=channels) - mu2.pow(2)
    sigma12 =  F.conv2d(img1 * img2, window, padding=window_size//2, groups=channels) - mu12


    SSIM_n = (2 * mu1 * mu2 + C1) * (2 * sigma12 + C2)
    denom = ((mu1**2 + mu2**2 + C1) * (sigma1_sq + sigma2_sq + C2))
    return torch.clamp((1 - SSIM_n / (denom + 1e-8)), min=0.0, max=1.0).mean().item()