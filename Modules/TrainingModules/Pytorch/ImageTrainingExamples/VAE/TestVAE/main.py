import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import numpy as np
from PIL import Image

# Custom dataset class with black images
class CustomDataset(Dataset):
    def __init__(self, num_samples, image_size):
        self.num_samples = num_samples
        self.image_size = image_size

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # Generate black image
        image = np.zeros((self.image_size, self.image_size), dtype=np.uint8)
        image = Image.fromarray(image)
        transform = transforms.Compose([transforms.Resize((24, 24)), transforms.ToTensor()])
        image = transform(image)
        return image

# Define the VAE architecture for 28x28 images
class VAE(nn.Module):
    def __init__(self, input_size, latent_size):
        super(VAE, self).__init__()

        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(128 * (input_size // 8) * (input_size // 8), 256),
            nn.ReLU(),
            nn.Linear(256, latent_size * 2)  # Multiply by 2 for mean and log-variance
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_size, 256),
            nn.ReLU(),
            nn.Linear(256, 128 * (input_size // 8) * (input_size // 8)),
            nn.ReLU(),
            nn.Unflatten(1, (128, input_size // 8, input_size // 8)),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 1, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid()  # Output values between 0 and 1 for image pixels
        )

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        # Encode
        enc_output = self.encoder(x)
        mu, log_var = enc_output.chunk(2, dim=1)
        z = self.reparameterize(mu, log_var)

        # Decode
        dec_output = self.decoder(z)

        return dec_output, mu, log_var

# Loss function for VAE
def vae_loss(x, recon_x, mu, log_var):
    # Reconstruction loss
    recon_loss = nn.functional.binary_cross_entropy(recon_x, x, reduction='sum')

    # KL divergence loss
    kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())

    # Total VAE loss
    total_loss = recon_loss + kl_loss

    return total_loss

# Set up data loader with custom dataset
custom_dataset = CustomDataset(num_samples=1000, image_size=28)  # Adjust num_samples as needed
dataloader = DataLoader(custom_dataset, batch_size=64, shuffle=True, num_workers=4)

# Initialize VAE model
input_size = 28  # Assuming MNIST-like images with size 28x28
latent_size = 2  # 2D latent space for visualization
vae = VAE(input_size, latent_size)

# Set up optimizer
optimizer = optim.Adam(vae.parameters(), lr=0.001)

# Training loop
num_epochs = 10
for epoch in range(num_epochs):
    for batch_idx, data in enumerate(dataloader):
        optimizer.zero_grad()

        recon_data, mu, log_var = vae(data)
        loss = vae_loss(data, recon_data, mu, log_var)

        loss.backward()
        optimizer.step()

        if batch_idx % 100 == 0:
            print(f'Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item()}')

# Save the trained model
torch.save(vae.state_dict(), 'vae_model.pth')