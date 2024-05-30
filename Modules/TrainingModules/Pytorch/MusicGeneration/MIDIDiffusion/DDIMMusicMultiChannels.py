# Source : https://colab.research.google.com/github/huggingface/notebooks/blob/main/diffusers/training_example.ipynb#scrollTo=-vQPIkaSRDKF
# https://huggingface.co/docs/diffusers/main/en/tutorials/basic_training

# Steps :
# 1. Load a midi file
# 2. Convert it numpy nd array
# 3. Build a dataset
# 4. Convert to tensor
# 5. Remap values to only have a subrange
# 6. Train model


# Config
from dataclasses import dataclass

@dataclass
class TrainingConfig:
    image_size = 16  # the generated image resolution
    train_batch_size = 16
    eval_batch_size = 1  # how many images to sample during evaluation
    num_epochs = 50
    gradient_accumulation_steps = 1
    learning_rate = 1e-4
    lr_warmup_steps = 500
    save_image_epochs = 1
    save_model_epochs = 1
    mixed_precision = 'fp16'  # `no` for float32, `fp16` for automatic mixed precision
    output_dir = 'Assets/Models/ddim-music-16-MultiChannels'  # the model namy locally and on the HF Hub

    push_to_hub = False  # whether to upload the saved model to the HF Hub
    hub_private_repo = False  
    overwrite_output_dir = True  # overwrite the old model when re-running the notebook
    seed = 0
    num_train_timesteps=1000

    loss_function = None


import torch.nn as nn
import torch

class DiffusionLoss(nn.Module):
    def __init__(self):
        super(DiffusionLoss, self).__init__()

    def forward(self, predicted_diffusion, target_diffusion):
        # Extract the mask and pitches images from the predicted_diffusion tensor
        predicted_mask = predicted_diffusion[:, 0, :, :]  # Assuming mask is the first image
        predicted_pitches = predicted_diffusion[:, 1, :, :]  # Assuming pitches is the second image
        
        # Extract the mask and pitches from the target diffusion tensor
        target_mask = target_diffusion[:, 0, :, :]  # Assuming mask is the first image
        target_pitches = target_diffusion[:, 1, :, :]  # Assuming pitches is the second image
        
        # Apply the mask to the pitches image
        masked_pitches = predicted_pitches * target_mask.unsqueeze(1)  # Broadcasting the mask to the pitches dimensions
        target_masked_pitches = target_pitches * target_mask.unsqueeze(1)  # Broadcasting the mask to the pitches dimensions

        # Calculate the difference between masked pitches and target pitches
        pitches_diff = masked_pitches - target_masked_pitches
        
        # Calculate the difference between predicted mask and target mask
        mask_diff = predicted_mask - target_mask
        
        # Compute the loss for pitches (e.g., mean squared error)
        pitches_loss = torch.mean(torch.square(pitches_diff))
        
        # Compute the loss for mask (e.g., mean squared error)
        mask_loss = torch.mean(torch.square(mask_diff))
        
        # Combine the losses (if needed)
        total_loss = pitches_loss + mask_loss
        
        return total_loss

config = TrainingConfig()

config.loss_function = DiffusionLoss()


import matplotlib.pyplot as plt

# Preprocess Data
from torchvision import transforms
import torch

import Helpers.LoadDataset

import numpy as np

def remapArray(array):
    npArray = array.numpy().astype(np.float32)

    return torch.from_numpy(npArray)

def toNdArray(array):
    return np.array(array)

preprocess = transforms.Compose(
    [
        transforms.Lambda(toNdArray),
        transforms.ToTensor(),
        transforms.Lambda(remapArray),
        transforms.Normalize([0.5], [0.5]),
    ]
)

from datasets import Dataset
from PyMIDIMusic import MIDIToVector

def loadDataset(config, preprocess):
    music, test, tokens = MIDIToVector.GetTokens()

    # print("BRFG : ", test.notesPerTiming)

    notesPerTiming = test.notesPerTiming

    # Cut beginning
    i = 0
    while (len(notesPerTiming[i]) == 0 and i < len(notesPerTiming)):
        i += 1

    # Do not cut sentences, start at the beginning of one
    # notesPerTiming = notesPerTiming[i // 16 * 16:-1]

    # Take a 16*16 image
    notesPerTiming = notesPerTiming[0:16*16]

    # print("BRFG : ", notesPerTiming)

    # tokens = np.empty(16*16*40)

    grid = np.empty(16*16*40)
    grid = grid.reshape((16, 16, 40))

    # tokens2 = np.empty(16*16*40)
    # tokens2 = tokens2.reshape((16, 16, 40))
    for j in range(grid.shape[0]):
        for i in range(grid.shape[1]):

            for k in range(grid.shape[2]):
                grid[j][i][k] = 0.0

            for k in range(len(notesPerTiming[j + i * grid.shape[0]])):
                v = notesPerTiming[j + i * grid.shape[0]][k]
                if (v >= 40 or v < 80):
                    grid[j][i][v-40] = 1.0

    import sys
    np.set_printoptions(threshold=sys.maxsize)
    print(grid)

    # Convert the list to a NumPy array
    new_array = np.array(grid)
    new_array = new_array.astype(dtype=np.float32)

    data_dict = {"image": ([new_array]*5000)}
    dataset = Dataset.from_dict(data_dict)
    dataset.set_format(type="numpy")

    def transform(examples):
        images = [preprocess(image) for image in examples["image"]]
        return {"images": images}

    dataset.set_transform(transform)

    train_dataloader = torch.utils.data.DataLoader(dataset, batch_size=config.train_batch_size, shuffle=True)

    return dataset, train_dataloader


dataset, train_dataloader = loadDataset(config, preprocess)

import os
# Save the images
sample_image = np.array(dataset[0]['images'].unsqueeze(0)).astype(dtype=np.float32)
# print("transformed: ", sample_image)
test_dir = os.path.join(config.output_dir, "tensor")
os.makedirs(test_dir, exist_ok=True)
if (len(sample_image) == 1):
    sample_image[0].tofile(f"{test_dir}.floatarray")




from diffusers import UNet2DModel


model = UNet2DModel(
    sample_size=config.image_size,  # the target image resolution
    in_channels=40,  # the number of input channels, 3 for RGB images
    out_channels=40,  # the number of output channels
    layers_per_block=3,  # how many ResNet layers to use per UNet block
    block_out_channels=(64, 128, 256),  # the number of output channes for each UNet block
    down_block_types=( 
        "DownBlock2D",  # a regular ResNet downsampling block
        # "DownBlock2D", 
        "DownBlock2D", 
        "DownBlock2D", 
        # "AttnDownBlock2D",  # a ResNet downsampling block with spatial self-attention
        # "DownBlock2D",
    ), 
    up_block_types=(
        # "UpBlock2D",  # a regular ResNet upsampling block
        # "AttnUpBlock2D",  # a ResNet upsampling block with spatial self-attention
        "UpBlock2D", 
        "UpBlock2D", 
        # "UpBlock2D", 
        "UpBlock2D"  
      ),
)

# Noise Scheduler

from diffusers import DDIMScheduler

noise_scheduler = DDIMScheduler(num_train_timesteps=1000)

import torch

sample_image = dataset[0]['images'].unsqueeze(0)
noise = torch.randn(sample_image.shape)
timesteps = torch.LongTensor([50])
noisy_image = noise_scheduler.add_noise(sample_image, noise, timesteps)



import torch.nn.functional as F

noise_pred = model(noisy_image, timesteps).sample
loss = F.mse_loss(noise_pred, noise)


# Training

optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)

from diffusers.optimization import get_cosine_schedule_with_warmup

lr_scheduler = get_cosine_schedule_with_warmup(
    optimizer=optimizer,
    num_warmup_steps=config.lr_warmup_steps,
    num_training_steps=(len(train_dataloader) * config.num_epochs),
)


# Training
                
import accelerate
import Helpers.Training
                    
args = (config, model, noise_scheduler, optimizer, train_dataloader, lr_scheduler)

accelerate.notebook_launcher(Helpers.Training.train_loop, args, num_processes=1)

