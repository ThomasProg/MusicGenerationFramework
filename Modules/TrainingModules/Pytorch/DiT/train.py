# Source : https://colab.research.google.com/github/huggingface/notebooks/blob/main/diffusers/training_example.ipynb#scrollTo=-vQPIkaSRDKF
# https://huggingface.co/docs/diffusers/main/en/tutorials/basic_training

shouldPlot = False
resume_from_checkpoint = True #"Assets/Models/ddpm-butterflies-32/unet/diffusion_pytorch_model.safetensors"



# Config


from dataclasses import dataclass

@dataclass
class TrainingConfig:
    image_size = 32  # the generated image resolution
    train_batch_size = 16
    eval_batch_size = 16  # how many images to sample during evaluation
    num_epochs = 50
    gradient_accumulation_steps = 1
    learning_rate = 1e-4
    lr_warmup_steps = 500
    save_image_epochs = 1
    save_model_epochs = 1
    mixed_precision = 'fp16'  # `no` for float32, `fp16` for automatic mixed precision
    output_dir = 'Assets/Models/dit-flowers-32'  # the model namy locally and on the HF Hub

    push_to_hub = False  # whether to upload the saved model to the HF Hub
    hub_private_repo = False  
    overwrite_output_dir = True  # overwrite the old model when re-running the notebook
    seed = 0
    num_train_timesteps=1000

config = TrainingConfig()






# Loading dataset

from datasets import load_dataset

config.dataset_name = "huggan/smithsonian_butterflies_subset"
dataset = load_dataset(config.dataset_name, split="train")

# Feel free to try other datasets from https://hf.co/huggan/ too! 
# Here's is a dataset of flower photos:
# config.dataset_name = "huggan/flowers-102-categories"
# dataset = load_dataset(config.dataset_name, split="train")

# Or just load images from a local folder!
# config.dataset_name = "imagefolder"
# dataset = load_dataset(config.dataset_name, data_dir="path/to/folder")





import matplotlib.pyplot as plt

if shouldPlot:
    fig, axs = plt.subplots(1, 4, figsize=(16, 4))
    for i, image in enumerate(dataset[:4]["image"]):
        axs[i].imshow(image)
        axs[i].set_axis_off()
    fig.show()
    plt.show()







# Preprocess Data

from torchvision import transforms

preprocess = transforms.Compose(
    [
        transforms.Resize((config.image_size, config.image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ]
)

def transform(examples):
    # print("Aaa ", examples["image"][0])
    images = [preprocess(image.convert("RGB")) for image in examples["image"]]
    return {"images": images}

dataset.set_transform(transform)


if shouldPlot:
    fig, axs = plt.subplots(1, 4, figsize=(16, 4))
    for i, image in enumerate(dataset[:4]["images"]):
        axs[i].imshow(image.permute(1, 2, 0).numpy() / 2 + 0.5)
        axs[i].set_axis_off()
    fig.show()
    plt.show()




import torch

train_dataloader = torch.utils.data.DataLoader(dataset, batch_size=config.train_batch_size, shuffle=True)


if False:
    import PIL
    import numpy as np
    from datasets import Dataset

    image = PIL.Image.open("Assets/Datasets/Flowers102/flowers-102/jpg/image_00001.jpg")

    plt.imshow(image)
    plt.show()


    # image_array = np.array(image)
    data_dict = {"image": ([image]*5000)}
    dataset = Dataset.from_dict(data_dict)
    dataset.set_format(type="numpy")

    preprocess = transforms.Compose(
        [
            transforms.Resize((config.image_size, config.image_size)),
            # transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )

    def transform(examples):
        images = [preprocess(image.convert("RGB")) for image in examples["image"]]
        return {"images": images}

    dataset.set_transform(transform)

    train_dataloader = torch.utils.data.DataLoader(dataset, batch_size=config.train_batch_size, shuffle=True)

    fig, axs = plt.subplots(1, 4, figsize=(16, 4))
    for i, image in enumerate(dataset[:1]["images"]):
        axs[i].imshow(image.permute(1, 2, 0).numpy() / 2 + 0.5)
        axs[i].set_axis_off()
    fig.show()
    plt.show()


from diffusers import UNet2DModel


# model = UNet2DModel(
#     sample_size=config.image_size,  # the target image resolution
#     in_channels=3,  # the number of input channels, 3 for RGB images
#     out_channels=3,  # the number of output channels
#     layers_per_block=2,  # how many ResNet layers to use per UNet block
#     block_out_channels=(128, 128, 256, 256, 512, 512),  # the number of output channes for each UNet block
#     down_block_types=( 
#         "DownBlock2D",  # a regular ResNet downsampling block
#         "DownBlock2D", 
#         "DownBlock2D", 
#         "DownBlock2D", 
#         "AttnDownBlock2D",  # a ResNet downsampling block with spatial self-attention
#         "DownBlock2D",
#     ), 
#     up_block_types=(
#         "UpBlock2D",  # a regular ResNet upsampling block
#         "AttnUpBlock2D",  # a ResNet upsampling block with spatial self-attention
#         "UpBlock2D", 
#         "UpBlock2D", 
#         "UpBlock2D", 
#         "UpBlock2D"  
#       ),
# )

from diffusers import AutoencoderKL, DiTTransformer2DModel

dit_transformer = DiTTransformer2DModel(sample_size=32, in_channels=3, out_channels=3, num_attention_heads=4, num_layers=3)
# dit_transformer.config.out_channels = 3
dit_transformer.to("cuda")
# dit_transformer.config

# autoencoder = AutoencoderKL(input_dim=64*64*3, latent_dim=128)
autoencoder = AutoencoderKL(sample_size=32, in_channels=3, out_channels=3, latent_channels=3)
autoencoder.to("cuda")


# print("XXX : ", dataset[0]['images'])
sample_image = dataset[0]['images'].unsqueeze(0)
# print('Input shape:', sample_image.shape)
# print('Output shape:', model(sample_image, timestep=0).sample.shape)



# Noise Scheduler

from diffusers import DDIMScheduler

noise_scheduler = DDIMScheduler(num_train_timesteps=1000)
# noise_scheduler = DDIMScheduler(num_train_timesteps=200)



import torch
from PIL import Image

noise = torch.randn(sample_image.shape)
timesteps = torch.LongTensor([50])
noisy_image = noise_scheduler.add_noise(sample_image, noise, timesteps)

if shouldPlot:
    img = Image.fromarray(((noisy_image.permute(0, 2, 3, 1) + 1.0) * 127.5).type(torch.uint8).numpy()[0])

    plt.imshow(img)
    plt.show()




import torch.nn.functional as F

# noise_pred = model(noisy_image, timesteps).sample
# loss = F.mse_loss(noise_pred, noise)

import torch.nn as nn
loss = criterion = nn.MSELoss()




# Training

# optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
optimizer = torch.optim.AdamW(list(dit_transformer.parameters()) + list(autoencoder.parameters()), lr=config.learning_rate)

from diffusers.optimization import get_cosine_schedule_with_warmup

lr_scheduler = get_cosine_schedule_with_warmup(
    optimizer=optimizer,
    num_warmup_steps=config.lr_warmup_steps,
    num_training_steps=(len(train_dataloader) * config.num_epochs),
)



# from diffusers import DDIMPipeline
from diffusers import DiTPipeline

import math
import os

def make_grid(images, rows, cols):
    w, h = images[0].size
    grid = Image.new('RGB', size=(cols*w, rows*h))
    for i, image in enumerate(images):
        grid.paste(image, box=(i%cols*w, i//cols*h))
    return grid

def evaluate(config, epoch, pipeline):
    # Sample some images from random noise (this is the backward diffusion process).
    # The default pipeline output type is `List[PIL.Image]`
    images = pipeline(
        class_labels = torch.tensor([1]).to("cuda"),
        num_inference_steps=25,
        # batch_size = config.eval_batch_size, 
        generator=torch.manual_seed(config.seed)
    ).images

    # Make a grid out of the images
    image_grid = make_grid(images, rows=1, cols=1)

    # Save the images
    test_dir = os.path.join(config.output_dir, "samples")
    os.makedirs(test_dir, exist_ok=True)
    image_grid.save(f"{test_dir}/{epoch:04d}.png")








from accelerate import Accelerator
from huggingface_hub import HfFolder, Repository, whoami

from tqdm.auto import tqdm
from pathlib import Path
import os
import json

def get_full_repo_name(model_id: str, organization: str = None, token: str = None):
    if token is None:
        token = HfFolder.get_token()
    if organization is None:
        username = whoami(token)["name"]
        return f"{username}/{model_id}"
    else:
        return f"{organization}/{model_id}"

def save_checkpoint(checkpoint_path, epoch, model, noise_scheduler, optimizer, lr_scheduler):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        # 'noise_scheduler_state_dict': noise_scheduler.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'lr_scheduler_state_dict': lr_scheduler.state_dict(),
    }, checkpoint_path + "/checkpoint")
    print("Checkpoint saved.")

def load_checkpoint(checkpoint_path, model, noise_scheduler, optimizer, lr_scheduler):
    checkpoint = torch.load(checkpoint_path + "/checkpoint")
    model.load_state_dict(checkpoint['model_state_dict'])
    # noise_scheduler.load_state_dict(checkpoint['noise_scheduler_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    lr_scheduler.load_state_dict(checkpoint['lr_scheduler_state_dict'])
    epoch = checkpoint['epoch']
    print(f"Checkpoint loaded. Resuming from epoch {epoch + 1}.")

    return epoch + 1

def train_loop(config, model, noise_scheduler, optimizer: torch.optim.Optimizer, train_dataloader, lr_scheduler):
    start_epoch = 0
    if resume_from_checkpoint and os.path.isfile(config.output_dir + "/checkpoint"):
        start_epoch = load_checkpoint(config.output_dir, model, noise_scheduler, optimizer, lr_scheduler)

    # Initialize accelerator and tensorboard logging
    accelerator = Accelerator(
        # mixed_precision=config.mixed_precision,
        gradient_accumulation_steps=config.gradient_accumulation_steps, 
        # log_with="tensorboard",
        project_dir=os.path.join(config.output_dir, "logs")
    )
    if accelerator.is_main_process:
        if config.push_to_hub:
            repo_name = get_full_repo_name(Path(config.output_dir).name)
            repo = Repository(config.output_dir, clone_from=repo_name)
        elif config.output_dir is not None:
            os.makedirs(config.output_dir, exist_ok=True)
        accelerator.init_trackers("train_example")
    
    # Prepare everything
    # There is no specific order to remember, you just need to unpack the 
    # objects in the same order you gave them to the prepare method.
    model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, lr_scheduler
    )
    
    global_step = 0

    # Now you train the model
    for epoch in range(start_epoch, config.num_epochs):
        progress_bar = tqdm(total=len(train_dataloader), disable=not accelerator.is_local_main_process)
        progress_bar.set_description(f"Epoch {epoch}")

        for step, batch in enumerate(train_dataloader):
            clean_images = batch['images']
            clean_images.to("cuda")
            # Sample noise to add to the images
            noise = torch.randn(clean_images.shape).to(clean_images.device)
            bs = clean_images.shape[0]

            # Sample a random timestep for each image
            timesteps = torch.randint(0, noise_scheduler.num_train_timesteps, (bs,), device=clean_images.device).long()

            # Add noise to the clean images according to the noise magnitude at each timestep
            # (this is the forward diffusion process)
            noisy_images = noise_scheduler.add_noise(clean_images, noise, timesteps)
            
            with accelerator.accumulate(model):
                # Predict the noise residual
                labels = torch.full(tuple([noisy_images.shape[0]]), 1).to(clean_images.device)
                noise_pred = model(hidden_states=noisy_images, timestep=timesteps, class_labels = labels, return_dict=False)[0]
                loss = F.mse_loss(noise_pred, noise)
                accelerator.backward(loss)

                accelerator.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
            
            progress_bar.update(1)
            logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0], "step": global_step}
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)
            global_step += 1

        # After each epoch you optionally sample some demo images with evaluate() and save the model
        if accelerator.is_main_process:
            # pipeline = DiTPipeline(unet=accelerator.unwrap_model(model), scheduler=noise_scheduler)
            # pipeline = DiTPipeline(unet=accelerator.unwrap_model(model), vae = None, scheduler=noise_scheduler)
            pipeline = DiTPipeline(transformer=dit_transformer, vae=autoencoder, scheduler=noise_scheduler)

            if (epoch + 1) % config.save_image_epochs == 0 or epoch == config.num_epochs - 1:
                evaluate(config, epoch, pipeline)

            if (epoch + 1) % config.save_model_epochs == 0 or epoch == config.num_epochs - 1:
                if config.push_to_hub:
                    repo.push_to_hub(commit_message=f"Epoch {epoch}", blocking=True)
                else:
                    save_checkpoint(config.output_dir, epoch, model, noise_scheduler, optimizer, lr_scheduler)



# Training
                
import accelerate
                    
args = (config, dit_transformer, noise_scheduler, optimizer, train_dataloader, lr_scheduler)

accelerate.notebook_launcher(train_loop, args, num_processes=1)


# Results

import glob

sample_images = sorted(glob.glob(f"{config.output_dir}/samples/*.png"))
Image.open(sample_images[-1])
