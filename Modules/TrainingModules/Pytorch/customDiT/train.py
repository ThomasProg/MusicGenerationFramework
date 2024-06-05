#!/usr/bin/env python3

import copy
from dataclasses import asdict

import numpy as np
import torch
import torchvision
import torchvision.utils as vutils
import wandb
from accelerate import Accelerator
from diffusers import AutoencoderKL
from PIL.Image import Image
from torch import Tensor, nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from denoiser import Denoiser
from diffusion import DiffusionGenerator
from configs import ModelConfig


def eval_gen(diffuser: DiffusionGenerator, labels: Tensor, img_size: int) -> Image:
    class_guidance = 4.5
    seed = 10
    out, _ = diffuser.generate(
        labels=torch.repeat_interleave(labels, 2, dim=0),
        num_imgs=16,
        class_guidance=class_guidance,
        seed=seed,
        n_iter=40,
        exponent=1,
        sharp_f=0.1,
        img_size=img_size
    )

    out = to_pil((vutils.make_grid((out + 1) / 2, nrow=8, padding=4)).float().clip(0, 1))
    out.save(f"emb_val_cfg:{class_guidance}_seed:{seed}.png")

    return out


def count_parameters(model: nn.Module):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def count_parameters_per_layer(model: nn.Module):
    for name, param in model.named_parameters():
        print(f"{name}: {param.numel()} parameters")


to_pil = torchvision.transforms.ToPILImage()


def update_ema(ema_model: nn.Module, model: nn.Module, alpha: float = 0.999):
    with torch.no_grad():
        for ema_param, model_param in zip(ema_model.parameters(), model.parameters()):
            ema_param.data.mul_(alpha).add_(model_param.data, alpha=1 - alpha)



def main(config: ModelConfig) -> None:
    """main train loop to be used with accelerate"""
    denoiser_config = config.denoiser_config
    train_config = config.train_config
    dataconfig = config.data_config

    log_with="wandb" if train_config.use_wandb else None
    accelerator = Accelerator(mixed_precision="fp16", log_with=log_with)

    accelerator.print("Loading Data:")
    # latent_train_data = torch.tensor(np.load(dataconfig.latent_path), dtype=torch.float32)
    # train_label_embeddings = torch.tensor(np.load(dataconfig.text_emb_path), dtype=torch.float32)
    # emb_val = torch.tensor(np.load(dataconfig.val_path), dtype=torch.float32)
    emb_val = torch.randn((768//4, 1)) # None
    dataset = dataconfig.dataset #TensorDataset(latent_train_data, train_label_embeddings)
    
    # Example batch of input data (images)
    input_data = torch.randn(32, 3, 64, 64)  # Assuming batch size of 32, RGB images of size 64x64

    # Example batch of labels (classification)
    labels = torch.randint(0, 10, (32,), dtype=torch.long)  # Assuming 10 classes, batch size of 32

    # Wrap input data and labels into a dataset
    dataset = TensorDataset(input_data, labels)

    train_loader = DataLoader(dataset, batch_size=train_config.batch_size, shuffle=True)

    vae = None
    # vae = AutoencoderKL.from_pretrained(config.vae_cfg.vae_name, torch_dtype=config.vae_cfg.vae_dtype)

    # if accelerator.is_main_process:
    #     vae = vae.to(accelerator.device)

    model = Denoiser(**asdict(denoiser_config))

    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=train_config.lr)

    if train_config.compile:
        accelerator.print("Compiling model:")
        model = torch.compile(model)

    if not train_config.from_scratch:
        accelerator.print("Loading Model:")
        wandb.restore(
            train_config.model_name, run_path=f"apapiu/cifar_diffusion/runs/{train_config.run_id}", replace=True
        )
        full_state_dict = torch.load(train_config.model_name)
        model.load_state_dict(full_state_dict["model_ema"])
        optimizer.load_state_dict(full_state_dict["opt_state"])
        global_step = full_state_dict["global_step"]
    else:
        global_step = 0

    if accelerator.is_local_main_process:
        ema_model = copy.deepcopy(model).to(accelerator.device)
        diffuser = DiffusionGenerator(ema_model, vae, accelerator.device, torch.float32)

    accelerator.print("model prep")
    model, train_loader, optimizer = accelerator.prepare(model, train_loader, optimizer)

    if train_config.use_wandb:
        accelerator.init_trackers(project_name="cifar_diffusion", config=asdict(config))

    accelerator.print(count_parameters(model))
    accelerator.print(count_parameters_per_layer(model))

    ### Train:
    for i in range(1, train_config.n_epoch + 1):
        accelerator.print(f"epoch: {i}")

        for x, y in tqdm(train_loader):
            if (vae != None):
                x = x / config.vae_cfg.vae_scale_factor

            noise_level = torch.tensor(
                np.random.beta(train_config.beta_a, train_config.beta_b, len(x)), device=accelerator.device
            )
            signal_level = 1 - noise_level
            noise = torch.randn_like(x)

            x_noisy = noise_level.view(-1, 1, 1, 1) * noise + signal_level.view(-1, 1, 1, 1) * x

            x_noisy = x_noisy.float()
            noise_level = noise_level.float()
            label = y

            prob = 0.15
            mask = torch.rand(y.size(0), device=accelerator.device) < prob
            label[mask] = 0  # OR replacement_vector

            if global_step % train_config.save_and_eval_every_iters == 0:
                accelerator.wait_for_everyone()
                if accelerator.is_main_process:
                    ##eval and saving:
                    # out = eval_gen(diffuser=diffuser, labels=emb_val, img_size=denoiser_config.image_size)
                    # out.save("img.jpg")
                    if train_config.use_wandb:
                        accelerator.log({f"step: {global_step}": wandb.Image("img.jpg")})

                    opt_unwrapped = accelerator.unwrap_model(optimizer)
                    full_state_dict = {
                        "model_ema": ema_model.state_dict(),
                        "opt_state": opt_unwrapped.state_dict(),
                        "global_step": global_step,
                    }
                    if train_config.save_model:
                        accelerator.save(full_state_dict, train_config.model_name)
                        if train_config.use_wandb:
                            wandb.save(train_config.model_name)

            model.train()

            with accelerator.accumulate():
                ###train loop:
                optimizer.zero_grad()

                pred = model(x_noisy, noise_level.view(-1, 1), label)
                loss = loss_fn(pred, x)
                accelerator.log({"train_loss": loss.item()}, step=global_step)
                accelerator.backward(loss)
                optimizer.step()

                if accelerator.is_main_process:
                    update_ema(ema_model, model, alpha=train_config.alpha)

            global_step += 1
    accelerator.end_training()

    print("end of training")


# args = (config, data_path, val_path)
# notebook_launcher(training_loop)






# from train import main
from configs import DataConfig, ModelConfig, TrainConfig, DenoiserConfig
from accelerate import notebook_launcher
import pathlib

run_id='' #@param {type:"string"}
n_epoch=40 #@param {type:"integer"}


import os
os.environ["HF_DATASETS_OFFLINE"] = "1"

from datasets import load_dataset
dataset = load_dataset("nelorth/oxford-flowers")


# # Define your custom function
# def custom_function(img):
#     # Apply any custom transformation to the image
#     # img = img.rotate(45)  # Example: Rotate the image by 45 degrees
#     tmp = img['image']
#     for i in range(len(tmp)):
#         tmp[i].resize((256,256))
#         tmp[i] = torchvision.transforms.functional.pil_to_tensor(tmp[i])
#     return (tmp, img['label'])

def custom_function(sample):
    images = sample['image']
    resized_images = []
    for image in images:
        resized_image = torchvision.transforms.functional.resize(image, (256, 256))
        resized_image = torchvision.transforms.functional.pil_to_tensor(resized_image)
        resized_images.append(resized_image)

    labels = torch.stack([torch.tensor(l) for l in sample['label']])
    
    
    return (torch.stack(resized_images), labels)



import torchvision.transforms as transforms
custom_transform = transforms.Compose([
    lambda x: custom_function(x),
    # transforms.Resize((256, 256)),  # Resize the image
    # transforms.ToTensor(),           # Convert the image to a PyTorch tensor
    # Add more transformations if needed
])

test = dataset["test"]
train = dataset["train"]

train.set_transform(custom_transform)

print(train)

data_config = DataConfig(dataset=train)
# data_config = DataConfig(latent_path=train['image'],
#                         text_emb_path=train['label'])

denoiser_config = DenoiserConfig(image_size=32)

model_cfg = ModelConfig(
    data_config=data_config,
    denoiser_config=denoiser_config,
    train_config=TrainConfig(
        n_epoch=1,
        save_model=False,
        compile=True,
        use_wandb=False
        ),
)

notebook_launcher(main, (model_cfg,), num_processes=1)





