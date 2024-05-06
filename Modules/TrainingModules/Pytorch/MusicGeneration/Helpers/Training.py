from accelerate import Accelerator
from huggingface_hub import HfFolder, Repository, whoami

from tqdm.auto import tqdm
from pathlib import Path
import os
import torch

from diffusers import DDIMPipeline
import torch.nn.functional as F

import PIL
import numpy as np

def make_grid(images, rows, cols):
    w, h = images[0].size
    grid = PIL.Image.new('L', size=(cols*w, rows*h))
    for i, image in enumerate(images):
        grid.paste(image, box=(i%cols*w, i//cols*h))
    return grid

def evaluate(config, epoch, pipeline):
    # Sample some images from random noise (this is the backward diffusion process).
    # The default pipeline output type is `List[PIL.Image]`
    images = pipeline(
        batch_size = config.eval_batch_size, 
        generator=torch.manual_seed(config.seed),
        output_type = "np.array"
    ).images

    # for i in range(len(images)):
    #     print("im", np.asarray(images[i]))
    #     images[i] = unnormalize_image(images[i], 0.5, 0.5)




    # # Make a grid out of the images
    # # image_grid = make_grid(images, rows=(1 + config.eval_batch_size / 4), cols=4)
    # image_grid = make_grid(images, rows=1, cols=1)

    # Save the images
    test_dir = os.path.join(config.output_dir, "samples")
    os.makedirs(test_dir, exist_ok=True)
    # image_grid.save(f"{test_dir}/{epoch:04d}.png")





    # for (j,i),imageArray in np.ndenumerate(images):
    if (len(images) == 1):
        images[0].tofile(f"{test_dir}/{epoch:04d}.floatarray")


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
    if os.path.isfile(config.output_dir + "/checkpoint"):
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
                noise_pred = model(noisy_images, timesteps, return_dict=False)[0]
                if (config.loss_function != None):
                    loss = config.loss_function(noise_pred, noise)
                else:
                    loss = F.mse_loss(noise_pred, noise)

                # loss = F.mse_loss(noise_pred, noise)
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
            pipeline = DDIMPipeline(unet=accelerator.unwrap_model(model), scheduler=noise_scheduler)

            if (epoch + 1) % config.save_image_epochs == 0 or epoch == config.num_epochs - 1:
                evaluate(config, epoch, pipeline)

            if (epoch + 1) % config.save_model_epochs == 0 or epoch == config.num_epochs - 1:
                if config.push_to_hub:
                    repo.push_to_hub(commit_message=f"Epoch {epoch}", blocking=True)
                else:
                    save_checkpoint(config.output_dir, epoch, model, noise_scheduler, optimizer, lr_scheduler)


