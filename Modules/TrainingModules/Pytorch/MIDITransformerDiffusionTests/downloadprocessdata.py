from PIL import Image
import torch
from torchvision.transforms import Resize, ToTensor
from datasets import load_dataset
dataset_name = 'pokemon-blip-captions-en-zh'

dataset = load_dataset(dataset_name)

def transform(example):
    example['image'] = example['image'].resize((256, 256), Image.BICUBIC)

    return example

dataset["train"] = dataset["train"].map(transform)




def custom_collate_fn(batch):
    images, texts = [], []
    for item in batch:
        image, text = item['image'], item['en_text']

        # Apply the transformation
        image = ToTensor()(image)

        images.append(image)
        texts.append(text)


    return torch.stack(images), texts

#dataset.set_format(type='torch', columns=['image', 'text'])
train_dataloader = torch.utils.data.DataLoader(dataset["train"], batch_size=32, shuffle=False, collate_fn=custom_collate_fn)




from tld.data import get_text_and_latent_embeddings_hdf5
import clip
from diffusers import AutoencoderKL
import os
import numpy as np


latent_save_path = 'pokemon_path'

model, preprocess = clip.load("ViT-L/14")

vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16)
vae = vae.to('cuda')
model.to('cuda')

if not os.path.exists(latent_save_path):
    os.mkdir(latent_save_path)

get_text_and_latent_embeddings_hdf5(train_dataloader, vae, model, drive_save_path=latent_save_path)

# img_encodings, text_encodings, text_captions = get_text_and_latent_embeddings_hdf5(train_dataloader, vae, model, drive_save_path=latent_save_path)

# np.save(os.path.join(latent_save_path, 'image_latents.npy'), img_encodings.numpy())
# np.save(os.path.join(latent_save_path, 'text_encodings.npy'), text_encodings.numpy())
