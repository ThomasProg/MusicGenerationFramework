



import os
os.environ["HF_DATASETS_OFFLINE"] = "1"

from datasets import load_dataset
dataset = load_dataset("nelorth/oxford-flowers")

test = dataset["test"]
train = dataset["train"]



import torch
import torchvision.transforms as transforms
from diffusion import DiffusionTransformer
from configs import LTDConfig

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
to_pil = transforms.ToPILImage()

cfg = LTDConfig()
diffusion_transformer = DiffusionTransformer(cfg)















# Forward Pass

from pydantic import BaseModel
class ImageRequest(BaseModel):
    prompt: str
    class_guidance: int = 6
    seed: int = 11
    num_imgs: int = 1
    img_size: int = 32


request = ImageRequest(prompt="A beautiful flower")

import io
img = diffusion_transformer.generate_image_from_text(
    prompt=request.prompt,
    class_guidance=request.class_guidance,
    seed=request.seed,
    num_imgs=request.num_imgs,
    img_size=request.img_size,
)
# # Convert PIL image to byte stream suitable for HTTP response
# img_byte_arr = io.BytesIO()
# img.save(img_byte_arr, format="PNG")

img_path = "generated_image.png"
img.save(img_path, format="PNG")



# print(dataset["test"][93]['label'])
