from diffusers import DiTPipeline, DPMSolverMultistepScheduler
import torch

pipe = DiTPipeline.from_pretrained("facebook/DiT-XL-2-256", torch_dtype=torch.float16)
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
# pipe = pipe.to("cuda")
pipe = pipe.to("cpu")

# pick words from Imagenet class labels
print(pipe.labels)  # to print all available words

# pick words that exist in ImageNet
words = ["white shark", "umbrella"]

class_ids = pipe.get_label_ids(words)

generator = torch.manual_seed(33)
output = pipe(class_labels=class_ids, num_inference_steps=3, generator=generator)

from matplotlib import pyplot as plt

image = output.images[0]  # label 'white shark'

plt.imshow(image)
plt.plot()

