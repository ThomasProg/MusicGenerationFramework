from diffusers import DDIMPipeline
import PIL.Image
import numpy as np

import time

# load model and scheduler
pipe = DDIMPipeline.from_pretrained("Assets/Models/ddim-butterflies-16")

genStartTime = time.time()

# run pipeline in inference (sample random noise and denoise)
image = pipe(eta=0.0, num_inference_steps=50).images[0]

print("--- %s seconds ---" % (time.time() - genStartTime))

# # process image to PIL
# image_processed = np.array(image).permute(0, 2, 3, 1)
# image_processed = (image_processed + 1.0) * 127.5
# image_processed = image_processed.numpy().astype(np.uint8)
# image_pil = PIL.Image.fromarray(image_processed[0])

# save image
image.save("aaaa.png")