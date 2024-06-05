from tld.configs import LTDConfig, DenoiserConfig, TrainConfig
from tld.diffusion import DiffusionTransformer

from PIL import Image

print("Start")

denoiser_cfg = DenoiserConfig(n_channels=4) #configure your model here.
cfg = LTDConfig(denoiser_cfg=denoiser_cfg)

diffusion_transformer = DiffusionTransformer(cfg)

print("Generate From Text")

out = diffusion_transformer.generate_image_from_text(prompt="a cute cat")

print("End Generation")

out.save("result.png")
