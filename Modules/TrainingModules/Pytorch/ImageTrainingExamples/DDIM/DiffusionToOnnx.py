# @TODO
# only the unet is being exported ; the scheduler should be exported too
# the unet alone can't be used to generate an image

import torch

from diffusers import DDIMPipeline
from diffusers import UNet2DModel

from dataclasses import dataclass

import os
# Set TORCH_LOGS and TORCHDYNAMO_VERBOSE environment variables
# os.environ['TORCH_LOGS'] = '+dynamo'
# os.environ['TORCHDYNAMO_VERBOSE'] = '1'


@dataclass
class TrainingConfig:
    image_size = 32 #128  # the generated image resolution
    train_batch_size = 16
    eval_batch_size = 16  # how many images to sample during evaluation
    num_epochs = 50
    gradient_accumulation_steps = 1
    learning_rate = 1e-4
    lr_warmup_steps = 1000
    save_image_epochs = 1
    save_model_epochs = 2
    mixed_precision = 'fp16'  # `no` for float32, `fp16` for automatic mixed precision
    output_dir = 'Assets/Models/ddim-butterflies-16'  # the model namy locally and on the HF Hub

    push_to_hub = False  # whether to upload the saved model to the HF Hub
    hub_private_repo = False  
    overwrite_output_dir = True  # overwrite the old model when re-running the notebook
    seed = 0
    num_train_timesteps=500

config = TrainingConfig()

# Load the pipeline
pipeline = DDIMPipeline.from_pretrained(config.output_dir)
model = pipeline.unet

# # Load epoch information from configuration
# config_path = os.path.join(config.output_dir, "config.json")
# with open(config_path, "r") as f:
#     customConfig = json.load(f)
# start_epoch = customConfig.get("epoch", None) + 1

# # Load the optimizer state dictionary
# optimizer_state_dict = torch.load(os.path.join(config.output_dir, "optimizer_state_dict.pt"))

# # Assuming optimizer is already defined
# optimizer.load_state_dict(optimizer_state_dict)

# optimizer.zero_grad()

# # Load the optimizer state dictionary
# lr_scheduler_state_dict = torch.load(os.path.join(config.output_dir, "lr_scheduler_state_dict.pt"))

# # Assuming lr_scheduler is already defined
# lr_scheduler.load_state_dict(lr_scheduler_state_dict)



# Set the model to evaluation mode
model.eval()

# Example input tensor (you need to adjust this according to your model's input shape)
example_input = torch.randn(1, 3, config.image_size, config.image_size)

# Export the model to ONNX format
torch.onnx.export(model,                  # PyTorch model to be converted
                  (example_input, 10),         # Example input tensor
                  "Assets/Models/ddim-butterflies-16/model.onnx",       # Output ONNX file path
                  export_params=True,    # Export the trained parameters
                  opset_version=18,      # ONNX opset version
                  do_constant_folding=True,  # Optimize constant folding
                  input_names=["inputX"], # Name of the input tensor
                  output_names=["output"] # Name of the output tensor
                  )


# def forward(
#     self,
#     sample: torch.FloatTensor,
#     timestep: Union[torch.Tensor, float, int],
#     class_labels: Optional[torch.Tensor] = None,
#     return_dict: bool = True,
# ) -> Union[UNet2DOutput, Tuple]:

from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
output = model.forward(example_input, 10)

output = output.sample.detach().numpy()

output = output[0]
scaled_array = np.clip(output * 255, 0, 255).astype(np.uint8)

# Convert the numpy array to a PIL Image
pil_image = Image.fromarray(scaled_array.transpose(1, 2, 0))  # Assuming tensor is in (C, H, W) format


plt.imshow(pil_image)
plt.show()

# onnx_program = torch.onnx.dynamo_export(model, example_input, 10)