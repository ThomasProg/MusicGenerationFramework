# @TODO
# only the unet is being loaded ; the scheduler should be loaded too
# the unet alone can't be used to generate an image

import onnxruntime

# Load the ONNX model
onnx_model_path = "Assets/Models/ddim-butterflies-16/model.onnx"
ort_session = onnxruntime.InferenceSession(onnx_model_path)

# Prepare input data (replace this with your actual input data)
import numpy as np

# input_data1 = np.random.randn(1, 3, 32, 32).astype(np.float32)
input_data1 = np.zeros([1, 3, 32, 32]).astype(np.float32)
input_data = [input_data1, np.array(1).astype(np.int64)]



# Get the names of all expected inputs
input_names = [input.name for input in ort_session.get_inputs()]
print(input_names)

# Create the input feed dictionary
input_feed = {}
for i in range(len(input_names)):
    input_feed[input_names[i]] = input_data[i] 
# {input_name: input_data for input_name in input_names}

# Run inference
output = ort_session.run(None, input_feed)
# output = ort_session.run(None, {"inputX": input_data})

# Output will be a list of output tensors, you can process them as needed
# print(output)

from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

# pil_image = Image.fromarray((output.numpy() * 255).astype(np.uint8))

# Scale the numpy array values to the range [0, 255]

output = output[0][0]
scaled_array = np.clip(output * 255, 0, 255).astype(np.uint8)

# Convert the numpy array to a PIL Image
pil_image = Image.fromarray(scaled_array.transpose(1, 2, 0))  # Assuming tensor is in (C, H, W) format


plt.imshow(pil_image)
plt.show()

# w, h = output.size
# grid = Image.new('RGB', size=(w, h))
# # for i, image in enumerate(images):
# grid.paste(image, box=(i%cols*w, i//cols*h))