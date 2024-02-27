import tensorflow as tf
import matplotlib.pyplot as plt

# Load the saved model
# loaded_model = tf.saved_model.load('Assets/Models/KerasMnistModel')
# loaded_model = tf.keras.models.load_model('Model')

import numpy as np
from PIL import Image
def plot_latent_space(session, n=30, figsize=15):
    # display an n*n 2D manifold of digits
    digit_size = 28
    scale = 1.0
    figure = np.zeros((digit_size * n, digit_size * n))
    # linearly spaced coordinates corresponding to the 2D plot
    # of digit classes in the latent space
    grid_x = np.linspace(-scale, scale, n)
    grid_y = np.linspace(-scale, scale, n)[::-1]

    for i, yi in enumerate(grid_y):
        for j, xi in enumerate(grid_x):
            z_sample = np.array([[xi, yi]]).astype(np.float32)
            x_decoded = session.run(None, {'input_2': z_sample}) # ecoder.predict(z_sample)
            digit = x_decoded[0].reshape(digit_size, digit_size)
            figure[
                i * digit_size : (i + 1) * digit_size,
                j * digit_size : (j + 1) * digit_size,
            ] = digit

    plt.figure(figsize=(figsize, figsize))
    start_range = digit_size // 2
    end_range = n * digit_size + start_range
    pixel_range = np.arange(start_range, end_range, digit_size)
    sample_range_x = np.round(grid_x, 1)
    sample_range_y = np.round(grid_y, 1)
    plt.xticks(pixel_range, sample_range_x)
    plt.yticks(pixel_range, sample_range_y)
    plt.xlabel("z[0]")
    plt.ylabel("z[1]")
    plt.imshow(figure, cmap="Greys_r")
    plt.show()





# plot_latent_space(loaded_model)

import onnxruntime as rt

# Define the path to your ONNX model file
model_path = 'MyModel.onnx'

# Create an ONNX Runtime session
session = rt.InferenceSession(model_path)

# import numpy as np

# # Prepare input data (example, adjust as per your model's input requirements)
# input_data = np.random.randn(1, 2).astype(np.float32)  # Adjust shape and data type as needed

# # Run inference
# output = session.run(None, {'input_2': input_data})

plot_latent_space(session)