import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt

# Example of loading a real audio file and generating a magnitude spectrogram
filename = 'sunflower-street-drumloop-85bpm-163900mono.wav'
y, sr = librosa.load(filename, sr=None)
D = librosa.stft(y)
magnitude = np.abs(D)

# Assume `generated_magnitude` is obtained from your diffusion model
generated_magnitude = magnitude  # Replace with your model's output

# Apply the Griffin-Lim algorithm to estimate the phase
reconstructed_signal = librosa.griffinlim(generated_magnitude, n_iter=32)

# Save the reconstructed audio
librosa.output.write_wav('generated_audio.wav', reconstructed_signal, sr)