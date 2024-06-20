import numpy as np
from scipy.io import wavfile
from scipy.signal import stft, istft
import matplotlib.pyplot as plt

# Load the audio file
filename = 'sunflower-street-drumloop-85bpm-163900mono.wav'
sr, y = wavfile.read(filename)

# Compute the STFT of the signal
f, t, Zxx = stft(y, fs=sr)

# Get the magnitude and phase
magnitude = np.abs(Zxx)
phase = np.angle(Zxx)

# Display the magnitude spectrogram
plt.figure(figsize=(10, 6))
plt.pcolormesh(t, f, 20 * np.log10(magnitude), shading='gouraud')
plt.title('Magnitude Spectrogram')
plt.ylabel('Frequency [Hz]')
plt.xlabel('Time [sec]')
plt.colorbar(format='%+2.0f dB')
plt.tight_layout()
plt.show()

# Combine magnitude and phase to form the complex STFT matrix
Zxx_reconstructed = magnitude * np.exp(1j * phase)

# Compute the inverse STFT to get the time-domain signal
_, y_reconstructed = istft(Zxx_reconstructed, fs=sr)

# Save the reconstructed audio to a file
wavfile.write('reconstructed_audio.wav', sr, y_reconstructed.astype(np.int16))