import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.signal import stft

# Load the audio file
filename = 'sunflower-street-drumloop-85bpm-163900mono.wav'
sr, y = wavfile.read(filename)

# Compute the STFT of the signal
f, t, Zxx = stft(y, fs=sr)

# Get the phase
phase = np.angle(Zxx)

# Display the phase spectrogram
plt.figure(figsize=(10, 6))
plt.pcolormesh(t, f, phase, shading='gouraud')
plt.title('Phase Spectrogram')
plt.ylabel('Frequency [Hz]')
plt.xlabel('Time [sec]')
plt.colorbar(label='Phase [radians]')
plt.tight_layout()
plt.show()