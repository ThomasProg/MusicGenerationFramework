import matplotlib.pyplot as plt
from scipy import signal
from scipy.io import wavfile
import numpy as np

sample_rate, samples = wavfile.read('sunflower-street-drumloop-85bpm-163900mono.wav')
frequencies, times, spectrogram = signal.spectrogram(samples, sample_rate)

plt.pcolormesh(times, frequencies, 10*np.log10(spectrogram))
# plt.axis((None, None, 0, 200))
plt.ylabel('Frequency [Hz]')
plt.xlabel('Time [sec]')
plt.show()

# # for data transformation
# import numpy as np
# # for visualizing the data
# import matplotlib.pyplot as plt
# # for opening the media file
# import scipy.io.wavfile as wavfile

# Fs, aud = wavfile.read('sunflower-street-drumloop-85bpm-163900mono.wav')
# # select left channel only
# aud = aud[:]
# # trim the first 125 seconds
# first = aud[:int(Fs*125)]

# powerSpectrum, frequenciesFound, time, imageAxis = plt.specgram(first, Fs=Fs)
# plt.show()




