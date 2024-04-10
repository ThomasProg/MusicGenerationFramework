from PyMIDIMusic import *
import numpy as np

array = np.fromfile("Assets/Models/ddim-music-16-remapping/samples/0026.floatarray", dtype=np.float32)

import matplotlib.pyplot as plt

# plt.imshow(img)
# plt.show()

# Get the width and height of the image
# width, height = img.size

width, height = 16, 16 # array.shape

# Load the image data
# pixels = img.load()
# data = np.array(img)
data = array

# Boolean mask for values greater than or equal to the threshold
# mask = data >= 0.4

# Apply the mask to the array
# data = data[mask]

# Get the minimum value in the array
min_value = np.min(data)

# Get the maximum value in the array
max_value = np.max(data)

print("Minimum value:", min_value)
print("Maximum value:", max_value)


music = MIDIMusic() 
music.SetTicksPerQuarterNote(16)
for line in range(height):
    for pixel in range(width):
        noteOnOff = NoteOnOff()
        noteOnOff.SetDuration(10)

        p = array[pixel + line * 16]
        p = np.interp(p, (0, 1), (40, 80))
        # note = int(float(p) / 255 * 127)
        note = int(p)
        print(note)
        if (note > 50):
            noteOnOff.SetKey(note)
            noteOnOff.SetVelocity(120)
        else:
            noteOnOff.SetKey(0)
            noteOnOff.SetVelocity(0)

        noteOnOff.SetChannel(0)
        noteOnOff.SetDeltaTime(1)
        music.AddEvent(noteOnOff)

print("GetTicksPerQuarterNote(): ", music.GetTicksPerQuarterNote())
music.Play("Assets/Soundfonts/Touhou/Touhou.sf2")


while(True):
    pass

