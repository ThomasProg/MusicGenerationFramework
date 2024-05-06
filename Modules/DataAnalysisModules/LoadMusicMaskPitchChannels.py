from PyMIDIMusic import *
import numpy as np

# array = np.fromfile("Assets/Models/ddim-music-16-MaskPitchChannels/samples/0003.floatarray", dtype=np.float32)
# array = np.fromfile("Assets/Models/ddim-music-16-MaskPitchChannels/truth.floatarray", dtype=np.float32)
array = np.fromfile("Assets/Models/ddim-music-16-MaskPitchChannels/maskAndPitch.floatarray", dtype=np.float32)
array = array.reshape((16, 16, 2))
# print(array)
# print("f : ", np.interp(array, (0.0, 1.0), (40.0, 80.0)))
# array = np.fromfile("Assets/Models/ddim-music-16-MaskPitchChannels/tensor.floatarray", dtype=np.float32)

import matplotlib.pyplot as plt

# Generate some sample data (replace this with your own NumPy array)
x = np.linspace(0, 256, 256)  # Sample x values

# Plot the data
plt.plot(x, array.flatten()[1::2])
plt.title('Plot of a Pitches Array')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.grid(True)
plt.show(block=False)

plt.plot(x, array.flatten()[::2])
plt.title('Plot of a Mask Array')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.grid(True)
plt.show(block=False)




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

# print("Minimum value:", min_value)
# print("Maximum value:", max_value)


music = MIDIMusic() 
music.SetTicksPerQuarterNote(16)
for line in range(height):
    for pixel in range(width):
        noteOnOff = NoteOnOff()
        noteOnOff.SetDuration(10)

        p = array[line][pixel]
        p = np.interp(p, (0, 1), (40, 80))
        # note = int(float(p) / 255 * 127)

        # note = int(p)
        print(p)
        maskValue = int(p[0])
        note = int(p[1])

        # print(note)
        if (maskValue > 60):
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


plt.show()

while(True):
    pass

