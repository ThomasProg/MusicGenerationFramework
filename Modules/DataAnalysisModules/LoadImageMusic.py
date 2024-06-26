from PIL import Image
from PyMIDIMusic import *
import numpy as np

# img = Image.open("Assets/Models/ddim-music-16/samples/0004.png")
# img = Image.open("Assets/Models/ddim-music-16/samples/0008.png")
# img = Image.open("Assets/Models/ddim-music-16-remapping/samples/0009.png")
img = Image.open("Assets/Models/ddim-music-16/datasetInput/0000.png")
# img = Image.open("Assets/Models/ddim-music-16/inputMusic/0000.png")

# img = Image.open("Assets/Models/ddim-music-full/truth/0000.png")

# img = img.convert("L")

import matplotlib.pyplot as plt

plt.imshow(img)
plt.show()

# Get the width and height of the image
width, height = img.size

# Load the image data
pixels = img.load()
data = np.array(img)


# music = MIDIMusic()
# noteOnOff = NoteOnOff()
# noteOnOff.SetKey(60)
# noteOnOff.SetVelocity(50)
# noteOnOff.SetDuration(20)

# music.AddEvent(noteOnOff)

# print("shape : ", data.shape)
# print(data)

# Boolean mask for values greater than or equal to the threshold
mask = data >= 40

# Apply the mask to the array
data = data[mask]

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

        p = pixels[pixel, line]
        p = np.interp(p, (0, 255), (40, 90))
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





# from PIL import Image

# # Open the image
# image = Image.open("Assets/Models/ddim-music/truth/0000.png")

# # Get the width and height of the image
# width, height = image.size

# # Convert the image to grayscale
# image = image.convert("L")

# # Load the image data
# pixels = image.load()

# # Loop over each pixel
# for y in range(height):
#     for x in range(width):
#         # Get the pixel value at (x, y)
#         pixel_value = pixels[x, y]
        
#         # Now you can do whatever you want with the pixel value
#         # For example, print it
#         print("Pixel at ({}, {}): {}".format(x, y, pixel_value))