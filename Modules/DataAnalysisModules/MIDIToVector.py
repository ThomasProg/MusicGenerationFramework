import threading
from PyMIDIMusic import *
import matplotlib.pyplot as plt
import asyncio
import json

class Test(IMIDIEventReceiver):
    channels = []
    times = []
    notes = []
    colors = []

    notesPerTiming = []

    minPitch = 128
    maxPitch = 0

    def OnEvent(self, event):
        e = PMIDIEvent(event)
        if (e.GetDeltaTime() != 0):
            for i in range(e.GetDeltaTime()):
                self.notesPerTiming.append([])

    def OnNoteOnOff(self, event): 
        e = NoteOnOff(event)
        self.times.append(e.GetDeltaTime())
        self.notes.append(e.GetKey())
        self.channels.append(e.GetChannel())

        self.notesPerTiming[-1].append(e.GetKey())

        self.minPitch = min(self.minPitch, e.GetKey())
        self.maxPitch = max(self.maxPitch, e.GetKey())

music = MIDIMusic() 

f = open('settings.json', 'r')
json_dict = json.load(f)

music.LoadFromFile(json_dict["defaultMIDI"])


easyLib.MIDIMusic_FilterChannel(music.nativeObject, 9, True)
easyLib.MIDIMusic_Compress(music.nativeObject, 4*4)
easyLib.MIDIMusic_ConvertToMonoTrack(music.nativeObject)

easyLib.MIDIMusic_ConvertToNoteOnOff(music.nativeObject)

# easyLib.MIDIMusic_ConvertAbsolute(music.nativeObject)
easyLib.MIDIMusic_FilterInstruments(music.nativeObject, 0, 7, False)

tokenizer = Tokenizer(midiMusic=music.nativeObject)
tokenizer.BuildTokensFromNotes1()

music.Play(json_dict["defaultSoundfont"])

statMusic = music.Clone()
# easyLib.MIDIMusic_ConvertAbsolute(statMusic.nativeObject)

test = Test()
Dispatch(statMusic, test)

print("Min : ", test.minPitch)
print("Max : ", test.maxPitch)

# plt.figure(figsize=(10, 5))  # Set the figure size (optional)
# plt.scatter(test.times, test.notes, marker='o')

# plt.xlabel('Time')
# plt.ylabel('Note')
# plt.title("Channel per note")

# plt.show()
# # PlayMusic(music)

# while (True):
#     i = input()
#     if (i == "q"):
#         music.Stop()
#         quit()


import matplotlib.pyplot as plt
import numpy as np

grid = np.array([[1,8,13,29,17,26,10,4],[16,25,31,5,21,30,19,15]])
grid = [[]*len(test.notes)] * 1

# grid[0] = np.array(test.notes) / 127
# grid[1] = np.array(test.times) / 200
grid[0] = tokenizer.GetTokens()

# for v in grid[0]:
    # print(v)

# print(test.times)
print(test.notesPerTiming)
# grid[0] = np.stack([np.array(test.notes)/128, np.array(test.notes)/128, np.array(test.times)/500], axis=-1)



fig1, (ax1, ax2)= plt.subplots(2, sharex = True, sharey = False)
ax1.imshow(grid, interpolation ='none', aspect = 'auto')
ax2.imshow(grid, interpolation ='bicubic', aspect = 'auto')

# for (j,i),label in np.ndenumerate(grid):
#     ax1.text(i,j,label,ha='center',va='center')

plt.show()

print(len(test.times))

total = 0
for l in test.notesPerTiming:
    total += len(l)

print(total)


