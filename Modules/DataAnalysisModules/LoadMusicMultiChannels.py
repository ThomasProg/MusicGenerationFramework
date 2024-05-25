from PyMIDIMusic import *
import numpy as np

# array = np.fromfile("Assets/Models/ddim-music-16-MaskPitchChannels/samples/0003.floatarray", dtype=np.float32)
# array = np.fromfile("Assets/Models/ddim-music-16-MaskPitchChannels/truth.floatarray", dtype=np.float32)
array = np.fromfile("Assets/Models/ddim-music-16-MultiChannels/samples/0002.floatarray", dtype=np.float32)
# array = np.fromfile("Assets/Models/ddim-music-16-MaskPitchChannelsOpti/samples/0002.floatarray", dtype=np.float32)

array = array.reshape((16, 16, 40))

import sys
np.set_printoptions(threshold=sys.maxsize, suppress=True)

height = array.shape[0]
width = array.shape[1]
nbPossibleNotes = array.shape[2]

music = MIDIMusic() 
music.SetTicksPerQuarterNote(16)
for line in range(height):
    for pixel in range(width):
        noteOnOffs = []

        for k in range(nbPossibleNotes):
            proba = array[line][pixel][k]
            note = k + 40
            # print(note)

            if proba > 0.9:
                noteOnOff = NoteOnOff()
                noteOnOff.SetDuration(10)

                noteOnOff.SetKey(note)
                noteOnOff.SetVelocity(120)

                noteOnOff.SetChannel(0)
                noteOnOffs.append(noteOnOff)

                # music.AddEvent(noteOnOff)

        if (len(noteOnOffs) == 0):
            noteOnOff = NoteOnOff()
            noteOnOff.SetDuration(10)

            noteOnOff.SetKey(0)
            noteOnOff.SetVelocity(0)

            noteOnOff.SetChannel(0)
            noteOnOffs.append(noteOnOff)

        noteOnOffs[0].SetDeltaTime(1)

        for noteOnOff in noteOnOffs:
            music.AddEvent(noteOnOff)
            print(noteOnOff.GetKey())

print("GetTicksPerQuarterNote(): ", music.GetTicksPerQuarterNote())
music.Play("Assets/Soundfonts/Touhou/Touhou.sf2")


while(True):
    pass

