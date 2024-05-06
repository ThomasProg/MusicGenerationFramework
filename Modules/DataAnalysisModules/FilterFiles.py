# The goal of the file is to filter the midi files with a particular time signature etc

from PyMIDIMusic import *
import os

class Test(IMIDIEventReceiver):
    nominator = 0
    denominator = 0
    clocks = 0
    notes = 0

    # def OnNoteOn(self, event): 
    #     e = NoteOn(event)
    #     self.times.append(e.GetDeltaTime())
    #     self.notes.append(e.GetKey())
    #     self.channels.append(e.GetChannel())

    def OnTimeSignature(self, event): 
        e = TimeSignature(event)
        self.nominator = e.GetNominator()
        self.denominator = e.GetDenominator()
        self.clocks = e.GetClocks()
        self.notes = e.GetNotes()

folder_path = 'Assets/Datasets/LakhMidi-Clean'

nbFound = 0
total = 0

import matplotlib.pyplot as plt

# Walk through the folder and its subfolders recursively
for root, dirs, files in os.walk(folder_path):
    for filename in files:
        if filename.endswith(".mid"):
            # Construct the full path to the file
            file_path = os.path.join(root, filename)

            music = MIDIMusic() 

            if (music.LoadFromFile(file_path)):
                # easyLib.MIDIMusic_ConvertAbsolute(music.nativeObject)
                test = Test()
                Dispatch(music, test)

                # the denominator doesn't matter much since it can be compressed or uncompressed
                if (test.nominator == 4):
                    nbFound += 1

                    music.Play("Assets/Soundfonts/Touhou/Touhou.sf2")

                    plt.figure(figsize=(10, 5))  # Set the figure size (optional)
                    plt.title(filename)
                    plt.show()

                    music.Stop()

                    # while(True):
                    #     pass




                    # print("clocks:",test.clocks)
                    # print("notes:",test.notes)

                # print(filename)
            # else:
            #     print(f"Error processing file {filename}")

            total += 1
            print(nbFound, " / ", total)


