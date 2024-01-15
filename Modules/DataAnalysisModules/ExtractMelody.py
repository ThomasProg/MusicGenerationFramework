import threading
from PyMIDIMusic import *
import matplotlib.pyplot as plt
import asyncio

class Test(IMIDIEventReceiver):
    channels = []
    times = []
    notes = []
    colors = []

    def OnNoteOnOff(self, event): 
        e = NoteOnOff(event)
        self.times.append(e.GetDeltaTime())
        self.notes.append(e.GetKey())
        self.channels.append(e.GetChannel())

music = MIDIMusic() 

music.LoadFromFile("C:/Users/thoma/PandorasBox/Projects/ModularMusicGenerationModules/Assets/Datasets/LakhMidi-Clean/Ludwig_van_Beethoven/Fur_Elise.1.mid")


easyLib.MIDIMusic_FilterChannel(music.nativeObject, 9, True)
easyLib.MIDIMusic_ConvertToMonoTrack(music.nativeObject)

easyLib.MIDIMusic_ConvertToNoteOnOff(music.nativeObject)

# easyLib.MIDIMusic_ConvertAbsolute(music.nativeObject)
easyLib.MIDIMusic_FilterInstruments(music.nativeObject, 0, 7, False)

music.Play("C:/Users/thoma/PandorasBox/Projects/ModularMusicGenerationModules/Assets/Soundfonts/Touhou/Touhou.sf2")
# easyLib.MIDIMusic_ConvertAbsolute(music.nativeObject)

test = Test()
Dispatch(music, test)

plt.figure(figsize=(10, 5))  # Set the figure size (optional)
plt.scatter(test.times, test.notes, marker='o')

plt.xlabel('Time')
plt.ylabel('Note')
plt.title("Channel per note")

plt.show()

# easyLib.MIDIMusic_ConvertRelative(music.nativeObject)

# PlayMusic(music)

while (True):
    i = input()
    if (i == "q"):
        music.Stop()
        quit()




