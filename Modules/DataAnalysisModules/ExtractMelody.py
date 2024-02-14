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

    def OnNoteOnOff(self, event): 
        e = NoteOnOff(event)
        self.times.append(e.GetDeltaTime())
        self.notes.append(e.GetKey())
        self.channels.append(e.GetChannel())

music = MIDIMusic() 

f = open('settings.json', 'r')
json_dict = json.load(f)

music.LoadFromFile(json_dict["defaultMIDI"])


easyLib.MIDIMusic_FilterChannel(music.nativeObject, 9, True)
easyLib.MIDIMusic_ConvertToMonoTrack(music.nativeObject)

easyLib.MIDIMusic_ConvertToNoteOnOff(music.nativeObject)

# easyLib.MIDIMusic_ConvertAbsolute(music.nativeObject)
easyLib.MIDIMusic_FilterInstruments(music.nativeObject, 0, 7, False)

music.Play(json_dict["defaultSoundfont"])

statMusic = music.Clone()
easyLib.MIDIMusic_ConvertAbsolute(statMusic.nativeObject)

test = Test()
Dispatch(statMusic, test)

plt.figure(figsize=(10, 5))  # Set the figure size (optional)
plt.scatter(test.times, test.notes, marker='o')

plt.xlabel('Time')
plt.ylabel('Note')
plt.title("Channel per note")

plt.show()
# PlayMusic(music)

while (True):
    i = input()
    if (i == "q"):
        music.Stop()
        quit()




