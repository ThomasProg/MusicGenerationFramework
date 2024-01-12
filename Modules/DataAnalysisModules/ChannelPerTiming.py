from PyMIDIMusic import *
import matplotlib.pyplot as plt

class Test(IMIDIEventReceiver):
    channels = []
    times = []
    notes = []
    colors = []

    def OnNoteOn(self, event): 
        e = NoteOn(event)
        self.times.append(e.GetDeltaTime())
        self.notes.append(e.GetKey())
        self.channels.append(e.GetChannel())


music = MIDIMusic() 

music.LoadFromFile("C:/Users/thoma/PandorasBox/Projects/ModularMusicGenerationModules/Assets/Datasets/LakhMidi-Clean/Ludwig_van_Beethoven/Fur_Elise.1.mid")

easyLib.MIDIMusic_ConvertAbsolute(music.nativeObject)

test = Test()
Dispatch(music, test)

plt.figure(figsize=(10, 5))  # Set the figure size (optional)
plt.scatter(test.times, test.channels, marker='o')

plt.xlabel('Time')
plt.ylabel('Note')
plt.title("Channel per note")

plt.show()