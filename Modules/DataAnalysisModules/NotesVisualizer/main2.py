# from PyEasyMidiFileParserCpp import *

# import sys
# print(sys.path)
# from MIDIMusic.py import *
from PyEasyMidiFileParserCppX import *
import matplotlib.pyplot as plt

class Test(IMIDIEventReceiver):
    times = []
    notes = []
    colors = []
    time = 0

    def OnEvent(self, event): 
        self.time += PMIDIEvent(event).GetDeltaTime()

    def OnSysEvent(self, event): pass
    def OnMetaEvent(self, event): pass
    def OnChannelEvent(self, event): pass

    def OnNoteOn(self, event): 
        e = NoteOn(event)
        print(str(e.GetKey()))
        print(str(e.GetDeltaTime()))
        self.times.append(self.time)
        self.notes.append(e.GetKey())
    def OnNoteOff(self, event): pass
    def OnNoteOnOff(self, event): pass


music = MIDIMusic() 

music.LoadFromFile("C:/Users/thoma/PandorasBox/Projects/ModularMusicGenerationModules/Assets/Datasets/LakhMidi-Clean/Ludwig_van_Beethoven/Fur_Elise.1.mid")

test = Test()
Dispatch(music, test)

plt.figure(figsize=(10, 5))  # Set the figure size (optional)
# plt.plot(times, notes, marker='o')  # 'o' adds points at data
# plt.scatter(times, notes, marker='o', color = colors)
plt.scatter(test.times, test.notes, marker='o')

plt.xlabel('Time')
plt.ylabel('Note')
# plt.title(filename + " / Track : " + str(trackIndex))
plt.title("Title")

plt.show()