from PyMIDIMusic import *
import matplotlib.pyplot as plt
import json

f = open('settings.json', 'r')
json_dict = json.load(f)

def GetTokens():
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

    music.LoadFromFile(json_dict["defaultMIDI"])

    easyLib.MIDIMusic_FilterChannel(music.nativeObject, 9, True)
    easyLib.MIDIMusic_Compress(music.nativeObject, 4*4)
    easyLib.MIDIMusic_ConvertToMonoTrack(music.nativeObject)

    easyLib.MIDIMusic_ConvertToNoteOnOff(music.nativeObject)

    easyLib.MIDIMusic_FilterInstruments(music.nativeObject, 0, 7, False)

    tokenizer = Tokenizer(midiMusic=music.nativeObject)
    tokenizer.BuildTokensFromNotes1()

    # statMusic = music.Clone()

    # while(True):
    #     pass

    test = Test()
    Dispatch(music, test)

    # while(True):
    #     pass

    return music, test, tokenizer.GetTokens()


import matplotlib.pyplot as plt
import numpy as np

def DisplayMusicRhythm(grid):
    fig1, (ax1, ax2)= plt.subplots(2, sharex = True, sharey = False)
    ax1.imshow(grid, interpolation ='none', aspect = 'auto')
    ax2.imshow(grid, interpolation ='bicubic', aspect = 'auto')

    # for (j,i),label in np.ndenumerate(grid):
    #     ax1.text(i,j,label,ha='center',va='center')

    plt.show()

    # print(len(test.times))

    # total = 0
    # for l in test.notesPerTiming:
    #     total += len(l)

    # print(total)


if __name__ == '__main__':
    music, test, tokens = GetTokens()

    # music.Play(json_dict["defaultSoundfont"])

    print("Min : ", test.minPitch)
    print("Max : ", test.maxPitch)

    # grid = [[]*len(test.notes)] * 1
    # grid[0] = tokens

    # grid[0] = np.array(test.notes) / 127
    # grid[1] = np.array(test.times) / 200
    

    # grid = [tokens]
    # print(len(tokens))
    tokens.pop()
    grid = np.array(tokens).reshape((252, 16))
    # grid = np.array(tokens).reshape((252, 16)).transpose(1,0)
    

    # for v in grid[0]:
        # print(v)

    # print(test.times)
    print(test.notesPerTiming)
    print(len(test.notesPerTiming))
    # grid[0] = np.stack([np.array(test.notes)/128, np.array(test.notes)/128, np.array(test.times)/500], axis=-1)


    DisplayMusicRhythm(grid)


