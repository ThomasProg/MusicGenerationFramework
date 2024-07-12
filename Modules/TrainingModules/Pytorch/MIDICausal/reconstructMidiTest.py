# from structuredTokenizer import MIDIStructuredTokenizer
# tokenizer = MIDIStructuredTokenizer()

from sequentialTokenizer import MIDISequentialTokenizer
tokenizer = MIDISequentialTokenizer()

# midi_path = 'FurElise.mid'
midi_path = "Andre_Andante.mid"
# midi_path = "Assets/Datasets/LakhMidiClean/Ludwig_van_Beethoven/5th_Symphony.mid"

import miditoolkit
midiFile1 = miditoolkit.MidiFile(midi_path)
notes = []
for instr in midiFile1.instruments:
    notes.extend(instr.notes) 
# notes = midiFile1.instruments[0].notes
notes.sort(key=lambda x: x.start)
midiFile1.instruments[0].notes = notes

encodedMIDI = tokenizer.encode(midiFile1)
print("encoded:")
for note in midiFile1.instruments[0].notes:
    print(note.pitch, end=" ")
    # print(note.start, end=" ")


midiFile = tokenizer.decode(encodedMIDI)
midiFile.ticks_per_beat = midiFile1.ticks_per_beat

print()
print("decoded:")
for note in midiFile.instruments[0].notes:
    print(note.pitch, end=" ")
    # print(note.start, end=" ")

midiFile.dump("test.mid")
