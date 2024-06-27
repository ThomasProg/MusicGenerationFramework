


midi_path = "Assets/Datasets/LakhMidiClean/Ludwig_van_Beethoven/5th_Symphony.mid"

import miditoolkit
midiFile = miditoolkit.MidiFile(midi_path)

for note in midiFile.instruments[0].notes:
    note.start //= 10
    note.end //= 10

midiFile.ticks_per_beat //= 10

midiFile.dump("midiCompressionTest.mid")





