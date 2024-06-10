
from pathlib import Path

from tokenizers import ByteLevelBPETokenizer

paths = [str(x) for x in Path("./eo_data/").glob("**/*.txt")]


import miditoolkit

class MIDITokenizer:
    def __init__(self):
        self.pitch_offset = 21  # MIDI note number for A0
        self.num_pitches = 88   # Number of piano keys (A0 to C8)

    def note_to_token(self, note):
        start = note.start
        pitch = note.pitch - self.pitch_offset
        velocity = note.velocity
        duration = note.end - note.start
        return [start, pitch, velocity, duration]

    def token_to_note(self, token):
        start = token[0]
        pitch = token[1] + self.pitch_offset
        velocity = token[2]
        duration = token[3]
        return miditoolkit.Note(start=start, end=start+duration, pitch=pitch, velocity=velocity)

    def encode(self, notes):
        tokens = []
        for note in notes:
            tokens.extend(self.note_to_token(note))
        return tokens

    def decode(self, tokens):
        notes = []
        for i in range(0, len(tokens), 4):
            note = self.token_to_note(tokens[i:i+4])
            notes.append(note)
        return notes


def midi_to_tokens(midi_path, tokenizer):
    midi_obj = miditoolkit.MidiFile(midi_path)
    notes = midi_obj.instruments[0].notes
    tokens = tokenizer.encode(notes)
    return tokens

# Example usage
midi_path = 'FurElise.mid'
tokenizer = MIDITokenizer()
tokens = midi_to_tokens(midi_path, tokenizer)
print(tokens)









# file = './esperanto/midi_sequences.txt'
# string_sequences = ''.join(map(chr, tokens))
string_sequences = ''.join(map(str, tokens))
# # Write the string sequences to a temporary file
# with open(file, 'w') as f:
#     f.write(string_sequences)


# Initialize a tokenizer
tokenizer = ByteLevelBPETokenizer()

# Customize training
tokenizer.train_from_iterator([string_sequences], vocab_size=52_000, min_frequency=2, special_tokens=[
    "<s>",
    "<pad>",
    "</s>",
    "<unk>",
    "<mask>",
])

# Save files to disk
import os
if not os.path.exists("./esperanto"):
    os.mkdir("./esperanto")
tokenizer.save_model("./esperanto", "esperberto")

