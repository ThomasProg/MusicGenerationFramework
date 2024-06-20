
import miditoolkit
class MIDITokenizer:
    pad_token = 0
    pad_token_id = 0
    eos_token = 0

    def __init__(self):
        self.pitch_offset = 21  # MIDI note number for A0
        self.num_pitches = 88   # Number of piano keys (A0 to C8)
        self.model_input_names = ['input_ids']

    def fileToText(self, filename):
        midi_obj = miditoolkit.MidiFile(filename)
        notes = midi_obj.instruments[0].notes
        # tokens = []
        # for note in notes:
        #     tokens.append((note.start, note.pitch, note.velocity, note.end - note.start))
        # tokens.sort(key=lambda x: x[0])  # Ensure the tokens are in order of time
        notes.sort(key=lambda x: x.start)

        text = ""
        for note in notes:
            # start, pitch, velocity, duration = note
            # tokens.append(start)
            # tokens.append(pitch - self.pitch_offset)
            # tokens.append(velocity)
            # tokens.append(duration)
            text += "Pitch" + str(note.pitch) + "\n"
            text += note.start

        return text

    def encode(self, filename):
        midi_obj = miditoolkit.MidiFile(filename)
        notes = midi_obj.instruments[0].notes
        # tokens = []
        # for note in notes:
        #     tokens.append((note.start, note.pitch, note.velocity, note.end - note.start))
        # tokens.sort(key=lambda x: x[0])  # Ensure the tokens are in order of time
        notes.sort(key=lambda x: x.start)

        tokens = []
        for note in notes:
            # start, pitch, velocity, duration = note
            # tokens.append(start)
            # tokens.append(pitch - self.pitch_offset)
            # tokens.append(velocity)
            # tokens.append(duration)
            tokens.append(note.pitch)

        # mask = [1] * len(tokens)

        return tokens #[tokens, mask]

    def decode(self, tokens):
        notes = []
        for i in range(0, len(tokens), 4):
            start = tokens[i]
            pitch = tokens[i+1] + self.pitch_offset
            velocity = tokens[i+2]
            duration = tokens[i+3]
            notes.append((start, pitch, velocity, duration))
        return notes
