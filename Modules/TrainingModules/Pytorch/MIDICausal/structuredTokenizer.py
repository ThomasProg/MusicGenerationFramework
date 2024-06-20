
import miditoolkit
class MIDIStructuredTokenizer:
    addVelocity = False

    def __init__(self):
        pass

    def fileToText(self, filename):
        midi_obj = miditoolkit.MidiFile(filename)
        notes = midi_obj.instruments[0].notes
        # tokens = []
        # for note in notes:
        #     tokens.append((note.start, note.pitch, note.velocity, note.end - note.start))
        # tokens.sort(key=lambda x: x[0])  # Ensure the tokens are in order of time
        notes.sort(key=lambda x: x.start)

        text = ""
        for i, note in enumerate(notes):
            text += "Pitch:" + str(note.pitch) + "\n"

            if (self.addVelocity):
                text += "Velocity:" + str(note.velocity) + "\n"

            text += "Duration:" + str(note.end - note.start) + "\n"

            if (i == 0):
                text += "TimeShift:" + str(note.start) + "\n"
            else:
                text += "TimeShift:" + str(note.start - notes[i-1].start) + "\n"

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
