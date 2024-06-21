
import miditoolkit
class MIDIStructuredTokenizer:
    addVelocity = False

    vocab = {


    }

    timeShiftToken = 0
    pitchToken = 0
    durationToken = 0

    def __init__(self):
        self.generateVocab()

    def generateVocab(self):
        self.timeShiftToken = 0
        self.vocab["TimeShift"] = self.timeShiftToken

        self.pitchToken = 1
        self.vocab["Pitch"] = self.pitchToken

        self.durationToken = 2
        self.vocab["Duration"] = self.durationToken

        for i in range(0, 127): 
            self.vocab[str(i)] = i


    # def fileToText(self, filename):
    #     midi_obj = miditoolkit.MidiFile(filename)
    #     notes = midi_obj.instruments[0].notes
    #     notes.sort(key=lambda x: x.start)

    #     text = ""
    #     for i, note in enumerate(notes):
    #         text += "Pitch:" + str(note.pitch) + "\n"

    #         if (self.addVelocity):
    #             text += "Velocity:" + str(note.velocity) + "\n"

    #         text += "Duration:" + str(note.end - note.start) + "\n"

    #         if (i == 0):
    #             text += "TimeShift:" + str(note.start) + "\n"
    #         else:
    #             text += "TimeShift:" + str(note.start - notes[i-1].start) + "\n"

    #     return text
    
    def midiFileToTokens(self, midiFile):
        notes = midiFile.instruments[0].notes
        notes.sort(key=lambda x: x.start)

        tokens = []
        for i, note in enumerate(notes):
            tokens.append(self.pitchToken)
            tokens.append(note.pitch)

            tokens.append(self.durationToken)
            tokens.append(note.duration)

            tokens.append(self.timeShiftToken)
            if (i == 0):
                tokens.append(note.start)
            else:
                tokens.append(note.start - notes[i-1].start)

        return tokens
    
    def fileToTokens(self, filename):
        return self.midiFileToTokens(miditoolkit.MidiFile(filename))

    def encode(self, midiFile):
        return self.midiFileToTokens(midiFile)

    def decode(self, tokens):
        notes = []

        start = 0
        for i in range(0, len(tokens), 6):
            pitchToken = tokens[i]
            pitch = tokens[i+1]

            durationToken = tokens[i+2]
            duration = tokens[i+3]

            timeShiftToken = tokens[i+4]
            timeShift = tokens[i+5]

            # notes.append((start, pitch, velocity, duration))

            if (pitchToken != self.pitchToken):
                print("invalid pitchToken")

            if (durationToken != self.durationToken):
                print("invalid pitchToken")

            if (timeShiftToken != self.timeShiftToken):
                print("invalid pitchToken")

            if (pitch > 127):
                pitch = 0

            velocity = 100
            start += timeShift
            end = start + duration
            notes.append(miditoolkit.Note(velocity, pitch, start, end))


        midiFile = miditoolkit.MidiFile()
        # midiFile.ticks_per_beat = 16
        midiFile.instruments.append(miditoolkit.Instrument(0, False, "Piano", notes))

        return midiFile
