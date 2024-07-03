
import miditoolkit
class MIDIStructuredTokenizer:
    addVelocity = False
    def __init__(self):
        pass

    def midiFileToTokens(self, midiFile):
        notes = midiFile.instruments[0].notes
        notes.sort(key=lambda x: x.start)

        tokens = []
        lastNoteStart = 0
        for i, note in enumerate(notes):
            assert(note.pitch >= 0 and note.pitch <= 127)
            tokens.append(note.pitch) # 0 - 127

            assert(note.duration >= 0 and note.duration <= 299 - 128)
            tokens.append(note.duration+128) # 128 - 299

            delta = note.start - lastNoteStart
            delta = min(delta, 999 - 300)
            assert(delta >= 0 and delta <= 999 - 300)
            if (i == 0):
                tokens.append(300) # remove delay at start
                # tokens.append(delta+300) # 300 - 999
            else:
                tokens.append(delta + 300) # 300 - 999
                lastNoteStart = note.start

        return tokens
    
    def fileToTokens(self, filename):
        return self.midiFileToTokens(miditoolkit.MidiFile(filename))

    def encode(self, midiFile):
        return self.midiFileToTokens(midiFile)
    
    def decode(self, tokens):
        notes = []

        start = 0
        
        pitch = None
        duration = None
        timeShift = None
        for token in tokens:
            if (token >= 0 and token <= 127):
                pitch = token
            elif (token >= 128 and token <= 299):
                duration = token - 128
            elif (token >= 300 and token <= 999):
                timeShift = token - 300

            if (pitch != None and duration != None and timeShift != None):
                velocity = 100
                start += timeShift
                # print(timeShift)
                end = start + duration
                notes.append(miditoolkit.Note(velocity, pitch, start, end))

                pitch = None
                duration = None
                timeShift = None

        midiFile = miditoolkit.MidiFile()
        midiFile.instruments.append(miditoolkit.Instrument(0, False, "Piano", notes))

        return midiFile

    # Make sure notes are valid
    def decode2(self, tokens):
        notes = []

        start = 0
        for i in range(0, len(tokens)-2, 3):
            pitch = tokens[i]

            if (pitch > 127 or pitch <= 0):
                i -= 2
                print("Invalid pitch: %s ; skipping" % pitch)
                continue

            duration = tokens[i+1] - 128
            if (duration < 0):
                print("duration : ", duration)
                continue

            timeShift = tokens[i+2] - 300
            if (timeShift < 0):
                print("timeShift : ", timeShift)
                continue

            velocity = 100
            start += timeShift
            end = start + duration
            notes.append(miditoolkit.Note(velocity, pitch, start, end))


        midiFile = miditoolkit.MidiFile()
        midiFile.instruments.append(miditoolkit.Instrument(0, False, "Piano", notes))

        return midiFile
