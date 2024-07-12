
import miditoolkit
import numpy as np

class MIDISequentialTokenizer:
    minPitch = 0 # 0
    maxPitch = 127 # 127

    addPitchTokens = True

    concatToken = 0
    skipToken = 0

    durationMult = 1#300 #1

    def __init__(self):
        self._nbTokens = 0

        if (self.addPitchTokens):
            self._firstPitchToken = self._nbTokens
            self._nbTokens += self.maxPitch - self.minPitch + 1

        self.concatToken = self._nbTokens
        self._nbTokens += 1

        self.skipToken = self._nbTokens
        self._nbTokens += 1

        self.bos_token_id = self._nbTokens
        self.eos_token_id = self._nbTokens
        self._nbTokens += 1

    @property
    def nbTokens(self):
        return self._nbTokens

    ## ========= PITCH ========= ## 

    def firstPitchToken(self):
        return self._firstPitchToken

    def lastPitchToken(self):
        return self.firstPitchToken() + self.maxPitch - self.minPitch

    def pitchToToken(self, pitch):
        assert(pitch >= self.minPitch and pitch <= self.maxPitch)
        return round(np.interp(pitch, [self.minPitch, self.maxPitch], [self.firstPitchToken(), self.lastPitchToken()]))
    
    def tokenToPitch(self, token):
        assert(self.isPitchToken(token))
        return round(np.interp(token, [self.firstPitchToken(), self.lastPitchToken()], [self.minPitch, self.maxPitch]))
    
    def isPitchToken(self, token):
        return token >= self.firstPitchToken() and token <= self.lastPitchToken()

    ## ========= ENCODE / DECODE ========= ## 

    def midiFileToTokens(self, midiFile: miditoolkit.MidiFile):
        notes = []
        for instr in midiFile.instruments:
            notes.extend(instr.notes) 
        # notes = midiFile.instruments[0].notes
        notes.sort(key=lambda x: x.start)
        midiFile.instruments[0].notes = notes

        tokens = [self.bos_token_id] # begin token
        lastNote = None
        for note in notes:
            note is miditoolkit.Note

            if (lastNote != None):
                delta = note.start - lastNote.start
            else:
                delta = 0

            # add concatenation token
            if (delta == 0):
                tokens.append(self.concatToken)

            for i in range(delta - 1):
            # if (delta > 1):
                tokens.append(self.skipToken)

            if (self.addPitchTokens):
                tokens.append(self.pitchToToken(note.pitch))

            lastNote = note


        tokens.append(self.eos_token_id) # end token
        return tokens


    def fileToTokens(self, filename):
        return self.midiFileToTokens(miditoolkit.MidiFile(filename))

    def encode(self, midiFile):
        return self.midiFileToTokens(midiFile)
    
    def __call__(self, midiFile):
        return self.midiFileToTokens(midiFile)
    
    # Make sure notes are valid
    # Events must be in chunks
    def decode(self, tokens):
        notes = []

        start = 0

        timeShift = 1
        for token in tokens:

            if (token == self.concatToken):
                timeShift = max(timeShift-1, 0)

            if (token == self.skipToken):
                timeShift += 1

            if (self.addPitchTokens and self.isPitchToken(token)):
                pitch = self.tokenToPitch(token)

                duration = self.durationMult * 500

                velocity = 100
                if (pitch < 50):
                    velocity = 50
                    
                start += timeShift * self.durationMult
                end = start + duration * self.durationMult

                notes.append(miditoolkit.Note(velocity, pitch, start, end))

                timeShift = 1

        midiFile = miditoolkit.MidiFile()
        midiFile.instruments.append(miditoolkit.Instrument(0, False, "Piano", notes))

        return midiFile

