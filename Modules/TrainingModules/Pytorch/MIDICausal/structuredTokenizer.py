
import miditoolkit
import numpy as np

class MIDIStructuredTokenizer:
    minPitch = 0
    maxPitch = 127

    nbPitchDeltaTokens = 257 # should be impair to be symmetric
    nbDurationTokens = 200
    nbTimeShiftTokens = 100

    nbPitchChromaTokens = 12 # Constant for chroma
    nbPitchOctaveTokens = 11 # Constant for chroma

    addPitchTokens = True
    addPitchDeltaTokens = False
    addPitchAsChromaticScale = False
    addPitchOctave = False
    addDurationTokens = False
    addTimeShiftTokens = True

    pitchDeltaStartPitch = 76

    def __init__(self):
        self._nbTokens = 0

        if (self.addPitchTokens):
            assert(not(self.addPitchDeltaTokens))
            assert(not(self.addPitchAsChromaticScale))
            self._firstPitchToken = self._nbTokens
            self._nbTokens += self.maxPitch - self.minPitch

        if (self.addPitchDeltaTokens):
            assert(not(self.addPitchTokens))
            assert(not(self.addPitchAsChromaticScale))
            assert(self.nbPitchDeltaTokens % 2 == 1)
            self._firstPitchDeltaToken = self._nbTokens
            self._nbTokens += self.nbPitchDeltaTokens

        if (self.addPitchAsChromaticScale):
            assert(not(self.addPitchTokens))
            assert(not(self.addPitchDeltaTokens))
            self._firstPitchChromaToken = self._nbTokens
            self._nbTokens += self.nbPitchChromaTokens

        if (self.addPitchOctave):
            assert(not(self.addPitchTokens))
            assert(not(self.addPitchDeltaTokens))
            self._firstPitchOctaveToken = self._nbTokens
            self._nbTokens += self.nbPitchOctaveTokens

        if (self.addDurationTokens):
            self._firstDurationToken = self._nbTokens
            self._nbTokens += self.nbDurationTokens

        if (self.addTimeShiftTokens):
            self._firstTimeShiftToken = self._nbTokens
            self._nbTokens += self.nbTimeShiftTokens

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
    
    ## ========= PITCH DELTA ========= ## 

    def firstPitchDeltaToken(self):
        return self._firstPitchDeltaToken

    def lastPitchDeltaToken(self):
        return self.firstPitchDeltaToken() + self.nbPitchDeltaTokens - 1

    def pitchDeltaToToken(self, pitchDelta):
        assert(pitchDelta >= - (self.nbPitchDeltaTokens//2) and pitchDelta < self.nbPitchDeltaTokens//2)
        return round(np.interp(pitchDelta, [- (self.nbPitchDeltaTokens // 2), self.nbPitchDeltaTokens // 2], [self.firstPitchDeltaToken(), self.lastPitchDeltaToken()]))
    
    def tokenToPitchDelta(self, token):
        assert(self.isPitchDeltaToken(token))
        return round(np.interp(token, [self.firstPitchDeltaToken(), self.lastPitchDeltaToken()], [- (self.nbPitchDeltaTokens // 2), self.nbPitchDeltaTokens // 2]))
    
    def isPitchDeltaToken(self, token):
        return token >= self.firstPitchDeltaToken() and token <= self.lastPitchDeltaToken()
    
    ## ========= PITCH AS CHROMATIC SCALE ========= ## 

    def firstPitchChromaToken(self):
        return self._firstPitchChromaToken

    def lastPitchChromaToken(self):
        return self.firstPitchChromaToken() + self.nbPitchChromaTokens - 1

    def pitchChromaToToken(self, pitchChroma):
        pitchChroma = pitchChroma % self.nbPitchChromaTokens
        return round(np.interp(pitchChroma, [0, self.nbPitchChromaTokens - 1], [self.firstPitchChromaToken(), self.lastPitchChromaToken()]))
    
    def tokenToPitchChroma(self, token):
        assert(self.isPitchChromaToken(token))
        return round(np.interp(token, [self.firstPitchChromaToken(), self.lastPitchChromaToken()], [0, self.nbPitchChromaTokens - 1]))
    
    def isPitchChromaToken(self, token):
        return token >= self.firstPitchChromaToken() and token <= self.lastPitchChromaToken()
    
    ## ========= PITCH OCTAVE ========= ## 

    def firstPitchOctaveToken(self):
        return self._firstPitchOctaveToken

    def lastPitchOctaveToken(self):
        return self.firstPitchOctaveToken() + self.nbPitchOctaveTokens - 1

    def pitchOctaveToToken(self, pitchOctave):
        pitchOctave = pitchOctave // self.nbPitchChromaTokens
        return round(np.interp(pitchOctave, [0, self.nbPitchOctaveTokens - 1], [self.firstPitchOctaveToken(), self.lastPitchOctaveToken()]))
    
    def tokenToPitchOctave(self, token):
        assert(self.isPitchOctaveToken(token))
        return round(np.interp(token, [self.firstPitchOctaveToken(), self.lastPitchOctaveToken()], [0, self.nbPitchOctaveTokens - 1]))
    
    def isPitchOctaveToken(self, token):
        return token >= self.firstPitchOctaveToken() and token <= self.lastPitchOctaveToken()
    
    def getPitchFromChromaAndOctave(self, chroma, octave):
        return octave * 12 + chroma

    ## ========= DURATION ========= ## 

    def firstDurationToken(self):
        return self._firstDurationToken

    def lastDurationToken(self):
        return self.firstDurationToken() + self.nbDurationTokens - 1

    def durationToToken(self, duration):
        # assert(duration >= 0 and duration < self.nbDurationTokens)
        assert(duration >= 0)
        duration = min(duration, self.nbDurationTokens - 1)
        return round(np.interp(duration, [0, self.nbDurationTokens-1], [self.firstDurationToken(), self.lastDurationToken()]))
    
    def tokenToDuration(self, token):
        assert(self.isDurationToken(token))
        return round(np.interp(token, [self.firstDurationToken(), self.lastDurationToken()], [0, self.nbDurationTokens-1]))
    
    def isDurationToken(self, token):
        return token >= self.firstDurationToken() and token <= self.lastDurationToken()
    
    ## ========= TIME SHIFT ========= ## 

    def firstTimeShiftToken(self):
        return self._firstTimeShiftToken

    def lastTimeShiftToken(self):
        return self.firstTimeShiftToken() + self.nbTimeShiftTokens - 1
    
    def timeShiftToToken(self, timeShift):
        assert(timeShift >= 0)
        timeShift = min(timeShift, self.nbTimeShiftTokens-1)
        return round(np.interp(timeShift, [0, self.nbTimeShiftTokens-1], [self.firstTimeShiftToken(), self.lastTimeShiftToken()]))
    
    def tokenToTimeShift(self, token):
        assert(self.isTimeShiftToken(token))
        return round(np.interp(token, [self.firstTimeShiftToken(), self.lastTimeShiftToken()], [0, self.nbTimeShiftTokens-1]))
    
    def isTimeShiftToken(self, token):
        return token >= self.firstTimeShiftToken() and token <= self.lastTimeShiftToken()

    ## ========= ENCODE / DECODE ========= ## 

    def midiFileToTokens(self, midiFile):
        notes = midiFile.instruments[0].notes
        notes.sort(key=lambda x: x.start)

        tokens = [self.bos_token_id] # begin token
        lastNote = None
        for note in notes:
            if (self.addPitchTokens):
                tokens.append(self.pitchToToken(note.pitch))

            if (self.addPitchAsChromaticScale):
                tokens.append(self.pitchChromaToToken(note.pitch))
            if (self.addPitchOctave):
                tokens.append(self.pitchOctaveToToken(note.pitch))

            if (self.addPitchDeltaTokens):
                if (lastNote != None):
                    tokens.append(self.pitchDeltaToToken(note.pitch - lastNote.pitch))
                else:
                    tokens.append(self.pitchDeltaToToken(0))

            if (self.addDurationTokens):
                tokens.append(self.durationToToken(note.duration))

            if (self.addTimeShiftTokens):
                if (lastNote != None):
                    tokens.append(self.timeShiftToToken(note.start - lastNote.start))
                else:
                    tokens.append(self.timeShiftToToken(0))

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
        
        pitch = None
        pitchDelta = None
        duration = None
        timeShift = None
        pitchChroma = None
        pitchOctave = None

        def reset(self):
            nonlocal pitch, pitchDelta, duration, timeShift, pitchChroma, pitchOctave

            if self.addPitchTokens:
                pitch = None
            elif self.addPitchDeltaTokens:
                pitchDelta = None
            else:
                pitch = 60
                pitchDelta = 0

            if self.addDurationTokens:
                duration = None
            else:
                duration = 1

            if self.addTimeShiftTokens:
                timeShift = None
            else:
                timeShift = 1

            pitchChroma = None
            pitchOctave = None

        reset(self)
        if (self.addPitchDeltaTokens):
            pitch = self.pitchDeltaStartPitch

        for token in tokens:
            if (self.addPitchTokens and self.isPitchToken(token)):
                pitch = self.tokenToPitch(token)
            elif (self.addPitchDeltaTokens and self.isPitchDeltaToken(token)):
                pitchDelta = self.tokenToPitchDelta(token)
            elif (self.addPitchAsChromaticScale and self.isPitchChromaToken(token)):
                pitchChroma = self.tokenToPitchChroma(token)
            elif (self.addPitchOctave and self.isPitchOctaveToken(token)):
                pitchOctave = self.tokenToPitchOctave(token)
            elif (self.addDurationTokens and self.isDurationToken(token)):
                duration = self.tokenToDuration(token)
            elif (self.addTimeShiftTokens and self.isTimeShiftToken(token)):
                timeShift = self.tokenToTimeShift(token)

            pitchReady = self.addPitchTokens and pitch != None
            pitchDeltaReady = self.addPitchDeltaTokens and pitchDelta != None
            pitchChromaReady = self.addPitchAsChromaticScale and pitchChroma != None
            pitchOctaveReady = not(self.addPitchAsChromaticScale) or pitchOctave != None

            if ((pitchReady or pitchDeltaReady or (pitchChromaReady and pitchOctaveReady)) and duration != None and timeShift != None):
                velocity = 100
                start += timeShift
                end = start + duration

                if (self.addPitchDeltaTokens):
                    pitch += pitchDelta
                    pitch = max(min(127, pitch), 0)
                    # assert(pitch >= 0 and pitch <= 127)

                if (self.addPitchAsChromaticScale):
                    if not(self.addPitchAsChromaticScale):
                        pitchOctave = 5
                    pitch = pitchChroma + pitchOctave * self.nbPitchChromaTokens

                notes.append(miditoolkit.Note(velocity, pitch, start, end))

                reset(self)

        midiFile = miditoolkit.MidiFile()
        midiFile.instruments.append(miditoolkit.Instrument(0, False, "Piano", notes))

        return midiFile

