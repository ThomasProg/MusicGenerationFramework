
import miditoolkit
import numpy as np

class MIDIStructuredTokenizer:
    minPitch = 0
    maxPitch = 127

    nbPitchDeltaTokens = 39 # should be impair to be symmetric
    nbDurationTokens = 200
    nbTimeShiftTokens = 100

    addPitchTokens = False
    addPitchDeltaTokens = True
    addDurationTokens = True
    addTimeShiftTokens = True

    pitchDeltaStartPitch = 60

    def __init__(self):
        self._nbTokens = 0

        if (self.addPitchTokens):
            self._firstPitchToken = self._nbTokens
            self._nbTokens += self.maxPitch - self.minPitch

        if (self.addPitchDeltaTokens):
            assert(self.nbPitchDeltaTokens % 2 == 1)
            self._firstPitchDeltaToken = self._nbTokens
            self._nbTokens += self.nbPitchDeltaTokens

        if (self.addDurationTokens):
            self._firstDurationToken = self._nbTokens
            self._nbTokens += self.nbDurationTokens

        if (self.addTimeShiftTokens):
            self._firstTimeShiftToken = self._nbTokens
            self._nbTokens += self.nbTimeShiftTokens

        self.bos_token_id = self._nbTokens
        self.eos_token_id = self._nbTokens
        self._nbTokens += 1

        assert(self.addPitchTokens ^ self.addPitchDeltaTokens)

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
        assert(pitchDelta >= - self.nbPitchDeltaTokens//2 and pitchDelta < self.nbPitchDeltaTokens//2)
        return round(np.interp(pitchDelta, [- self.nbPitchDeltaTokens // 2, self.nbPitchDeltaTokens // 2], [self.firstPitchDeltaToken(), self.lastPitchDeltaToken()]))
    
    def tokenToPitchDelta(self, token):
        assert(self.isPitchDeltaToken(token))
        return round(np.interp(token, [self.firstPitchDeltaToken(), self.lastPitchDeltaToken()], [- self.nbPitchDeltaTokens // 2, self.nbPitchDeltaTokens // 2]))
    
    def isPitchDeltaToken(self, token):
        return token >= self.firstPitchDeltaToken() and token <= self.lastPitchDeltaToken()

    ## ========= DURATION ========= ## 

    def firstDurationToken(self):
        return self._firstDurationToken

    def lastDurationToken(self):
        return self.firstDurationToken() + self.nbDurationTokens - 1

    def durationToToken(self, duration):
        assert(duration >= 0 and duration < self.nbDurationTokens)
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

        tokens = []
        lastNote = None
        for note in notes:
            if (self.addPitchTokens):
                tokens.append(self.pitchToToken(note.pitch))

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

        return tokens
    
    def fileToTokens(self, filename):
        return self.midiFileToTokens(miditoolkit.MidiFile(filename))

    def encode(self, midiFile):
        return self.midiFileToTokens(midiFile)
    
    # Make sure notes are valid
    def decode(self, tokens):
        notes = []

        start = 0
        
        pitch = None
        pitchDelta = None
        duration = None
        timeShift = None

        def reset(self):
            nonlocal pitch, pitchDelta, duration, timeShift

            if self.addPitchTokens:
                pitch = None
            else:
                pitch = 60

            if self.addPitchDeltaTokens:
                pitchDelta = None
            else:
                pitchDelta = 0

            if self.addDurationTokens:
                duration = None
            else:
                duration = 1

            if self.addTimeShiftTokens:
                timeShift = None
            else:
                timeShift = 1

        reset(self)
        if (self.addPitchDeltaTokens):
            pitch = self.pitchDeltaStartPitch

        for token in tokens:
            if (self.addPitchTokens and self.isPitchToken(token)):
                pitch = self.tokenToPitch(token)
            elif (self.addPitchDeltaTokens and self.isPitchDeltaToken(token)):
                pitchDelta = self.tokenToPitchDelta(token)
            elif (self.addDurationTokens and self.isDurationToken(token)):
                duration = self.tokenToDuration(token)
            elif (self.addTimeShiftTokens and self.isTimeShiftToken(token)):
                timeShift = self.tokenToTimeShift(token)

            if (((self.addPitchTokens and pitch != None) or (self.addPitchDeltaTokens and pitchDelta != None)) and duration != None and timeShift != None):
                velocity = 100
                start += timeShift
                end = start + duration

                if (self.addPitchDeltaTokens):
                    pitch += pitchDelta

                notes.append(miditoolkit.Note(velocity, pitch, start, end))

                reset(self)

        midiFile = miditoolkit.MidiFile()
        midiFile.instruments.append(miditoolkit.Instrument(0, False, "Piano", notes))

        return midiFile
