
import miditoolkit
import numpy as np

class MIDIStructuredTokenizer:
    minPitch = 0
    maxPitch = 127

    nbDurationTokens = 200
    nbTimeShiftTokens = 100

    addPitchTokens = True
    addDurationTokens = True
    addTimeShiftTokens = True

    def __init__(self):
        self.nbTokens = 0

        if (self.addPitchTokens):
            self._firstPitchToken = self.nbTokens
            self.nbTokens += self.maxPitch - self.minPitch

        if (self.addDurationTokens):
            self._firstDurationToken = self.nbTokens
            self.nbTokens += self.nbDurationTokens

        if (self.addTimeShiftTokens):
            self._firstTimeShiftToken = self.nbTokens
            self.nbTokens += self.nbTimeShiftTokens

    ## ========= PITCH ========= ## 

    def firstPitchToken(self):
        return self._firstPitchToken

    def lastPitchToken(self):
        return self.maxPitch - self.minPitch

    def pitchToToken(self, pitch):
        assert(pitch >= self.minPitch and pitch <= self.maxPitch)
        return round(np.interp(pitch, [self.minPitch, self.maxPitch], [self.firstPitchToken(), self.lastPitchToken()]))
    
    def tokenToPitch(self, token):
        assert(self.isPitchToken(token))
        return round(np.interp(token, [self.firstPitchToken(), self.lastPitchToken()], [self.minPitch, self.maxPitch]))
    
    def isPitchToken(self, token):
        return token >= self.firstPitchToken() and token <= self.lastPitchToken()

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
        lastNoteStart = 0
        for i, note in enumerate(notes):
            if (self.addPitchTokens):
                tokens.append(self.pitchToToken(note.pitch))

            if (self.addDurationTokens):
                tokens.append(self.durationToToken(note.duration))

            if (self.addTimeShiftTokens):
                delta = note.start - lastNoteStart
                tokens.append(self.timeShiftToToken(delta))
                lastNoteStart = note.start

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
        duration = None
        timeShift = None

        def reset(self):
            nonlocal pitch, duration, timeShift

            if self.addPitchTokens:
                pitch = None
            else:
                pitch = 60

            if self.addDurationTokens:
                duration = None
            else:
                duration = 1

            if self.addTimeShiftTokens:
                timeShift = None
            else:
                timeShift = 1

        reset(self)

        for token in tokens:
            if (self.addPitchTokens and self.isPitchToken(token)):
                pitch = self.tokenToPitch(token)
            elif (self.addDurationTokens and self.isDurationToken(token)):
                duration = self.tokenToDuration(token)
            elif (self.addTimeShiftTokens and self.isTimeShiftToken(token)):
                timeShift = self.tokenToTimeShift(token)

            if (pitch != None and duration != None and timeShift != None):
                velocity = 100
                start += timeShift
                end = start + duration
                notes.append(miditoolkit.Note(velocity, pitch, start, end))

                reset(self)

        midiFile = miditoolkit.MidiFile()
        midiFile.instruments.append(miditoolkit.Instrument(0, False, "Piano", notes))

        return midiFile

