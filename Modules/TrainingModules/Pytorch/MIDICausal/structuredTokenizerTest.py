from structuredTokenizer import MIDIStructuredTokenizer

structuredTokenizer = MIDIStructuredTokenizer()

if structuredTokenizer.addPitchTokens:
    for i in range(0, 128):
        prediction = structuredTokenizer.tokenToPitch(structuredTokenizer.pitchToToken(i))
        assert(prediction == i)

if structuredTokenizer.addPitchAsChromaticScale:
    for i in range(0, 128):
        prediction = structuredTokenizer.tokenToPitchChroma(structuredTokenizer.pitchChromaToToken(i))
        # assert(prediction == i)
        # print(prediction)

if structuredTokenizer.addPitchOctave:
    for i in range(0, 128):
        prediction = structuredTokenizer.tokenToPitchOctave(structuredTokenizer.pitchOctaveToToken(i))
        # assert(prediction == i)
        # print(prediction)

if structuredTokenizer.addPitchAsChromaticScale and structuredTokenizer.addPitchOctave:
    for i in range(0, 128):
        chromaPrediction = structuredTokenizer.tokenToPitchChroma(structuredTokenizer.pitchChromaToToken(i))
        octavePrediction = structuredTokenizer.tokenToPitchOctave(structuredTokenizer.pitchOctaveToToken(i))
        prediction = structuredTokenizer.getPitchFromChromaAndOctave(chromaPrediction, octavePrediction)
        assert(prediction == i)

if structuredTokenizer.addDurationTokens:
    for i in range(0, 100):
        prediction = structuredTokenizer.tokenToDuration(structuredTokenizer.durationToToken(i))
        assert(prediction == i)

if structuredTokenizer.addTimeShiftTokens:
    for i in range(0, 100):
        prediction = structuredTokenizer.tokenToTimeShift(structuredTokenizer.timeShiftToToken(i))
        assert(prediction == i)

if structuredTokenizer.addPitchDeltaTokens:
    print("[%d, %d]" % (- (structuredTokenizer.nbPitchDeltaTokens//2), structuredTokenizer.nbPitchDeltaTokens//2))
    print("[%d, %d]" % (structuredTokenizer.firstPitchDeltaToken(), structuredTokenizer.lastPitchDeltaToken()))

    for i in range(-20, 20):
        token = structuredTokenizer.pitchDeltaToToken(i)
        prediction = structuredTokenizer.tokenToPitchDelta(token)
        assert(prediction == i)



















# midi_path = 'FurElise.mid'
midi_path = "Andre_Andante.mid"
# midi_path = "Assets/Datasets/LakhMidiClean/Ludwig_van_Beethoven/5th_Symphony.mid"

import miditoolkit
midiFile = miditoolkit.MidiFile(midi_path)
multSet = set()
for change in midiFile.time_signature_changes:
    multSet.add(change.denominator)
    multSet.add(change.numerator)
mult = 1
for m in multSet:
    mult *= m
# assert(len(midiFile.time_signature_changes) == 1)
# timeSign = midiFile.time_signature_changes[0]
    
print("original:")
newTicksPerBeat = mult # 4*4=16
for note in midiFile.instruments[0].notes:
    note.start = round(note.start * newTicksPerBeat / midiFile.ticks_per_beat)
    note.end = round(note.end * newTicksPerBeat / midiFile.ticks_per_beat)
    # print("%d / %d" % (note.start, note.end))
    print(note.pitch, end=" ")
print()

midiFile.ticks_per_beat = newTicksPerBeat

encodedMIDI = structuredTokenizer.encode(midiFile)
midiFile = structuredTokenizer.decode(encodedMIDI)

print("decoded:")
for note in midiFile.instruments[0].notes:
    print(note.pitch, end=" ")

# encodedMIDI = []
midiFile.ticks_per_beat = newTicksPerBeat
midiFile.dump("test.mid")

















