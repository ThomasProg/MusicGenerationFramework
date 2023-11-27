#include "MIDIEvents.h"
#include "ConvertingParser.h"
#include "Player.h"
#include <fluidsynth.h>

void NoteOn::Play()
{
    fluid_synth_noteon(synth, channel, key, velocity);
}

void NoteOff::Play()
{
    fluid_synth_noteoff(synth, channel, key);
}

void ControlChange::Play()
{
    fluid_synth_cc(synth, channel, (int)ctrl, value);
}

void PitchBend::Play()
{
    fluid_synth_pitch_bend(synth, channel, value);
}

void ProgramChange::Play()
{
    fluid_synth_program_change(synth, channel, newProgram);
}