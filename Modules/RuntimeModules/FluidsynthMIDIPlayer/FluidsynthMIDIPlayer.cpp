#include "FluidsynthMIDIPlayer.h"
#include <fluidsynth.h>

FluidsynthMIDIPlayer::FluidsynthMIDIPlayer()
{
    settings = new_fluid_settings();
    synth = new_fluid_synth(settings);

    // Create an audio driver
    adriver = new_fluid_audio_driver(settings, synth);

    // Increase the volume (gain) for the entire synth
    double volume = 0.1; // Adjust the volume level (1.0 is the default, higher values increase the volume)
    fluid_settings_setnum(settings, "synth.gain", volume);
}

int FluidsynthMIDIPlayer::LoadSoundfont(const char* sfPath)
{
    return fluid_synth_sfload(synth, sfPath, 1);
}

void FluidsynthMIDIPlayer::OnNoteOn(const NoteOn& noteOn)
{ 
    Super::OnNoteOn(noteOn);
    fluid_synth_noteon(synth, noteOn.channel, noteOn.key, noteOn.velocity);
}
void FluidsynthMIDIPlayer::OnNoteOff(const NoteOff& noteOff)
{ 
    Super::OnNoteOff(noteOff);
    fluid_synth_noteoff(synth, noteOff.channel, noteOff.key);
}
void FluidsynthMIDIPlayer::OnNoteOnOff(const NoteOnOff& noteOnOff)
{ 
    Super::OnNoteOnOff(noteOnOff);
    fluid_synth_noteon(synth, noteOnOff.channel, noteOnOff.key, noteOnOff.velocity);
    // TODO : noteoff
}

void FluidsynthMIDIPlayer::OnProgramChange(const ProgramChange& programChange) 
{
    Super::OnProgramChange(programChange); 
    fluid_synth_program_change(synth, programChange.channel, programChange.newProgram);
}
void FluidsynthMIDIPlayer::OnControlChange(const ControlChange& controlChange) 
{
    Super::OnControlChange(controlChange); 
    fluid_synth_cc(synth, controlChange.channel, (int)controlChange.ctrl, controlChange.value);
}
void FluidsynthMIDIPlayer::OnPitchBend(const PitchBend& pitchBend) 
{
    Super::OnPitchBend(pitchBend); 
    fluid_synth_pitch_bend(synth, pitchBend.channel, pitchBend.value);
}