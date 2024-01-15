#pragma once

#include "AMIDIPlayer.h"
#include "Fluidsynth/types.h"

class MIDIPARSEREXPORT FluidsynthMIDIPlayer : public AMIDIPlayer
{
private:
    using Super = AMIDIPlayer;

    fluid_settings_t* settings;
    fluid_synth_t* synth;
    fluid_audio_driver_t* adriver;

public:
    FluidsynthMIDIPlayer();

    int LoadSoundfont(const char* sfPath);
    virtual void OnNoteOn(const NoteOn& noteOn) override;
    virtual void OnNoteOff(const NoteOff& noteOff) override;
    virtual void OnNoteOnOff(const NoteOnOff& noteOnOff) override;

    virtual void OnProgramChange(const ProgramChange& programChange) override;
    virtual void OnControlChange(const ControlChange& controlChange) override;
    virtual void OnPitchBend(const PitchBend& pitchBend) override;
};

extern "C"
{
    MIDIPARSEREXPORT FluidsynthMIDIPlayer* FluidsynthMIDIPlayer_Create();
    MIDIPARSEREXPORT void FluidsynthMIDIPlayer_Delete(class FluidsynthMIDIPlayer* player);
}
