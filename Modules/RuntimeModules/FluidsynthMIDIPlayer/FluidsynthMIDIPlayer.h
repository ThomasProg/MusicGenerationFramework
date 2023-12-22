#pragma once

#include "AMIDIPlayer.h"

typedef struct _fluid_hashtable_t fluid_settings_t;             /**< Configuration settings instance */
typedef struct _fluid_synth_t fluid_synth_t;                    /**< Synthesizer instance */
typedef struct _fluid_audio_driver_t fluid_audio_driver_t;      /**< Audio driver instance */

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
    virtual void OnPitchBend(const PitchBend& pitchBend);
};
