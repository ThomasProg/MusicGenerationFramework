#pragma once

#include <cstdint>
#include "MIDIHelpers.h"

typedef struct _fluid_synth_t fluid_synth_t;  

struct MIDIPARSEREXPORT MIDIEvent
{
    // fluid_synth_t*
    fluid_synth_t* synth = nullptr;
    uint32_t start = 0; // in ms

    virtual void Play() = 0;
    virtual ~MIDIEvent() noexcept {}
};

struct MIDIPARSEREXPORT MIDIChannelEvent : MIDIEvent
{
    uint32_t channel = 0;
};

struct MIDIPARSEREXPORT NoteOn : MIDIChannelEvent
{
    uint32_t key = 0;
    uint32_t velocity = 0;

    virtual void Play() override;
};

struct MIDIPARSEREXPORT NoteOff : MIDIChannelEvent
{
    uint32_t key = 0;

    virtual void Play() override;
};

struct MIDIPARSEREXPORT ControlChange : MIDIChannelEvent
{
    EControlChange ctrl;
    uint32_t value = 0;

    virtual void Play() override;
};

struct MIDIPARSEREXPORT PitchBend : MIDIChannelEvent
{
    uint32_t value = 0;

    virtual void Play() override;
};


struct MIDIPARSEREXPORT ProgramChange : MIDIChannelEvent
{
    uint32_t newProgram = 0;

    virtual void Play() override;
};

// struct MIDIPARSEREXPORT LambdaMIDIEvent : MIDIEvent
// {
//     std::function<void()> lambda;
//     virtual void Play() override
//     {
//         lambda();
//     }
// };

// MIDIPARSEREXPORT NoteOn* CastToNoteOn(MIDIEvent* obj)
// {
//     return dynamic_cast<NoteOn*>(obj);
// }