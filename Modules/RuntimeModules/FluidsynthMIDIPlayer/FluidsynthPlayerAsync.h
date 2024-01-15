#pragma once

#include "Macros.h"
#include "FluidsynthMIDIPlayer.h"
#include "MIDIPlayerAsync.h"
#include <future>

class MIDIPARSEREXPORT FluidsynthPlayerAsync : public MIDIPlayerAsync
{
public:
    FluidsynthMIDIPlayer fluidsynthPlayer;
    std::future<void> playerFuture;

    FluidsynthPlayerAsync(class MIDIMusic* music);
};

extern "C"
{
    MIDIPARSEREXPORT class FluidsynthPlayerAsync* FluidsynthPlayerAsync_Create(class MIDIMusic* music, const char* soundfontPath);
    MIDIPARSEREXPORT void FluidsynthPlayerAsync_Destroy(class FluidsynthPlayerAsync* player);
    MIDIPARSEREXPORT void FluidsynthPlayerAsync_Play(FluidsynthPlayerAsync* player);
    MIDIPARSEREXPORT void FluidsynthPlayerAsync_PlayAsync(FluidsynthPlayerAsync* player);
}
