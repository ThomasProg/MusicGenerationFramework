// #include "ConvertingParser.h"
// #include "MIDIEvents.h"
#include "MIDIMusic.h"
#include "FluidsynthMIDIPlayer.h"
#include "MIDIPlayerAsync.h"
#include "MIDIParserBase.h"
#include "Converters/MIDIMusic_NoteOnOffConverter.h"
#include "Converters/MIDIMusic_CompressorConverter.h"
#include "Converters/MIDIMusic_MonoTrackConverter.h"
#include "Converters/MIDIMusic_InstrumentFilter.h"
#include <iostream>
#include <future>
#include <map>
#include <memory>
#include <cstdint>
#include "PMIDIEvent.h"

#include "EventsPrinter.h"
#include "MIDIMusicFiller.h"

// // Monotrack only
// std::map<uint32_t, uint32_t> GetCountPerProgram(const MIDIMusic& music, uint32_t& nbBeats)
// {
//     std::map<uint32_t, uint32_t> programToCount;
//     std::map<uint32_t, uint32_t> channelToProgram;
//     nbBeats = 0;
//     for (auto& track : music.tracks)
//     {
//         for (auto& e : track.midiEvents)
//         {
//             if (std::shared_ptr<ProgramChange> event = dynamic_pointer_cast<ProgramChange>(e))
//             {
//                 channelToProgram[event->channel] = event->newProgram;
//             }
//             else if (std::shared_ptr<NoteOnOff> event = dynamic_pointer_cast<NoteOnOff>(e))
//             {
//                 if (event->channel == 9) nbBeats++;
//                 else programToCount[channelToProgram[event->channel]]++;
//             }
//             else if (std::shared_ptr<NoteOn> event = dynamic_pointer_cast<NoteOn>(e))
//             {
//                 if (event->channel == 9) nbBeats++;
//                 else programToCount[channelToProgram[event->channel]]++;
//             }
//         }
//     }
//     return programToCount;
// }

#include "Utilities.h"
#include "FluidsynthPlayerAsync.h"

void playMusic()
{
    MIDIMusic* music = MIDIMusic_Create();

    MIDIMusic_LoadFromFile(music, "/home/progz/projects/MusicGenerationFramework/" "Assets/Datasets/LakhMidi-Clean/Ludwig_van_Beethoven/Fur_Elise.1.mid");

    FluidsynthPlayerAsync* player = FluidsynthPlayerAsync_Create(music, "/home/progz/projects/MusicGenerationFramework/" "Assets/Soundfonts/Touhou/Touhou.sf2");
    FluidsynthPlayerAsync_PlayAsync(player);

    FluidsynthPlayerAsync_Destroy(player);
    MIDIMusic_Destroy(music);
}

void createMusic()
{
    MIDIMusic* music = MIDIMusic_Create();

    // MIDIMusic_LoadFromFile(music, "/home/progz/projects/MusicGenerationFramework/" "Assets/Datasets/LakhMidi-Clean/Ludwig_van_Beethoven/Fur_Elise.1.mid");

    NoteOnOff* noteOnOff = NoteOnOff_Create();
    noteOnOff->deltaTime = 0;
    noteOnOff->duration = 50;
    noteOnOff->key = 60;
    noteOnOff->velocity = 120;
    noteOnOff->channel = 0;
    MIDIMusic_AddEvent(music, noteOnOff);


    FluidsynthPlayerAsync* player = FluidsynthPlayerAsync_Create(music, "/home/progz/projects/MusicGenerationFramework/" "Assets/Soundfonts/Touhou/Touhou.sf2");
    FluidsynthPlayerAsync_PlayAsync(player);

    FluidsynthPlayerAsync_Destroy(player);
    MIDIMusic_Destroy(music);
}

int main()
{
    // playMusic();
    createMusic();
}