#include "MelodyExtractor.h"
#include "Converters/MIDIMusic_CompressorConverter.h"
#include "Converters/MIDIMusic_NoteOnOffConverter.h"
#include "Converters/MIDIMusic_MonoTrackConverter.h"
#include "Converters/MIDIMusic_InstrumentFilter.h"
#include "Converters/MIDIMusic_ChannelFilter.h"

// // Monotrack only
// std::map<uint32_t, uint32_t> GetCountPerProgram(const MIDIMusic& music)
// {
//     std::map<uint32_t, uint32_t> programToCount;
//     uint32_t currentProgram = 0;
//     for (auto& track : music.tracks)
//     {
//         for (auto& e : track.midiEvents)
//         {
//             if (std::shared_ptr<ProgramChange> event = dynamic_pointer_cast<ProgramChange>(e))
//             {
//                 currentProgram = event->newProgram;
//             }
//             else 
//             {
//                 programToCount[currentProgram]++;
//             }
//         }
//     }
//     return programToCount;
// }

void MelodyExtractor::Convert(class MIDIMusic& music)
{
    std::vector<uint32_t> programs = music.GetProgramsList();

    MIDIMusic_NoteOnOffConverter().Convert(music);
    MIDIMusic_ChannelFilter(9, true); // channel 9 is reserved to beats ; so we are removing beats
    MIDIMusic_MonoTrackConverter().ConvertUnsafe(music);
    
    MIDIMusic_InstrumentFilter(0, 7, false).Convert(music); // only keeping piano

    MIDIMusic_InstrumentFilter(112, 119, true).Convert(music); // Remove Percussive
    MIDIMusic_InstrumentFilter(120, 127, true).Convert(music); // Remove Sound Effects


    MIDIMusic_AbsoluteConverter::Convert(music);
    {
        MIDIMusic_CompressorConverter(4*4).ConvertUnsafe(music);
        MIDIMusic_MonoTrackConverter().ConvertUnsafe(music);
    }
    MIDIMusic_RelativeConverter::Convert(music);
}