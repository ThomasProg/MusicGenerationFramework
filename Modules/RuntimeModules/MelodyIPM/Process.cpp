#include "Process.h"
#include "Converters/MIDIMusic_CompressorConverter.h"
#include "Converters/MIDIMusic_NoteOnOffConverter.h"
#include "Converters/MIDIMusic_MonoTrackConverter.h"
#include "Converters/MIDIMusic_InstrumentFilter.h"
#include "Converters/MIDIMusic_AbsoluteConverter.h"
#include "Converters/MIDIMusic_RelativeConverter.h"
#include <cassert>
#include <iostream>

#include "MelodyExtractor.h"

void PrintInformation(MIDIMusic* music)
{
    std::cout << "NbTracks : " << music->tracks.size() << std::endl;
    std::cout << "NbChannels : " << music->GetNbChannels() << std::endl;
    std::cout << "Duration (in seconds) : " << music->GetDurationInMicroseconds() / 1000.0 / 1000.0 << std::endl;
}

void MelodyRunIP1(MIDIMusic* music)
{
    assert(music != nullptr);

    MIDIMusic_NoteOnOffConverter().Convert(*music);

    MIDIMusic_AbsoluteConverter::Convert(*music);
    {
        MIDIMusic_CompressorConverter(4*4).ConvertUnsafe(*music);
        MIDIMusic_MonoTrackConverter().ConvertUnsafe(*music);
    }
    MIDIMusic_RelativeConverter::Convert(*music);

    MIDIMusic_InstrumentFilter::Piano(*music);

    // music->ForEachTrack([](const std::vector<MIDIMusic::TrackData>::iterator& it)
    // {
    //     uint32_t program = 0;
    //     it->ForEachEvent([&program](const std::vector<std::shared_ptr<PMIDIEvent>>::iterator& it)
    //     {
    //         if (const NoteOnOff* noteOnOff = dynamic_cast<const NoteOnOff*>(it->get()))
    //         {
    //             if (noteOnOff->)
    //             // return noteOnOff->
    //         }
    //         else if (const ProgramChange* programChange = dynamic_cast<const ProgramChange*>(it->get()))
    //         {
    //             program = programChange->newProgram;
    //         }
    //         return true;
    //     });
    //     return false;
    // });
}

