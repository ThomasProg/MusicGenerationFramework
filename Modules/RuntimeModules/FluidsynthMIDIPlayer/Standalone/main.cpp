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

// Monotrack only
std::map<uint32_t, uint32_t> GetCountPerProgram(const MIDIMusic& music, uint32_t& nbBeats)
{
    std::map<uint32_t, uint32_t> programToCount;
    std::map<uint32_t, uint32_t> channelToProgram;
    nbBeats = 0;
    for (auto& track : music.tracks)
    {
        for (auto& e : track.midiEvents)
        {
            if (std::shared_ptr<ProgramChange> event = dynamic_pointer_cast<ProgramChange>(e))
            {
                channelToProgram[event->channel] = event->newProgram;
            }
            else if (std::shared_ptr<NoteOnOff> event = dynamic_pointer_cast<NoteOnOff>(e))
            {
                if (event->channel == 9) nbBeats++;
                else programToCount[channelToProgram[event->channel]]++;
            }
            else if (std::shared_ptr<NoteOn> event = dynamic_pointer_cast<NoteOn>(e))
            {
                if (event->channel == 9) nbBeats++;
                else programToCount[channelToProgram[event->channel]]++;
            }
        }
    }
    return programToCount;
}



int main()
{
    std::string midiPath;
    std::cout << "Path of the midi file:" << std::endl;
    std::cin >> midiPath;
    if (midiPath == "n")
    {
        //midiPath = "C:/Users/thoma/PandorasBox/Projects/ModularMusicGenerationModules/Assets/Datasets/LakhMidi-full/1/1a0b35079fd7d1e6d007e59f923643f4.mid"; 
        // midiPath = "C:/Users/thoma/PandorasBox/Projects/ModularMusicGenerationModules/Assets/Datasets/LakhMidi-Clean/Ludwig_van_Beethoven/Fur_Elise.1.mid";
        midiPath = ASSETS_PATH "Fur_Elise.1.mid";
          //midiPath = "C:/Users/thoma/PandorasBox/Projects/ModularMusicGenerationModules/Assets/Datasets/LakhMidi-Clean/Ludwig_van_Beethoven/Fur_Elise.mid";
         //midiPath = "C:/Users/thoma/PandorasBox/Projects/ModularMusicGenerationModules/Assets/Datasets/LakhMidi-Clean/Ludwig_van_Beethoven/Menuet.mid";
        //midiPath = "C:/Users/thoma/PandorasBox/Projects/ModularMusicGenerationModules/Assets/Datasets/LakhMidi-Clean/The_Beatles/Devil_in_Her_Heart.mid";
         //midiPath = "C:/Users/thoma/PandorasBox/Projects/ModularMusicGenerationModules/Assets/Datasets/LakhMidi-Clean/Ludwig_van_Beethoven/5th_Symphony.mid";
        // midiPath = "C:/Users/thoma/PandorasBox/Projects/ModularMusicGenerationModules/Assets/Datasets/LakhMidi-Clean/Blondie/Call_Me.2.mid";
        //midiPath = "C:/Users/thoma/Downloads/Never-Gonna-Give-You-Up-3.mid";
    }

    // ConvertingParser parser;
    // parser.parser.LoadFromFile(midiPath.c_str());

    MIDIMusic music;
    try
    {
        MIDIMusicFiller filler;
        filler.music = &music;

        MIDIParserBase parserBase;
        parserBase.observer = &filler;
        parserBase.LoadFromFile(midiPath.c_str());

        MIDIMusic_MonoTrackConverter().Convert(music);
        MIDIMusic_NoteOnOffConverter().Convert(music);
        //MIDIMusic_InstrumentFilter(30, 30, false).Convert(music);

        //MIDIMusic_NoteOnOffConverter().Convert(music);
        //MIDIMusic_CompressorConverter(4*4).Convert(music);
    }
    catch (const std::exception& e)
    {
        std::cout << "ERROR : MelodyGenerator_LoadFile : " << e.what() << std::endl;
        return 0;
    }

    EventsPrinter printer;
    //for (auto& track : music.tracks)
    //{
    //    for (auto& e : track.midiEvents)
    //    {
    //        e->Execute(&printer);
    //    }
    //}

    std::cout << "NbTracks : " << music.tracks.size() << std::endl;
    std::cout << "NbChannels : " << music.GetNbChannels() << std::endl;
    std::cout << "Duration (in seconds) : " << music.GetDurationInMicroseconds() / 1000.0 / 1000.0 << std::endl;

    uint32_t nbBeats;
    std::map<uint32_t, uint32_t> counterPerProgram = GetCountPerProgram(music, nbBeats);
    for (auto& [program, count] : counterPerProgram)
    {
        std::cout << "Program : " << program << " / Count : " << count << std::endl;
    }
    std::cout << "Percussions Count : " << nbBeats << std::endl;

    // Load a SoundFont file
    std::string sfPath;
    std::cout << "Path of the soundfont file:" << std::endl;
    std::cin >> sfPath;
    if (sfPath == "n")
    {
        //   sfPath = "C:/Users/thoma/PandorasBox/Projects/ModularMusicGenerationModules/Assets/Soundfonts/Touhou/Touhou.sf2"; 
          sfPath = ASSETS_PATH "Touhou.sf2"; 
        // sfPath = "C:/Users/thoma/Downloads/Minecraft/Minecraft Note Block Studio (ver3.3.4).sf2"; 
        //sfPath = "C:/Users/thoma/Downloads/Minecraft/Minecraft GM.sf2"; 
         //sfPath = "C:/Users/thoma/PandorasBox/Projects/ModularMusicGenerationModules/Assets/Soundfonts/Undertale/undertale.sf2"; 
        //sfPath = "C:/Users/thoma/PandorasBox/Projects/ModularMusicGenerationModules/Assets/Soundfonts/Roland_SC88/Roland_SC-88.sf2"; 
    }

    FluidsynthMIDIPlayer player;
    MIDIPlayerAsync playerAsync;
    playerAsync.player = &player;

    int sf = player.LoadSoundfont(sfPath.c_str());
    player.music = &music;
    // player.notesPerTrack = std::move(parser.notesPerTrack);


    //std::cout << music.tracks.size();
    //music.tracks.erase(music.tracks.begin() + 0);
    //music.tracks.erase(music.tracks.begin() + 2);
    //music.tracks[0] = music.tracks[2];
    //music.tracks.resize(1);

    //playerAsync.Play();
    auto future = std::async(std::launch::async, [&playerAsync]
    { 
        playerAsync.Play();
        // player.Play();
    });

    while (1)
    {
        std::string s;
        std::cin >> s;

        int newTime = std::stof(s) * 1000;
        // playerAsync.SetTime(newTime);
        // player.SetTime(newTime);
    }
}