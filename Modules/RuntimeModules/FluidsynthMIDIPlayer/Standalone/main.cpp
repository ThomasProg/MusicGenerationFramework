// #include "ConvertingParser.h"
// #include "MIDIEvents.h"
#include "MIDIMusic.h"
#include "FluidsynthMIDIPlayer.h"
#include "MIDIPlayerAsync.h"
#include "MIDIParserBase.h"
#include "Converters/MIDIMusic_NoteOnOffConverter.h"
#include "Converters/MIDIMusic_CompressorConverter.h"
#include "Converters/MIDIMusic_MonoTrackConverter.h"
#include <iostream>
#include <future>

#include "EventsPrinter.h"

int main()
{
    std::string midiPath;
    std::cout << "Path of the midi file:" << std::endl;
    std::cin >> midiPath;
    if (midiPath == "n")
    {
        //midiPath = "C:/Users/thoma/PandorasBox/Projects/ModularMusicGenerationModules/Assets/Datasets/LakhMidi-full/1/1a0b35079fd7d1e6d007e59f923643f4.mid"; 
        midiPath = "C:/Users/thoma/PandorasBox/Projects/ModularMusicGenerationModules/Assets/Datasets/LakhMidi-Clean/Ludwig_van_Beethoven/Fur_Elise.1.mid";
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


    {
        auto& track = music.tracks[0];

        for (auto& e : track.midiEvents)
        {
            std::cout << "DT : " << e->deltaTime << std::endl;
        }
    }
    

    double lastTime = 0;
    uint32_t temp = 0.5 * 1000.0 * 1000.0; // microseconds / quarter note

    for (auto& track : music.tracks)
    //auto& track = music.tracks[0];
    {
        double s = 0;
        int32_t trackNbTicks = 0;
        for (auto& e : track.midiEvents)

            //for (auto& e : music.tracks[2].midiEvents)
        {
            //double tempo = 1000.0 * 1000.0 * 60.0 / double(t);
            // 60 / 32 = 1.875
            double tick_duration = temp / music.ticksPerQuarterNote; // micros/ticks = micros / quarterNote    /    ticks / quarterNote
            s += e->deltaTime * tick_duration; // ticks * micros/ticks

            trackNbTicks += e->deltaTime;

            // ticks
            // ticksPerQuarterNote
            // 


            //if (s > lastTime)
            {
                lastTime = s;
                std::cout << e->deltaTime << " / total time : " << s / 1000 / 1000 << " / ";
                e->Execute(&printer);
            }

            if (Tempo* tempo = dynamic_cast<Tempo*>(e.get()))
            {
                temp = tempo->newTempo;
            }

        }

        //std::cout << "TrackNbTicks : " << trackNbTicks << std::endl;

        //double tick_duration = temp / music.ticksPerQuarterNote; // s/ticks = s / quarterNote    /    ticks / quarterNote
        //std::cout << "Track Total Time : " << trackNbTicks * tick_duration / 1000 / 1000 << std::endl;
    }


    // Load a SoundFont file
    std::string sfPath;
    std::cout << "Path of the soundfont file:" << std::endl;
    std::cin >> sfPath;
    if (sfPath == "n")
    {
          sfPath = "C:/Users/thoma/PandorasBox/Projects/ModularMusicGenerationModules/Assets/Soundfonts/Touhou/Touhou.sf2"; 
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


    std::cout << music.tracks.size();
    //music.tracks.erase(music.tracks.begin() + 2);
    //music.tracks[0] = music.tracks[2];
    //music.tracks.resize(1);

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