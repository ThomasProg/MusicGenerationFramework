// #include "ConvertingParser.h"
// #include "MIDIEvents.h"
#include "MIDIMusic.h"
#include "FluidsynthMIDIPlayer.h"
#include "MIDIPlayerAsync.h"
#include "MIDIParserBase.h"
#include "Converters/MIDIMusic_NoteOnOffConverter.h"
#include "Converters/MIDIMusic_CompressorConverter.h"
#include <iostream>
#include <future>

int main()
{
    std::string midiPath;
    std::cout << "Path of the midi file:" << std::endl;
    std::cin >> midiPath;
    if (midiPath == "n")
    {
        //midiPath = "C:/Users/thoma/PandorasBox/Projects/ModularMusicGenerationModules/Assets/Datasets/LakhMidi-full/1/1a0b35079fd7d1e6d007e59f923643f4.mid"; 
        //midiPath = "C:/Users/thoma/PandorasBox/Projects/ModularMusicGenerationModules/Assets/Datasets/LakhMidi-Clean/Ludwig_van_Beethoven/Fur_Elise.1.mid";
          midiPath = "C:/Users/thoma/PandorasBox/Projects/ModularMusicGenerationModules/Assets/Datasets/LakhMidi-Clean/Ludwig_van_Beethoven/Fur_Elise.mid";
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

        MIDIMusic_NoteOnOffConverter().Convert(music);
        MIDIMusic_CompressorConverter(4*4).Convert(music);
    }
    catch (const std::exception& e)
    {
        std::cout << "ERROR : MelodyGenerator_LoadFile : " << e.what() << std::endl;
        return 0;
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