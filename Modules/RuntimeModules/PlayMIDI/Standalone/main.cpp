#include "ConvertingParser.h"
#include "MIDIEvents.h"
#include "Player.h"
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
        // midiPath = "C:/Users/thoma/PandorasBox/Projects/ModularMusicGenerationModules/Assets/Datasets/LakhMidi-Clean/Ludwig_van_Beethoven/5th_Symphony.2.mid";
        // midiPath = "C:/Users/thoma/PandorasBox/Projects/ModularMusicGenerationModules/Assets/Datasets/LakhMidi-Clean/Blondie/Call_Me.2.mid";
        midiPath = "C:/Users/thoma/Downloads/Never-Gonna-Give-You-Up-3.mid";
    }

    ConvertingParser parser;
    parser.parser.LoadFromFile(midiPath.c_str());

    // Load a SoundFont file
    std::string sfPath;
    std::cout << "Path of the soundfont file:" << std::endl;
    std::cin >> sfPath;
    if (sfPath == "n")
    {
        //  sfPath = "C:/Users/thoma/PandorasBox/Projects/ModularMusicGenerationModules/Assets/Soundfonts/Touhou/Touhou.sf2"; 
        // sfPath = "C:/Users/thoma/Downloads/Minecraft/Minecraft Note Block Studio (ver3.3.4).sf2"; 
        sfPath = "C:/Users/thoma/Downloads/Minecraft/Minecraft GM.sf2"; 
         //sfPath = "C:/Users/thoma/PandorasBox/Projects/ModularMusicGenerationModules/Assets/Soundfonts/Undertale/undertale.sf2"; 
        //sfPath = "C:/Users/thoma/PandorasBox/Projects/ModularMusicGenerationModules/Assets/Soundfonts/Roland_SC88/Roland_SC-88.sf2"; 
    }

    Player player;
    int sf = player.LoadSoundfont(sfPath.c_str());
    player.notesPerTrack = std::move(parser.notesPerTrack);

    auto future = std::async(std::launch::async, [&player]
    { 
        player.Play();
    });

    while (1)
    {
        std::string s;
        std::cin >> s;

        int newTime = std::stof(s) * 1000;
        player.SetTime(newTime);
    }
}