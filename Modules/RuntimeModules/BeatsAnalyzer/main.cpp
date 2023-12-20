#include <filesystem>
#include <iostream>
#include <fstream>
#include <fluidsynth.h>
#include <thread>
#include "MIDIParserException.h"
#include "LoggingMIDIParser.h"

#pragma once

#include "AdvancedMIDIParser.h"
#include <fstream>

class AnalyzerMIDIParser : public AdvancedMIDIParser
{
public:
    using Super = AdvancedMIDIParser;

    std::map<uint32_t, std::vector<uint32_t>> notesPerTime;
    uint32_t lastTime = 0;

    virtual void OnNoteOn(int channel, int key, int velocity) 
    {
        // if (channel != 2)
        //     return;

        // if (key < 40)
        //     return;

        uint32_t time = timePerTrack[currentTrackIndex];// *(tempo / ticksPerQuarterNote) / 1000;

        if (time - lastTime < 50)
        {
            time = lastTime;
        }
        lastTime = time;

        auto it = notesPerTime.find(time);
        if (it != notesPerTime.end())
        {
            it->second.push_back(key);
        }
        else 
        {
            std::vector<uint32_t> temp;
            temp.push_back(key);
            notesPerTime.emplace(time, std::move(temp));
        }
    }
//     virtual void OnNoteOff(int channel, int key) 
//     {
//         timeScheduler.RunAt(timePerTrack[currentTrackIndex] * (tempo / ticksPerQuarterNote) / 1000, [channel, key]()
//         {
//             fluid_synth_noteoff(synth, channel, key);
//         });
//     }
//     virtual void OnProgramChange(int channel, int program) 
//     {
//         timeScheduler.RunAt(timePerTrack[currentTrackIndex] * (tempo / ticksPerQuarterNote) / 1000, [channel, program]()
//         {
//             fluid_synth_program_change(synth, channel, program);
//         });
//     }
//     virtual void OnControlChange(int channel, EControlChange ctrl, int value) 
//     {
//         timeScheduler.RunAt(timePerTrack[currentTrackIndex] * (tempo / ticksPerQuarterNote) / 1000, [channel, ctrl, value]()
//         {
//             fluid_synth_cc(synth, channel, (int)ctrl, value);
//         });
//     }
//     virtual void OnPitchBend(int channel, int value) 
//     {
//         timeScheduler.RunAt(timePerTrack[currentTrackIndex], [channel, value]()
//         {
//             fluid_synth_pitch_bend(synth, channel, value);
//         });
//     }


//     virtual void OnKeySignature(uint8_t sf, uint8_t mi) override
//     {
//         // std::cout << "sf : " << (int)sf << std::endl;
//         // if (mi == 0)
//         //     std::cout << "mi : major" << std::endl;
//         // else if (mi == 1)
//         //     std::cout << "mi : minor" << std::endl;
//         // else
//         //     std::cout << "mi : " << (int)mi << std::endl;
//     }

//     virtual void OnText(const char* text, uint32_t length) override
//     {
//         // std::cout << "OnText : " << std::string(text, length) << std::endl;
//     }
//     virtual void OnCopyright(const char* copyright, uint32_t length) override
//     {
//         // std::cout << "OnCopyright : " << std::string(copyright, length) << std::endl;
//     }
//     virtual void OnTrackName(const char* trackName, uint32_t length) override
//     {
//         // std::cout << "OnTrackName : " << std::string(trackName, length) << std::endl;
//     }
//     virtual void OnInstrumentName(const char* instrumentName, uint32_t length) override
//     {
//         // std::cout << "OnInstrumentName : " << std::string(instrumentName, length) << std::endl;
//     }
//     virtual void OnLyric(const char* lyric, uint32_t length) override
//     {
//         // std::cout << "OnLyric : " << std::string(lyric, length) << std::endl;
//     }
//     virtual void OnMarker(const char* markerName, uint32_t length) override
//     {
//         // std::cout << "OnMarker : " << std::string(markerName, length) << std::endl;
//     }
//     virtual void OnCuePoint(const char* cuePointName, uint32_t length) override
//     {
//         // std::cout << "OnCuePoint : " << std::string(cuePointName, length) << std::endl;
//     }
};







void displayError(const std::string& s)
{
    std::cout << "\033[1;31m" << s << "\033[0m\n" << std::endl;    
}

void displaySuccess(const std::string& s)
{
    std::cout << "\033[1;32m" << s << "\033[0m\n" << std::endl;    
}

void TryLoadFiles()
{   
    std::cout << "Please enter path : " << std::endl;

    std::string path;
    std::cin >> path;

    try 
    {
        AnalyzerMIDIParser parser;

        std::ifstream file (path, std::ios::in|std::ios::binary|std::ios::ate);
        if (file.is_open())
        {
            size_t size = file.tellg();
            char* memblock = new char [size];
            file.seekg (0, std::ios::beg);
            file.read (memblock, size);
            file.close();

            parser.parser.LoadFromBytes(memblock, size);
            // parser.OnLoadedFromFile(filename);

            delete[] memblock;
        }
        else
        {
            throw std::runtime_error("Couldn't open file : " + path);
        }

        displaySuccess("Loaded with success! " + path);
        // std::cout << "Loaded with success! " << path << std::endl;

        // std::ios::app is the open mode "append" meaning
        // new data will be written to the end of the file.
        std::ofstream out;
        out.open("config.txt", std::ios::app);
        out << path << " : " << "Success\n";

        fluid_settings_t* settings = new_fluid_settings();
        fluid_synth_t* synth = new_fluid_synth(settings);
        fluid_audio_driver_t* adriver = new_fluid_audio_driver(settings, synth);
        // double volume = 0.1; 
        // fluid_settings_setnum(settings, "synth.gain", volume);

        // auto undertalesfID = fluid_synth_sfload(synth, "C:/Users/thoma/Downloads/undertale.sf2", 1);
        int sfID = fluid_synth_sfload(synth, "C:/Users/thoma/Downloads/Touhou.sf2", 1);

        for (const auto& timeIt : parser.notesPerTime)
        {
            // if (timeIt.second.size() < 2)
            //     continue;

            for (const auto& noteIt : timeIt.second)
            {
                std::cout << noteIt << '\t';
                fluid_synth_noteon(synth, 0, noteIt, 127);
            }

            std::this_thread::sleep_for(std::chrono::milliseconds(500));
            std::cout << std::endl;
        }


    }
    catch (const MIDIParserException& e)
    {
        displayError("MIDIParserException : " + std::string(e.what()));
        // std::cout << "MIDIParserException : " << e.what() << std::endl;

        // std::ios::app is the open mode "append" meaning
        // new data will be written to the end of the file.
        std::ofstream out;
        out.open("config.txt", std::ios::app);
        out << path << " : " << "Failure\n";
    }
    catch (const std::exception& e)
    {
        displayError("std::exception : " + std::string(e.what()));
        // std::cout << "std::exception : " << e.what() << std::endl;

        // std::ios::app is the open mode "append" meaning
        // new data will be written to the end of the file.
        std::ofstream out;
        out.open("config.txt", std::ios::app);
        out << path << " : " << "Failure\n";
    }
}

int main()
{
    try 
    {
        TryLoadFiles();
    }
    catch (const std::exception& e)
    {
        std::cout << e.what() << std::endl;
    }
}

