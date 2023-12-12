#pragma once

#include "MIDIEvents.h"
#include <fluidsynth.h>
#include <vector>
#include <memory>
#include <mutex>
// .cpp
#include <chrono>
#include <thread>

class MIDIPARSEREXPORT Player
{
private:
    fluid_settings_t* settings;
    fluid_synth_t* synth;
    fluid_audio_driver_t* adriver;

    std::vector<int> trackIndices;

public:
    std::vector<std::vector<MIDIEvent*>> notesPerTrack;
    double time = 0.f;
    double addedTime = 0.f;

    std::atomic<bool> isPlaying;
    std::mutex m;

    uint32_t GetPlayerTime() const
    {
        return uint32_t((time + addedTime));
    }

    void InsertEvent(uint32_t track, MIDIEvent* event)
    {
        std::vector<MIDIEvent*>::iterator it;
        for (it = notesPerTrack[track].begin(); it != notesPerTrack[track].end() && (*it)->start <= event->start; ++it);

        notesPerTrack[track].insert(it, event);
    }

    Player()
    {
        settings = new_fluid_settings();
        synth = new_fluid_synth(settings);

        // Create an audio driver
        adriver = new_fluid_audio_driver(settings, synth);

        // Increase the volume (gain) for the entire synth
        double volume = 0.1; // Adjust the volume level (1.0 is the default, higher values increase the volume)
        fluid_settings_setnum(settings, "synth.gain", volume);
    }

    ~Player()
    {
        for (auto& notes : notesPerTrack)
        {
            for (auto& note : notes)
            {
                delete note;
            }
        }
    }

    void SetTime(double newTime)
    {
        m.lock();

        for (int i = 0; i < 16; i++)
        {
            fluid_synth_all_notes_off(synth, i);
        }

        //time = newTime;
        addedTime = newTime - time;

        for (int track = 0; track < GetNbTracks(); track++)
        {
            trackIndices[track] = 0;
        }

        uint32_t timeInMs = GetPlayerTime();
        for (int track = 0; track < GetNbTracks(); track++)
        {
            while (trackIndices[track] < notesPerTrack[track].size())
            {
                MIDIEvent* note = notesPerTrack[track][trackIndices[track]];

                //note->Play();

                if (note->start < timeInMs)
                {
                    trackIndices[track]++;
                }
                else
                {
                    break;
                }
            }
        }

        m.unlock();
    }

    inline uint16_t GetNbTracks() const
    {
        return notesPerTrack.size();
    }

    int LoadSoundfont(const char* sfPath)
    {
        return fluid_synth_sfload(synth, sfPath, 1);
    }

    void Play()
    {
        isPlaying = true;
        trackIndices.resize(GetNbTracks());

        std::chrono::time_point programBeginTime = std::chrono::high_resolution_clock::now();
        while (isPlaying.load())
        {
            m.lock();

            auto frameStartTime = std::chrono::high_resolution_clock::now();

            time = std::chrono::duration<double, std::milli>(frameStartTime - programBeginTime).count();
            double finalTime = time + addedTime;

            // the work...
            uint32_t timeInMs = uint32_t(finalTime);
            for (int track = 0; track < GetNbTracks(); track++)
            {
                while (trackIndices[track] < notesPerTrack[track].size() && notesPerTrack[track][trackIndices[track]]->start < timeInMs)
                {
                    MIDIEvent* note = notesPerTrack[track][trackIndices[track]];
                    note->synth = synth;
                    note->Play();
                    trackIndices[track]++;
                }
            }

            m.unlock();

            //auto frameEndTime = std::chrono::high_resolution_clock::now();

            //double dt = std::chrono::duration<double, std::milli>(frameEndTime - frameStartTime).count();

            //time += dt;
        }
    }
};
