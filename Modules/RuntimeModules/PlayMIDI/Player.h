#pragma once

#include "MIDIMusic.h"
#include <fluidsynth.h>
#include <vector>
#include <memory>
#include <mutex>
// .cpp
#include <chrono>
#include <thread>
#include <iostream>

class MIDIPARSEREXPORT Player : public IMIDIEventReceiver
{
private:
    using Super = IMIDIEventReceiver;

    fluid_settings_t* settings;
    fluid_synth_t* synth;
    fluid_audio_driver_t* adriver;

    std::vector<uint32_t> trackIndices;
    //std::vector<double> trackLastEventTime;
    std::vector<double> trackLastEventTime;

public:
    class MIDIMusic* music = nullptr;
    // std::vector<std::vector<MIDIEvent*>> notesPerTrack;
    double time = 0.f;
    double addedTime = 0.f;

    // only 3 bytes
    // msPerQuarterNote;
    uint32_t tempo = 500000; // 120 bpm by default 

    std::atomic<bool> isPlaying;
    std::mutex m;

    uint32_t GetPlayerTime() const
    {
        return uint32_t((time + addedTime));
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

    void SetTime(double newTime)
    {

    }

    inline uint16_t GetNbTracks() const
    {
        return music->tracks.size();
    }

    int LoadSoundfont(const char* sfPath)
    {
        return fluid_synth_sfload(synth, sfPath, 1);
    }

    void Play()
    {
        isPlaying = true;
        trackIndices.resize(GetNbTracks());
        trackLastEventTime.resize(GetNbTracks());

        for (double& d : trackLastEventTime)
        {
            d = 0.0;
        }

        std::chrono::time_point programBeginTime = std::chrono::high_resolution_clock::now();
        while (isPlaying.load())
        {
            m.lock();

            auto frameStartTime = std::chrono::high_resolution_clock::now();

            time = std::chrono::duration<double, std::micro>(frameStartTime - programBeginTime).count();
            double finalTime = time + addedTime;

            // the work...
            // uint32_t timeInMs = uint32_t(finalTime);
            for (int track = 0; track < GetNbTracks(); track++)
            {
                MIDIMusic::TrackData& tr = music->tracks[track]; 

                double timeSinceLastEvent = finalTime - trackLastEventTime[track];
                uint32_t tickSinceLastEvent = timeSinceLastEvent * music->ticksPerQuarterNote / double(tempo);

                if (trackIndices[track] < tr.midiEvents.size() && tr.midiEvents[trackIndices[track]]->deltaTime <= tickSinceLastEvent)
                {
                    PMIDIEvent& note = *tr.midiEvents[trackIndices[track]];
                    note.Execute(this);
                    trackIndices[track]++;
                    trackLastEventTime[track] += (double(note.deltaTime * tempo) / music->ticksPerQuarterNote);// microseconds to milliseconds 
                    // timePerTrack[currentTrackIndex] += deltaTime * (tempo / ticksPerQuarterNote); 

                    timeSinceLastEvent = finalTime - trackLastEventTime[track];
                    tickSinceLastEvent = timeSinceLastEvent * music->ticksPerQuarterNote / double(tempo);
                    //std::cout << "Tick2 : " << tickSinceLastEvent << std::endl;
                }
            }

            m.unlock();

            //auto frameEndTime = std::chrono::high_resolution_clock::now();

            //double dt = std::chrono::duration<double, std::milli>(frameEndTime - frameStartTime).count();

            //time += dt;
        }
    }

    virtual void OnNoteOn(const NoteOn& noteOn) override
    { 
        Super::OnNoteOn(noteOn);
        fluid_synth_noteon(synth, noteOn.channel, noteOn.key, noteOn.velocity);
    }
    virtual void OnNoteOff(const NoteOff& noteOff) override
    { 
        Super::OnNoteOff(noteOff);
        fluid_synth_noteoff(synth, noteOff.channel, noteOff.key);
    }
    virtual void OnNoteOnOff(const NoteOnOff& noteOnOff) override
    { 
        Super::OnNoteOnOff(noteOnOff);
        fluid_synth_noteon(synth, noteOnOff.channel, noteOnOff.key, noteOnOff.velocity);
        // TODO : noteoff
    }

    virtual void OnTempo(const Tempo& tempo) 
    {
        Super::OnTempo(tempo); 
        this->tempo = tempo.newTempo;
    }
    virtual void OnProgramChange(const ProgramChange& programChange) 
    {
        Super::OnProgramChange(programChange); 
        fluid_synth_program_change(synth, programChange.channel, programChange.newProgram);
    }
    virtual void OnControlChange(const ControlChange& controlChange) 
    {
        Super::OnControlChange(controlChange); 
        fluid_synth_cc(synth, controlChange.channel, (int)controlChange.ctrl, controlChange.value);
    }
    virtual void OnPitchBend(const PitchBend& pitchBend) 
    {
        Super::OnPitchBend(pitchBend); 
        fluid_synth_pitch_bend(synth, pitchBend.channel, pitchBend.value);
    }
};
