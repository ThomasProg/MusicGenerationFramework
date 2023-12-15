#pragma once


#include <mutex>
#include <map>

// .cpp
#include <chrono>
#include <thread>

class Sequencer
{
public:
    class Event
    {
    public:
        uint32_t startTime = 0; // in ms // TODO : remove ?

        virtual void Play() = 0;
        virtual ~Event() noexcept {}
    };

private:
    // std::vector<int> trackIndices;
    // std::vector<std::vector<MIDIEvent*>> notesPerTrack;
    std::multimap<uint32_t, Event*> events;
    std::multimap<uint32_t, Event*>::iterator currentEvent;

    // Set once when a playback starts
    std::chrono::steady_clock::time_point startTime;
    uint32_t playbackOffset = 0.f; // in ms

    // Set every frame
    uint32_t playbackTime = 0.f; // in ms

    // double time = 0.f;
    // double addedTime = 0.f;

    std::atomic<bool> isPlaying;
    std::mutex m;
    std::thread sequencerThread;

public:
    Sequencer()
    {
        currentEvent = events.begin();
    }
    ~Sequencer()
    {
        isPlaying = false;
        if (sequencerThread.joinable())
        {
            sequencerThread.join();
        }
    }

    bool IsPlaying() const
    {
        return isPlaying.load();
    }

    uint32_t GetPlaybackTime() const
    {
        return playbackTime;
    }

    void InsertEvent(Event* event)
    {
        m.lock();
        events.emplace(event->startTime, event);
        m.unlock();
    }

    void SetTime(uint32_t newTime)
    {
        m.lock();

        // for (int i = 0; i < 16; i++)
        // {
        //     fluid_synth_all_notes_off(synth, i);
        // }

        //time = newTime;

        startTime = std::chrono::high_resolution_clock::now();
        playbackOffset = newTime;

        for (currentEvent = events.begin(); currentEvent != events.end() && currentEvent->second->startTime < playbackOffset; currentEvent++);

        m.unlock();
    }

    void PlayAsync()
    {
        sequencerThread = std::thread([&](){Play();});
    }

    void PrePlay()
    {
        isPlaying = true;

        SetTime(0);
    }

    void Play()
    {
        PrePlay();

        while (isPlaying.load())
        {
            Step();
        }
    }

    void Step()
    {
        m.lock();

        auto frameStartTime = std::chrono::high_resolution_clock::now();
        playbackTime = std::chrono::duration<float, std::milli>(frameStartTime - startTime).count() - playbackOffset;

        // the work...
        while (currentEvent != events.end() && currentEvent->first < playbackTime)
        {
            currentEvent->second->Play();
            currentEvent++;
        }

        m.unlock();
    }

    void Step(uint32_t dt/*in milliseconds*/)
    {
        m.lock();

        playbackTime += dt;

        // the work...
        while (currentEvent != events.end() && currentEvent->first < playbackTime)
        {
            currentEvent->second->Play();
            currentEvent++;
        }

        m.unlock();
    }
};

extern "C"
{
    Sequencer* Sequencer_Create();
    void Sequencer_Delete(Sequencer* sequencer);
    void Sequencer_Play(Sequencer* sequencer);
    
    // More performance heavy than simply using Sequencer.InsertEvent()
    void Sequencer_AddEvent(Sequencer* sequencer, uint32_t startTime, void (*onPlay)(int32_t eventStartTime, void* data), void* data)
    {
        class CustomEvent : public Sequencer::Event
        {
        public:
            void (*onPlay)(int32_t eventStartTime, void*);
            void* data;

            virtual void Play() override
            {
                onPlay(startTime, data);
            }
        }; 

        CustomEvent* customEvent = new CustomEvent();
        customEvent->data = data;
        customEvent->onPlay = onPlay;
        customEvent->startTime = startTime;
        sequencer->InsertEvent(customEvent);
    }
}