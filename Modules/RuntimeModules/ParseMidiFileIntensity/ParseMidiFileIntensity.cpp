#include "ParseMidiFileIntensity.h"
#include "AdvancedMIDIParser.h"
#include <fstream>

// Define a function that returns a pointer to a vector of integers.
Intensities parseIntensities(const char* filepath)
{
    class Parser : public AdvancedMIDIParser
    {
        using Super = AdvancedMIDIParser;

    public:
        std::vector<std::vector<int>>* velocitiesPerTrack = new std::vector<std::vector<int>>();
        std::vector<std::vector<int>>* timesPerTrack = new std::vector<std::vector<int>>();

        virtual void OnFileHeaderDataLoaded(FileHeaderData& fileHeaderData) override
        {
            Super::OnFileHeaderDataLoaded(fileHeaderData);
            velocitiesPerTrack->resize(fileHeaderData.nbTracks);
            timesPerTrack->resize(fileHeaderData.nbTracks);
        }

        virtual void OnNoteOn(int channel, int key, int velocity) 
        {
            Super::OnNoteOn(channel, key, velocity);

            (*timesPerTrack)[currentTrackIndex].push_back(timePerTrack[currentTrackIndex]);
            (*velocitiesPerTrack)[currentTrackIndex].push_back(velocity);
        }

        virtual void OnNoteOff(int channel, int key) 
        {
            Super::OnNoteOff(channel, key);
        }
    };

    Parser p;

    std::ifstream file (filepath, std::ios::in|std::ios::binary|std::ios::ate);
    if (file.is_open())
    {
        size_t size = file.tellg();
        char* memblock = new char [size];
        file.seekg (0, std::ios::beg);
        file.read (memblock, size);
        file.close();

        p.LoadFromBytes(memblock, size);

        delete[] memblock;
    }
    else
    {
        throw std::runtime_error(std::string("Couldn't open file : ") + filepath);
    }

    // p.LoadFromFile(filepath);

    Intensities vec;
    // vec.length = velocities->size();
    // vec.data = velocities->data();
    // vec.ptr = velocities;

    vec.nbTracks = p.nbTracks;
    vec.timesPerTrack = p.timesPerTrack;
    vec.velocitiesPerTrack = p.velocitiesPerTrack;

    return vec;
}

// Define a function that takes a pointer to a vector and deletes it.
void deleteIntensities(Intensities vec)
{
    delete vec.timesPerTrack;
    delete vec.velocitiesPerTrack;
}

VectorPtr getTrackVelocities(Intensities vec, int track)
{
    // if (vec.velocitiesPerTrack == nullptr)
    // {
    //     VectorPtr v;
    //     v.data = nullptr;
    //     v.length = 0;
    //     return v;
    // }

    VectorPtr v;
    v.data = (*vec.velocitiesPerTrack)[track].data();
    v.length = (*vec.velocitiesPerTrack)[track].size();
    return v;
}

VectorPtr getTrackTimings(Intensities vec, int track)
{
    // if (vec.timesPerTrack == nullptr || (*vec.timesPerTrack).size() >= track)
    // {
    //     VectorPtr v;
    //     v.data = nullptr;
    //     v.length = 0;
    //     return v;
    // }

    VectorPtr v;
    v.data = (*vec.timesPerTrack)[track].data();
    v.length = (*vec.timesPerTrack)[track].size();
    return v;
}
