#pragma once

#if defined _WIN32 || defined _WIN64
#define EXAMPLELIBRARY_IMPORT __declspec(dllimport)
#elif defined __linux__
#define EXAMPLELIBRARY_IMPORT __attribute__((visibility("default")))
#else
#define EXAMPLELIBRARY_IMPORT
#endif

#include <vector>

// EXAMPLELIBRARY_IMPORT void ExampleLibraryFunction();

extern "C"
{
    struct FOnNotePlayed
    {
        int note; // midi
        int velocity;
    };

    struct FOnNoteStopped
    {
        int note; // midi
    };

    class MelodyGenerator;

    using UserDataPtr = void*;
    using OnNotePlayedPtr = void(*)(const FOnNotePlayed&, UserDataPtr);
    using OnNoteStoppedPtr = void(*)(const FOnNoteStopped&, UserDataPtr);

    // __declspec(dllexport) MelodyGenerator* CreateMelodyGenerator();
    __declspec(dllexport) class MelodyGenerator_Impl1* CreateMelodyGeneratorImpl1();
    __declspec(dllexport) void DeleteMelodyGenerator(MelodyGenerator*);

    __declspec(dllexport) void SetOnNotePlayedCallback(MelodyGenerator* generator, OnNotePlayedPtr callback);
    __declspec(dllexport) void SetOnNoteStoppedCallback(MelodyGenerator* generator, OnNoteStoppedPtr callback);

    __declspec(dllexport) void SetUserData(MelodyGenerator* generator, UserDataPtr);

    // __declspec(dllexport) void SetTonalCenter(MelodyGenerator* generator, int tonalCenter /* midi note */);
    // __declspec(dllexport) void SetScale(MelodyGenerator* generator, int* scale, int nb);

    // __declspec(dllexport) void SetBPM(MelodyGenerator* generator, int bpm);
    // __declspec(dllexport) void SetTempo(MelodyGenerator* generator, int tempo);


    __declspec(dllexport) void StartGeneration(MelodyGenerator* generator);
    // __declspec(dllexport) void StartGenerationAsync(MelodyGenerator* generator);
}

// class IVelocityGenerator
// {
// public:
//     std::vector<int> pastVelocity;

    
// };

// class ITonalityGenerator
// {
// public:

    
// };

// class IRhythmGenerator
// {
// public:

    
// };



// class Node
// {
// public:
//     void Execute();
// };


// class Constraint
// {

// };

// class Branch : Node
// {
// public:
//     Node* trueCase;
//     Node* falseCase;
// };

// class ISynth
// {

// };




// - Velocity map
// - Tonality map
// - Rhythm (map? / duration of each note)
// - BPM map?



class MelodyGenerator
{
public:
    void* userData = nullptr;
    std::vector<int32_t> buffer;

    OnNotePlayedPtr onNotePlayed;
    OnNoteStoppedPtr onNoteStopped;

    // // Score
    // ITonalityGenerator* tonalityGenerator = nullptr;
    // IRhythmGenerator* rhythmGenerator = nullptr;

    // // Performance
    // IVelocityGenerator* velocityGenerator = nullptr;

    // // Sound
    // ISynth* synth;

public:


    virtual void OnStart()
    {
        FOnNotePlayed note;
        note.note = 60;
        note.velocity = 120;
        onNotePlayed(note, userData);
        // onBufferGenerated([note], userData);
    }

    virtual int32_t GetBufferSize() const = 0;

    using TOnBufferGenerated = void(MelodyGenerator* generator);
    TOnBufferGenerated* onBufferGenerated;

    // virtual int32_t* GenerateNotes();
    virtual ~MelodyGenerator() = default;
};

class MelodyGenerator_Impl1 : public MelodyGenerator
{
public:
    int32_t nbBars = 4;
    int32_t nbNotesPerBar = 4;

    virtual ~MelodyGenerator_Impl1() = default;

    virtual int32_t GetBufferSize() const override 
    {
        return nbNotesPerBar*nbBars;
    }

    virtual void OnStart() override
    {
        buffer.resize(GetBufferSize());

        static constexpr int possibleNotes[] = {0,1,2,3,4,5,6};

        int i = 0;
        for (int32_t& note : buffer)
        {
            note = possibleNotes[i%(sizeof(possibleNotes) / sizeof(*possibleNotes))];
            i++;
        }

        onBufferGenerated(this);
    }
};

