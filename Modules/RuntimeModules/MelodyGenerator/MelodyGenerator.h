#pragma once

#if defined _WIN32 || defined _WIN64
#define EXAMPLELIBRARY_IMPORT __declspec(dllimport)
#elif defined __linux__
#define EXAMPLELIBRARY_IMPORT __attribute__((visibility("default")))
#else
#define EXAMPLELIBRARY_IMPORT
#endif

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

    __declspec(dllexport) MelodyGenerator* CreateMelodyGenerator();
    __declspec(dllexport) void DeleteMelodyGenerator(MelodyGenerator*);

    __declspec(dllexport) void SetOnNotePlayedCallback(MelodyGenerator* generator, OnNotePlayedPtr callback);
    __declspec(dllexport) void SetOnNoteStoppedCallback(MelodyGenerator* generator, OnNoteStoppedPtr callback);

    __declspec(dllexport) void SetUserData(MelodyGenerator* generator, UserDataPtr);

    // __declspec(dllexport) void SetTonalCenter(MelodyGenerator* generator, int tonalCenter /* midi note */);
    // __declspec(dllexport) void SetScale(MelodyGenerator* generator, int* scale, int nb);

    // __declspec(dllexport) void SetBPM(MelodyGenerator* generator, int bpm);
    // __declspec(dllexport) void SetTempo(MelodyGenerator* generator, int tempo);


    __declspec(dllexport) void StartGeneration(MelodyGenerator* generator);
}

// #include <vector>

// template<typename T>
// class CircularBuffer
// {
// public:
//     T* buffer = nullptr;
//     size_t start = 0;
//     size_t end = 0;

//     ~CircularBuffer()
//     {
//         if (buffer != nullptr)
//         {
//             delete buffer;
//         }
//     }

//     virtual void Resize(size_t size)
//     {
//         buffer = new T[size];
//     }

//     T& operator[](size_t index)
//     {
//         return buffer[index % buffer.size()];
//     }
// };


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


    void OnStart()
    {
        FOnNotePlayed note;
        note.note = 60;
        note.velocity = 120;
        onNotePlayed(note, userData);
    }

};
