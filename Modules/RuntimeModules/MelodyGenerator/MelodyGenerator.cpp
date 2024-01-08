#include "MelodyGenerator.h"

// MelodyGenerator* CreateMelodyGenerator()
// {
//     return new MelodyGenerator();
// }

MelodyGenerator_Impl1* CreateMelodyGeneratorImpl1()
{
    return new MelodyGenerator_Impl1();
}

void DeleteMelodyGenerator(MelodyGenerator* melodyGenerator)
{
    delete melodyGenerator;
}

void SetOnNotePlayedCallback(MelodyGenerator* generator, OnNotePlayedPtr callback)
{
    generator->onNotePlayed = callback;
}

void SetOnNoteStoppedCallback(MelodyGenerator* generator, OnNoteStoppedPtr callback)
{
    generator->onNoteStopped = callback;
}

void SetUserData(MelodyGenerator* generator, UserDataPtr userData)
{
    generator->userData = userData;
}

void StartGeneration(MelodyGenerator* generator)
{
    generator->OnStart();
}






