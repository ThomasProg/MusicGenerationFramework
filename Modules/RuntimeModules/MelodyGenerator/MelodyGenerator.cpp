#include "MelodyGenerator.h"
#include <iostream>

MelodyGenerator* CreateMelodyGenerator()
{
    return new MelodyGenerator();
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