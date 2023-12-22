#pragma once

#include "MelodyGenerator.h"

class MelodyGenerator_LoadFile : public MelodyGenerator
{
public:
    // MelodyGenerator_LoadFile();
    virtual ~MelodyGenerator_LoadFile() = default;

    const char* filePath = nullptr;

    // virtual int32_t GetBufferSize() const override 
    // {
    //     return buffer.size();
    // }

    virtual void OnStart() override;
};

extern "C"
{
    __declspec(dllexport) class MelodyGenerator_LoadFile* CreateMelodyGenerator_LoadFile();
}