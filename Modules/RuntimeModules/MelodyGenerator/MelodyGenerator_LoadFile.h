#pragma once

#include "MelodyGenerator.h"
#include <string>

class __declspec(dllexport) MelodyGenerator_LoadFile : public MelodyGenerator
{
private:
    std::string filePath;

public:
    // MelodyGenerator_LoadFile();
    virtual ~MelodyGenerator_LoadFile() = default;

    // virtual int32_t GetBufferSize() const override 
    // {
    //     return buffer.size();
    // }

    virtual void OnStart() override;
    void SetFilePath(const char* newFilePath);
};

extern "C"
{
    __declspec(dllexport) class MelodyGenerator_LoadFile* CreateMelodyGenerator_LoadFile();
}