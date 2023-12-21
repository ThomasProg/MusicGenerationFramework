#include "MelodyGenerator_LoadFile.h"
#include "MIDIMusic.h"
#include <iostream>

void MelodyGenerator_LoadFile::OnStart()
{
    MIDIMusic music;
    try
    {
        MIDIMusicFiller filler;
        filler.music = &music;

        MIDIParserBase parserBase;
        parserBase.observer = &filler;
        parserBase.LoadFromFile(filePath);
    }
    catch (const std::exception& e)
    {
        std::cout << "ERROR : MelodyGenerator_LoadFile : " << e.what() << std::endl;
        return;
    }

    if (music.tracks.size() > 0)
    {
        std::cout << "Amount of tracks : " << music.tracks.size() << std::endl;

        for (MIDIMusic::TrackData& trackData : music.tracks)
        {
            for (std::shared_ptr<PMIDIEvent>& e : trackData.midiEvents)
            {
                std::shared_ptr<NoteOn> noteOn = std::dynamic_pointer_cast<NoteOn>(e);

                if (noteOn)
                {
                    buffer.push_back(noteOn->key);
                }

                // NoteOff* noteOff = dynamic_cast<NoteOff*>(e.get());
                // if (noteOff)
                // {

                // }
            }
        }

        onBufferGenerated(this);
    }
}

MelodyGenerator_LoadFile* CreateMelodyGenerator_LoadFile()
{
    return new MelodyGenerator_LoadFile();
}