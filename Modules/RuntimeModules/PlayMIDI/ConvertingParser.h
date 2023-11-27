#pragma once

#include "AdvancedMIDIParser.h"
#include "MIDIEvents.h"
#include <memory>

class MIDIPARSEREXPORT ConvertingParser : public AdvancedMIDIParser
{
public:
    using Super = AdvancedMIDIParser;

    std::vector<std::vector<MIDIEvent*>> notesPerTrack;

public:
    ~ConvertingParser()
    {
        for (auto& notes : notesPerTrack)
        {
            for (auto& note : notes)
            {
                delete note;
            }
        }
    }

    virtual void OnFileHeaderDataLoaded(FileHeaderData& fileHeaderData) override
    {
        Super::OnFileHeaderDataLoaded(fileHeaderData);

        notesPerTrack.resize(fileHeaderData.nbTracks);
    }

    virtual void OnNoteOn(int channel, int key, int velocity) override
    {
        Super::OnNoteOn(channel, key, velocity);

        NoteOn* note = new NoteOn();
        note->channel = channel;
        note->key = key;
        note->velocity = velocity;
        note->start = timePerTrack[currentTrackIndex] / 1000;
        notesPerTrack[currentTrackIndex].emplace_back(std::move(note));
    }
    virtual void OnNoteOff(int channel, int key) override
    {
        Super::OnNoteOff(channel, key);

        NoteOff* note = new NoteOff();
        note->channel = channel;
        note->key = key;
        note->start = timePerTrack[currentTrackIndex] / 1000;
        notesPerTrack[currentTrackIndex].emplace_back(std::move(note));
    }
	
    virtual void OnProgramChange(int channel, int program) override
    {
        Super::OnProgramChange(channel, program);

        ProgramChange* note = new ProgramChange();
        note->channel = channel;
        note->newProgram = program;
        note->start = timePerTrack[currentTrackIndex] / 1000;
        notesPerTrack[currentTrackIndex].emplace_back(std::move(note));
    }

    virtual void OnControlChange(int channel, EControlChange ctrl, int value) override
    {
        Super::OnControlChange(channel, ctrl, value);

        ControlChange* note = new ControlChange();
        note->channel = channel;
        note->ctrl = ctrl;
        note->value = value;
        note->start = timePerTrack[currentTrackIndex] / 1000;
        notesPerTrack[currentTrackIndex].emplace_back(std::move(note));
    }

    virtual void OnPitchBend(int channel, int value) override
    {
        Super::OnPitchBend(channel, value);

        PitchBend* note = new PitchBend();
        note->channel = channel;
        note->value = value;
        note->start = timePerTrack[currentTrackIndex] / 1000;
        notesPerTrack[currentTrackIndex].emplace_back(std::move(note));
    }
};

extern "C"
{
	MIDIPARSEREXPORT struct Vector
	{
		int32_t size;
		void* data;
	};

	__declspec(dllexport) ConvertingParser* CreateConvertingParser()
	{
		return new ConvertingParser();
	}

	__declspec(dllexport) void* C()
	{
		return new ConvertingParser();
	}

	MIDIPARSEREXPORT int32_t GetNbTracksFromParser(ConvertingParser* parser)
	{
		return parser->nbTracks;
	}

	MIDIPARSEREXPORT Vector GetNotesFromTrack(ConvertingParser* parser, int track)
	{
		Vector v;
		v.size = parser->notesPerTrack[track].size();
		v.data = parser->notesPerTrack[track].data();
		return v;
	}

	MIDIPARSEREXPORT struct NoteOnStruct
	{
		int32_t start;
		int32_t channel;
		int32_t key;
		int32_t velocity;
		bool isNoteOn;
	};

	MIDIPARSEREXPORT struct ProgramChangeStruct
	{
		int32_t start;
		int32_t channel;
		uint32_t newProgram = 0;
		bool isProgramChange;
	};

	MIDIPARSEREXPORT NoteOnStruct CastToNoteOn(MIDIEvent* obj)
	{
		if (NoteOn* noteOn = dynamic_cast<NoteOn*>(obj))
		{
			NoteOnStruct n;
			n.start = noteOn->start;
			n.channel = noteOn->channel;
			n.key = noteOn->key;
			n.velocity = noteOn->velocity;
			n.isNoteOn = true;
			return n;
		}
		else 
		{
			NoteOnStruct n;
			n.isNoteOn = false;
			return n;
		}
	}

	MIDIPARSEREXPORT ProgramChangeStruct CastToProgramChange(MIDIEvent* obj)
	{
		if (ProgramChange* programChange = dynamic_cast<ProgramChange*>(obj))
		{
			ProgramChangeStruct n;
			n.start = programChange->start;
			n.channel = programChange->channel;
			n.newProgram = programChange->newProgram;
			n.isProgramChange = true;
			return n;
		}
		else
		{
			ProgramChangeStruct n;
			n.isProgramChange = false;
			return n;
		}
	}
}