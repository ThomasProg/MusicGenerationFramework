//#include "MelodyExtractor.h"
#include "MIDIMusic.h"
#include <iostream>
#include <filesystem>
#include <array>

void Analyze(MIDIMusic& music)
{
	std::array<size_t, 12> noteCount;

	for (size_t& count : noteCount)
	{
		count = 0;
	}

	std::vector<std::shared_ptr<NoteOn>> noteOnVec;

	for (MIDIMusic::TrackData& track : music.tracks)
	{
		for (std::shared_ptr<PMIDIEvent>& e : track.midiEvents)
		{

			std::shared_ptr<NoteOn> noteOn = std::dynamic_pointer_cast<NoteOn>(e);

			if (noteOn)
			{
				if (noteOn->key < 21)
					continue;

				//noteOnVec.push_back(noteOn);
				noteCount[(noteOn->key) % 12]++;
			}

			NoteOff* noteOff = dynamic_cast<NoteOff*>(e.get());
			if (noteOff)
			{

			}
		}
	}

	for (MIDIMusic::TrackData& track : music.tracks)
	{
		for (std::shared_ptr<PMIDIEvent>& e : track.midiEvents)
		{

			std::shared_ptr<NoteOn> noteOn = std::dynamic_pointer_cast<NoteOn>(e);

			if (noteOn)
			{
				std::cout << "First note : " << noteOn->key << " / " << noteOn->key % 12 << std::endl;
				break;
			}
		}
	}


	for (size_t count : noteCount)
	{
		std::cout << "- " << count << '\n';
	}
}

int main()
{
	const char* path = "C:/Users/thoma/PandorasBox/Projects/ModularMusicGenerationModules/Assets/Datasets/LakhMidi-Clean";
	for (const auto& entry : std::filesystem::recursive_directory_iterator(path))
	{
		if (entry.is_directory())
			continue;


		//std::string mainFolder = "C:/Users/thoma/PandorasBox/Projects/ModularMusicGenerationModules/Assets/Datasets/";
		//std::string file = "Maestro/2009/MIDI-Unprocessed_01_R1_2009_01-04_ORIG_MID--AUDIO_01_R1_2009_01_R1_2009_02_WAV.midi";
		//std::string file = "LakhMidi-Clean/Ludwig_van_Beethoven/Fur_Elise.mid";

		MIDIMusic music;
		try
		{
			MidiMusicParser parser;
			parser.music = &music;
			parser.parser.LoadFromFile(entry.path().string().c_str());
		}
		catch (const std::exception& e)
		{
		}


		//if ((int)music.mi != 0 || (int)music.sf != 0)
		if (music.isTonalitySet)
			std::cout << (int)music.mi << " / " << (int)music.sf << " / " << entry.path() << '\n';
		else
		{
			std::cout << "pass\n";
			continue;
		}

		Analyze(music);


	}
}