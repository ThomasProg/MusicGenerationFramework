cmake_minimum_required(VERSION 3.10)

set(ModulesDirectory "${CMAKE_SOURCE_DIR}/Modules/")

# Add ModularMusicGenerationCore
add_subdirectory(Modules/EasyMidiFileParserCpp)
add_subdirectory(Modules/ModularMusicGenerationCore)
add_subdirectory(Modules/ModularMusicGenerationTests)
add_subdirectory(Modules/LoadAllMIDI)
add_subdirectory(Modules/LoadSingleMIDI)
add_subdirectory(Modules/BeatsAnalyzer)
add_subdirectory(Modules/ParseMidiFileIntensity)