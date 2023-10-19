cmake_minimum_required(VERSION 3.10)

# set(ModulesDirectory "${CMAKE_SOURCE_DIR}/Modules/RuntimeModules")
set(ModulesDirectory "${CMAKE_CURRENT_SOURCE_DIR}")

add_subdirectory(EasyMidiFileParserCpp)
add_subdirectory(ModularMusicGenerationCore)
add_subdirectory(ModularMusicGenerationTests)
add_subdirectory(BeatsAnalyzer)
add_subdirectory(ParseMidiFileIntensity)