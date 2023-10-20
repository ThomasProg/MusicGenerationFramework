cmake_minimum_required(VERSION 3.10)

set(ModulesDirectory "${CMAKE_CURRENT_SOURCE_DIR}")

if(DOWNLOAD_MODULE_DEPENDENCIES)
    add_subdirectory(fluidsynth)
    add_subdirectory(OnnxRuntime)
endif()

add_subdirectory(EasyMidiFileParserCpp)
add_subdirectory(ModularMusicGenerationCore)
add_subdirectory(ModularMusicGenerationTests)
add_subdirectory(BeatsAnalyzer)
add_subdirectory(ParseMidiFileIntensity)