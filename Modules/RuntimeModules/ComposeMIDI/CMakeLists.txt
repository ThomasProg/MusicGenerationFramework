cmake_minimum_required(VERSION 3.10)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# FILE(GLOB SOURCES Src/* Include/*)
FILE(GLOB SOURCES *.cpp *.h)
set(TargetName "ComposeMIDI")
project(${TargetName})

# Create dynamic lib (.dll/.so)
# add_library(${TargetName} SHARED ${SOURCES})
add_executable(${TargetName} ${SOURCES})

# Add EasyMidiFileParserCpp
target_link_libraries(${TargetName} EasyMidiFileParserCpp)
target_link_libraries(${TargetName} FluidsynthMIDIPlayer)

# Add fluidsynth
set(enable-pkgconfig false) 
set(BUILD_SHARED_LIBS ON)

target_link_libraries(${TargetName} "${ModulesDirectory}/SFML/Downloads/lib/sfml-system.lib")
target_link_libraries(${TargetName} "${ModulesDirectory}/SFML/Downloads/lib/sfml-window.lib")
target_link_libraries(${TargetName} "${ModulesDirectory}/SFML/Downloads/lib/sfml-main.lib")


target_link_libraries(${TargetName} "${ModulesDirectory}/fluidsynth/Downloads/lib/libfluidsynth.dll.a")
target_compile_definitions(${TargetName} PRIVATE MMG_EXPORT)

target_include_directories(${TargetName} PUBLIC 
    "${PROJECT_SOURCE_DIR}/Public/" 
    "${ModulesDirectory}/fluidsynth/Downloads/include"
    "${ModulesDirectory}/EasyMidiFileParserCpp/Public"
    "${ModulesDirectory}/FluidsynthMIDIPlayer"
    "${ModulesDirectory}/SFML/Downloads/include"
)

add_custom_command(TARGET ${TargetName}
  POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E copy_directory 
  "${ModulesDirectory}/SFML/Downloads/bin/" 
  ${ComposeMIDI_BINARY_DIR}/$<CONFIG>/.
)

add_custom_command(TARGET ${TargetName}
  POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E copy_directory 
  "${ModulesDirectory}/fluidsynth/Downloads/bin/" 
  ${ComposeMIDI_BINARY_DIR}/$<CONFIG>/.
)

add_custom_command(TARGET ${TargetName}
  POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E copy_directory 
  ${EasyMidiFileParserCpp_BINARY_DIR}/$<CONFIG>/.
  ${ComposeMIDI_BINARY_DIR}/$<CONFIG>/.
)

add_custom_command(TARGET ${TargetName}
  POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E copy_directory 
  ${FluidsynthMIDIPlayer_BINARY_DIR}/$<CONFIG>/.
  ${ComposeMIDI_BINARY_DIR}/$<CONFIG>/.
)
