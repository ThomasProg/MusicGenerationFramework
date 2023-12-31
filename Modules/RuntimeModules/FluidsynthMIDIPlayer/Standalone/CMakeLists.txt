cmake_minimum_required(VERSION 3.10)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# FILE(GLOB SOURCES Src/* Include/*)
FILE(GLOB SOURCES *.cpp *.h)
set(TargetName "FluidsynthMIDIPlayerStandalone")
project(${TargetName})

# Create dynamic lib (.dll/.so)
add_executable(${TargetName} ${SOURCES})

# Add EasyMidiFileParserCpp
target_link_libraries(${TargetName} EasyMidiFileParserCpp)

target_link_libraries(${TargetName} FluidsynthMIDIPlayer)

# Add fluidsynth
set(enable-pkgconfig false) 
set(BUILD_SHARED_LIBS ON)

target_link_libraries(${TargetName} "${ModulesDirectory}/fluidsynth/Downloads/lib/libfluidsynth.dll.a")
target_compile_definitions(${TargetName} PRIVATE MMG_EXPORT)

target_include_directories(${TargetName} PUBLIC 
    "${PROJECT_SOURCE_DIR}/Public/" 
    "${ModulesDirectory}/fluidsynth/Downloads/include"
    "${ModulesDirectory}/EasyMidiFileParserCpp/Public"
    "${ModulesDirectory}/FluidsynthMIDIPlayer"
)

add_custom_command(TARGET ${TargetName}
  POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E copy_directory 
  "${ModulesDirectory}/fluidsynth/Downloads/bin/" 
  ${FluidsynthMIDIPlayerStandalone_BINARY_DIR}/$<CONFIG>/.
)

add_custom_command(TARGET ${TargetName}
  POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E copy_directory 
  ${EasyMidiFileParserCpp_BINARY_DIR}/$<CONFIG>/.
  ${FluidsynthMIDIPlayerStandalone_BINARY_DIR}/$<CONFIG>/.
)

add_custom_command(TARGET ${TargetName}
  POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E copy_directory 
  ${FluidsynthMIDIPlayer_BINARY_DIR}/$<CONFIG>/.
  ${FluidsynthMIDIPlayerStandalone_BINARY_DIR}/$<CONFIG>/.
)
