cmake_minimum_required(VERSION 3.10)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# FILE(GLOB SOURCES Src/* Include/*)
FILE(GLOB SOURCES *.cpp *.h)
set(TargetName "FluidsynthMIDIPlayerStandalone")
project(${TargetName})

# Create dynamic lib (.dll/.so)
add_executable(${TargetName} ${SOURCES})

# find_package(FluidSynth REQUIRED)

# if(FluidSynth_FOUND)
#     # include_directories(${FluidSynth_INCLUDE_DIRS})
#     target_link_libraries(${TargetName} ${FluidSynth_LIBRARIES})
# else()
#     message(FATAL_ERROR "FluidSynth not found. Please install it.")
# endif()


# Add EasyMidiFileParserCpp
target_link_libraries(${TargetName} EasyMidiFileParserCpp)
target_link_libraries(${TargetName} FluidsynthMIDIPlayer)


target_include_directories(${TargetName} PUBLIC 
    "${PROJECT_SOURCE_DIR}/Public/" 
    # "${ModulesDirectory}/fluidsynth/Downloads/include"
    "${ModulesDirectory}/EasyMidiFileParserCpp/Public"
    "${ModulesDirectory}/FluidsynthMIDIPlayer"
)

# Install the executable
install(TARGETS FluidsynthMIDIPlayerStandalone
    RUNTIME DESTINATION bin
)

# Set RPATH to include the bin directory (Linux/macOS)
set_target_properties(FluidsynthMIDIPlayerStandalone PROPERTIES
                      INSTALL_RPATH "$ORIGIN/bin")
    
# set_target_properties(FluidsynthMIDIPlayerStandalone PROPERTIES
#                       VS_DEBUGGER_WORKING_DIRECTORY "${CMAKE_INSTALL_PREFIX}/bin")

add_custom_command(TARGET FluidsynthMIDIPlayerStandalone
  POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E copy_directory 
  "${ModulesDirectory}/fluidsynth/Downloads/bin/" 
  ${FluidsynthMIDIPlayerStandalone_BINARY_DIR}/$<CONFIG>/.
)

add_custom_command(TARGET FluidsynthMIDIPlayerStandalone
  POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E copy_directory 
  ${EasyMidiFileParserCpp_BINARY_DIR}/$<CONFIG>/.
  ${FluidsynthMIDIPlayerStandalone_BINARY_DIR}/$<CONFIG>/.
)

add_custom_command(TARGET FluidsynthMIDIPlayerStandalone
  POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E copy_directory 
  ${FluidsynthMIDIPlayer_BINARY_DIR}/$<CONFIG>/.
  ${FluidsynthMIDIPlayerStandalone_BINARY_DIR}/$<CONFIG>/.
)
