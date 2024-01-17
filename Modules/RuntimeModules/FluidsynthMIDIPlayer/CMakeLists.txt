cmake_minimum_required(VERSION 3.12)
project(FluidsynthMIDIPlayer)

find_package(FluidSynth REQUIRED)

FILE(GLOB SOURCES *.cpp *.h)

# Create the shared library
add_library(FluidsynthMIDIPlayer SHARED ${SOURCES})

# Link the library with Fluidsynth
target_link_libraries(FluidsynthMIDIPlayer "${ModulesDirectory}/fluidsynth/Downloads/lib/libfluidsynth.dll.a")

target_link_libraries(FluidsynthMIDIPlayer EasyMidiFileParserCpp)

# Set include directories
target_include_directories(FluidsynthMIDIPlayer PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    "${ModulesDirectory}/EasyMidiFileParserCpp/Public"
    "${ModulesDirectory}/fluidsynth/Downloads/include"
)

target_include_directories(FluidsynthMIDIPlayer PRIVATE ${FluidSynth_INCLUDE_DIRS})

install(TARGETS FluidsynthMIDIPlayer
        LIBRARY DESTINATION bin
        RUNTIME DESTINATION bin)

file(GLOB DLL_FILES "${ModulesDirectory}/fluidsynth/Downloads/bin/*.dll")

# Install the DLL files to the bin directory
install(FILES ${DLL_FILES}
        DESTINATION bin)

# Set RPATH to include the bin directory (Linux/macOS)
set_target_properties(FluidsynthMIDIPlayer PROPERTIES
                      INSTALL_RPATH "$ORIGIN/bin")
    
set_target_properties(FluidsynthMIDIPlayer PROPERTIES
                      VS_DEBUGGER_WORKING_DIRECTORY "${CMAKE_INSTALL_PREFIX}/bin")

add_custom_command(TARGET FluidsynthMIDIPlayer
  POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E copy_directory 
  "${ModulesDirectory}/fluidsynth/Downloads/bin/" 
  ${FluidsynthMIDIPlayer_BINARY_DIR}/$<CONFIG>/.
)

add_custom_command(TARGET FluidsynthMIDIPlayer
  POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E copy_directory 
  ${EasyMidiFileParserCpp_BINARY_DIR}/$<CONFIG>/.
  ${FluidsynthMIDIPlayer_BINARY_DIR}/$<CONFIG>/.
)

add_subdirectory(Standalone)