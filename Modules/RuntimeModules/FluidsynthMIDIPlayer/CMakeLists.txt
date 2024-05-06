cmake_minimum_required(VERSION 3.12)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_BUILD_TYPE debug)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++23")

# Set compiler flags to enable C++ exceptions
if(CMAKE_COMPILER_IS_GNUCXX)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fexceptions")
elseif(MSVC)
    # Add any necessary flags for MSVC
    # For MSVC, exceptions are enabled by default
endif()

project(FluidsynthMIDIPlayer)

find_package(FluidSynth REQUIRED)

FILE(GLOB SOURCES *.cpp *.h)

# Create the shared library
add_library(FluidsynthMIDIPlayer SHARED ${SOURCES})

if (UNIX)
# Use PKG_CONFIG to find Fluidsynth
find_package(PkgConfig REQUIRED)
pkg_check_modules(FLUIDSYNTH REQUIRED fluidsynth)

target_link_libraries(FluidsynthMIDIPlayer PRIVATE ${FluidSynth_LIBRARIES} -lfluidsynth)
target_include_directories(FluidsynthMIDIPlayer PRIVATE ${FluidSynth_INCLUDE_DIRS})

endif (UNIX)

if (WIN32)
# Link the library with Fluidsynth
target_link_libraries(FluidsynthMIDIPlayer PRIVATE "${ModulesDirectory}/fluidsynth/Downloads/lib/libfluidsynth.dll.a")
target_include_directories(FluidsynthMIDIPlayer PUBLIC
    "${ModulesDirectory}/fluidsynth/Downloads/include"
)
endif (WIN32)

target_link_libraries(FluidsynthMIDIPlayer PRIVATE EasyMidiFileParserCpp)

# Set include directories
target_include_directories(FluidsynthMIDIPlayer PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    "${ModulesDirectory}/EasyMidiFileParserCpp/Public"
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

# add_custom_command(TARGET FluidsynthMIDIPlayer
#   POST_BUILD
#   COMMAND ${CMAKE_COMMAND} -E copy_directory 
#   "${ModulesDirectory}/fluidsynth/Downloads/bin/" 
#   ${FluidsynthMIDIPlayer_BINARY_DIR}/$<CONFIG>/.
# )

# add_custom_command(TARGET FluidsynthMIDIPlayer
#   POST_BUILD
#   COMMAND ${CMAKE_COMMAND} -E copy_directory 
#   ${EasyMidiFileParserCpp_BINARY_DIR}/$<CONFIG>/.
#   ${FluidsynthMIDIPlayer_BINARY_DIR}/$<CONFIG>/.
# )

add_subdirectory(Standalone)