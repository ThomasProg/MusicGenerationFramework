# cmake_minimum_required(VERSION 3.10)

# set(CMAKE_CXX_STANDARD 23)
# set(CMAKE_CXX_STANDARD_REQUIRED True)

# # FILE(GLOB SOURCES Src/* Include/*)
# FILE(GLOB SOURCES *.cpp *.h)
# set(TargetName "FluidsynthMIDIPlayer")
# project(${TargetName})

# # Create dynamic lib (.dll/.so)
# add_library(${TargetName} SHARED ${SOURCES})

# # Add EasyMidiFileParserCpp
# target_link_libraries(${TargetName} EasyMidiFileParserCpp)



# find_package(FluidSynth REQUIRED)

# if(FluidSynth_FOUND)
#     target_include_directories(${TargetName} PRIVATE ${FluidSynth_INCLUDE_DIRS})
#     target_link_libraries(${TargetName} PRIVATE ${FluidSynth_LIBRARIES})
#     # link_directories(${FluidSynth_LIBRARY_DIRS})
#     # target_link_libraries(${TargetName} ${FluidSynth_LIBRARIES})
# else()
#     message(FATAL_ERROR "FluidSynth not found. Please install it.")
# endif()


# # # Add fluidsynth
# # set(enable-pkgconfig false) 
# # set(BUILD_SHARED_LIBS ON)

# # target_link_libraries(${TargetName} "${ModulesDirectory}/fluidsynth/Downloads/lib/libfluidsynth.dll.a")
# # target_compile_definitions(${TargetName} PRIVATE MMG_EXPORT)

# target_include_directories(${TargetName} PUBLIC 
#     "${PROJECT_SOURCE_DIR}/Public/" 
#     # "${ModulesDirectory}/fluidsynth/Downloads/include"
#     "${ModulesDirectory}/EasyMidiFileParserCpp/Public"
# )

# # add_custom_command(TARGET ${TargetName}
# #   POST_BUILD
# #   COMMAND ${CMAKE_COMMAND} -E copy_directory 
# #   "${ModulesDirectory}/fluidsynth/Downloads/bin/" 
# #   ${FluidsynthMIDIPlayer_BINARY_DIR}/$<CONFIG>/.
# # )

# # add_custom_command(TARGET ${TargetName}
# #   POST_BUILD
# #   COMMAND ${CMAKE_COMMAND} -E copy_directory 
# #   ${EasyMidiFileParserCpp_BINARY_DIR}/$<CONFIG>/.
# #   ${FluidsynthMIDIPlayer_BINARY_DIR}/$<CONFIG>/.
# # )

# add_subdirectory(Standalone)

# target_link_libraries(${TargetName} PRIVATE FluidsynthMIDIPlayer)



cmake_minimum_required(VERSION 3.12)
project(FluidsynthMIDIPlayer)

# Use PKG_CONFIG to find Fluidsynth
find_package(PkgConfig REQUIRED)
pkg_check_modules(FLUIDSYNTH REQUIRED fluidsynth)


# # Set source files for the library
# set(SOURCE_FILES
#     FluidsynthMIDIPlayer.cpp
#     # Add other source files as needed
# )
FILE(GLOB SOURCES *.cpp *.h)

# Create the shared library
add_library(FluidsynthMIDIPlayer SHARED ${SOURCES})

# Link the library with Fluidsynth
# target_link_libraries(FluidsynthMIDIPlayer PRIVATE Fluidsynth)
target_link_libraries(FluidsynthMIDIPlayer PRIVATE ${FLUIDSYNTH_LIBRARIES} -lfluidsynth)

target_link_libraries(${TargetName} EasyMidiFileParserCpp)

# Set include directories
target_include_directories(FluidsynthMIDIPlayer PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    "${ModulesDirectory}/EasyMidiFileParserCpp/Public"
)

target_include_directories(FluidsynthMIDIPlayer PRIVATE ${FluidSynth_INCLUDE_DIRS})

# Install the library
install(TARGETS FluidsynthMIDIPlayer
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    RUNTIME DESTINATION bin
    INCLUDES DESTINATION .
)

add_subdirectory(Standalone)