cmake_minimum_required(VERSION 3.10)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# FILE(GLOB SOURCES Src/* Include/*)
FILE(GLOB SOURCES *.cpp *.h)
set(TargetName "MelodyExtractor")
project(${TargetName})

# Create dynamic lib (.dll/.so)
# add_library(${TargetName} SHARED ${SOURCES})
add_executable(${TargetName} ${SOURCES})

target_include_directories(${TargetName} PUBLIC 
    "${PROJECT_SOURCE_DIR}/Public/"
    "${ModulesDirectory}/EasyMidiFileParserCpp/Public"
)

target_link_libraries(${TargetName} EasyMidiFileParserCpp)

add_custom_command(TARGET ${TargetName}
  POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E copy_directory 
  ${EasyMidiFileParserCpp_BINARY_DIR}/$<CONFIG>/.
  ${MelodyExtractor_BINARY_DIR}/$<CONFIG>/.
)