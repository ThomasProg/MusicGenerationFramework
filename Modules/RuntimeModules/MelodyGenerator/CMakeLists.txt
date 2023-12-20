cmake_minimum_required(VERSION 3.10)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_BUILD_TYPE debug)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++23")

FILE(GLOB SOURCES ./*)
set(TargetName "MelodyGenerator")
project(${TargetName})

# Create library
add_library(${TargetName} SHARED ${SOURCES})


target_link_libraries(${TargetName} EasyMidiFileParserCpp)

target_include_directories(${TargetName} PUBLIC 
    "${PROJECT_SOURCE_DIR}/Public/" 
    "${ModulesDirectory}/EasyMidiFileParserCpp/Public"
)

add_custom_command(TARGET ${TargetName}
  POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E copy_directory 
  ${EasyMidiFileParserCpp_BINARY_DIR}/$<CONFIG>/.
  ${MelodyGenerator_BINARY_DIR}/$<CONFIG>/.
)
