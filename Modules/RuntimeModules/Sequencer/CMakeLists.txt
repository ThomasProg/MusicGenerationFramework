cmake_minimum_required(VERSION 3.10)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# FILE(GLOB SOURCES Src/* Include/*)
FILE(GLOB SOURCES *.cpp *.h)
set(TargetName "Sequencer")
project(${TargetName})

# Create dynamic lib (.dll/.so)
add_library(${TargetName} SHARED ${SOURCES})

target_include_directories(${TargetName} PUBLIC 
    "${PROJECT_SOURCE_DIR}/Public/" 
)
