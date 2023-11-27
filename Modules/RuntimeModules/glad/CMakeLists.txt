cmake_minimum_required(VERSION 3.10)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# FILE(GLOB SOURCES Src/* Include/*)
FILE(GLOB SOURCES src/*.c include/*.h)
set(TargetName "glad")
project(${TargetName})

# Create dynamic lib (.dll/.so)
add_library(${TargetName} STATIC ${SOURCES})

target_include_directories(${TargetName} PUBLIC 
    "${ModulesDirectory}/glad/include"
)
