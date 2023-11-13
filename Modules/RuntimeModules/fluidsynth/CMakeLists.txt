cmake_minimum_required(VERSION 3.10)

project("fluidsynth")

include(FetchContent)

function(download_file url)

FetchContent_Declare(fluidsynth
    URL ${url}
    DOWNLOAD_NO_EXTRACT false
    TLS_VERIFY true
    SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/Downloads/"
    DOWNLOAD_EXTRACT_TIMESTAMP true
)

FetchContent_Populate(fluidsynth)

endfunction(download_file)

message(STATUS "Downloading Fluidsynth Binaries...")
download_file(
    https://github.com/FluidSynth/fluidsynth/releases/download/v2.3.4/fluidsynth-2.3.4-win10-x64.zip
)
 