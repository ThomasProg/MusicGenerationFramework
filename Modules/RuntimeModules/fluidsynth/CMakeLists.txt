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

# === example
# download_file(
#   https://upload.wikimedia.org/wikipedia/commons/9/91/Octicons-mark-github.svg
#   12794390cce7d0682ffc783c785e4282305684431b30b29ed75c224da24035b4
# )

download_file(
    https://github.com/FluidSynth/fluidsynth/releases/download/v2.3.4/fluidsynth-2.3.4-win10-x64.zip
)
 