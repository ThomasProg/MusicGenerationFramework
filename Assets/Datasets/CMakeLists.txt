cmake_minimum_required(VERSION 3.10)

include(FetchContent)

function(download_dataset datasetName url)

project("${datasetName}")

message(STATUS "Downloading ${datasetName} Dataset...")
FetchContent_Declare("${datasetName}"
    URL ${url}
    DOWNLOAD_NO_EXTRACT false
    TLS_VERIFY true
    SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/${datasetName}"
    DOWNLOAD_EXTRACT_TIMESTAMP true
)

FetchContent_Populate("${datasetName}")

endfunction(download_dataset)

# https://magenta.tensorflow.org/datasets/maestro
download_dataset(
    Maestro
    https://storage.googleapis.com/magentadata/datasets/maestro/v3.0.0/maestro-v3.0.0-midi.zip
)

# https://magenta.tensorflow.org/datasets/e-gmd
download_dataset(
    ExpandedGrooveMIDI
    https://storage.googleapis.com/magentadata/datasets/e-gmd/v1.0.0/e-gmd-v1.0.0-midi.zip
)

# https://colinraffel.com/projects/lmd/
download_dataset(
    LakhMidi-full
    http://hog.ee.columbia.edu/craffel/lmd/lmd_full.tar.gz
)
