cmake_minimum_required(VERSION 3.10)

project("SFML")

include(FetchContent)

function(download_file url)

FetchContent_Declare(SFML
    URL ${url}
    DOWNLOAD_NO_EXTRACT false
    TLS_VERIFY true
    SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/Downloads/"
    DOWNLOAD_EXTRACT_TIMESTAMP true
)

FetchContent_Populate(SFML)

endfunction(download_file)

message(STATUS "Downloading SFML Binaries...")
download_file(
    https://www.sfml-dev.org/files/SFML-2.6.1-windows-vc17-64-bit.zip
)
 