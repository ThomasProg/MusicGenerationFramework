cmake_minimum_required(VERSION 3.10)

project("OnnxRuntime")

include(FetchContent)

function(download_file url)

FetchContent_Declare(OnnxRuntime
    URL ${url}
    DOWNLOAD_NO_EXTRACT false
    TLS_VERIFY true
    SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/Downloads/"
    DOWNLOAD_EXTRACT_TIMESTAMP true
)

FetchContent_Populate(OnnxRuntime)

endfunction(download_file)

message(STATUS "Downloading OnnxRuntime Binaries...")
download_file(
    https://www.nuget.org/api/v2/package/Microsoft.ML.OnnxRuntime.Gpu/1.16.1
)
 