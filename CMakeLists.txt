cmake_minimum_required(VERSION 3.10)

project("MusicGenerationFramework")

option(DOWNLOAD_ASSETS "Download assets automatically" ON)
option(BUILD_MODULES "Build modules automatically" ON)
option(DOWNLOAD_SUBMODULES "Download submodules automatically" ON) # [TODO]
option(DOWNLOAD_MODULE_DEPENDENCIES "Download module dependencies automatically" ON)

option(OUT_MUSICS_PATH "The folder in which the musics will be generated" ${CMAKE_SOURCE_DIR}/Out/Musics)
# option(DATASETS_PATH "The folder containing the datasets" "${CMAKE_SOURCE_DIR}/Assets/Datasets")
add_compile_definitions(OUT_MUSICS_PATH)
add_compile_definitions(DATASETS_PATH="${CMAKE_SOURCE_DIR}/Assets/Datasets/")
add_compile_definitions(ANALYSIS_MODULES_PATH="${CMAKE_SOURCE_DIR}/Modules/DataAnalysisModules/")

if(BUILD_MODULES)
    add_subdirectory(Modules)
endif()

if(DOWNLOAD_ASSETS)
    add_subdirectory(Assets)
endif()
