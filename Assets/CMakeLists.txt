cmake_minimum_required(VERSION 3.10)

option(DOWNLOAD_DATASETS "Download datasets automatically" ON)
option(DOWNLOAD_SOUNDFONTS "Download soundfonts automatically" ON)
option(DOWNLOAD_MODELS "Download models automatically" ON)

if(DOWNLOAD_DATASETS)
    add_subdirectory(Datasets)
endif()

if(DOWNLOAD_SOUNDFONTS)
    add_subdirectory(Soundfonts)
endif()

if(DOWNLOAD_MODELS)
    add_subdirectory(Models)
endif()

