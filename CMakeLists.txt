cmake_minimum_required(VERSION 3.10)

option(DOWNLOAD_ASSETS "Download assets automatically" ON)
option(BUILD_MODULES "Build modules automatically" ON)
option(DOWNLOAD_SUBMODULES "Download submodules automatically" ON) # [TODO]
option(DOWNLOAD_MODULE_DEPENDENCIES "Download module dependencies automatically" ON)

if(BUILD_MODULES)
    add_subdirectory(Modules)
endif()

# if(DOWNLOAD_ASSETS)
#     add_subdirectory(Assets)
# endif()
