cmake_minimum_required(VERSION 3.10)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

FILE(GLOB SOURCES Src/* Include/*)
set(TargetName "ModularMusicGenerationTests")
project(${TargetName})

# # Add ModularMusicGenerationCore
# add_subdirectory(ThirdParty/ModularMusicGenerationCore)

# Create library
add_executable(${TargetName} ${SOURCES})
target_link_libraries(${TargetName} ModularMusicGenerationCore)
# target_include_directories(${TargetName} PUBLIC 
#     "${PROJECT_SOURCE_DIR}/Include/" 
#     "${PROJECT_SOURCE_DIR}/ThirdParty/ModularMusicGenerationCore/Public"
# )
target_include_directories(${TargetName} PUBLIC 
    "${PROJECT_SOURCE_DIR}/Include/" 
    "${ModulesDirectory}/ModularMusicGenerationCore/Public"
)

# add_custom_command(TARGET ${TargetName}
#   POST_BUILD
#   COMMAND ${CMAKE_COMMAND} -E copy_directory 
#   ${ModularMusicGenerationCore_BINARY_DIR}/$<CONFIG>/.
#   ${ModularMusicGenerationTests_BINARY_DIR}/$<CONFIG>/.
# )
add_custom_command(TARGET ${TargetName}
  POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E copy_directory 
  ${ModularMusicGenerationCore_BINARY_DIR}/$<CONFIG>/.
  ${ModularMusicGenerationTests_BINARY_DIR}/$<CONFIG>/.
)
