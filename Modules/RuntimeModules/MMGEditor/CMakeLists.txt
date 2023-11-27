cmake_minimum_required(VERSION 3.10)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED True)
# add_compile_definitions(GLAD_GLAPI_EXPORT)

# FILE(GLOB SOURCES Src/* Include/*)
FILE(GLOB SOURCES *.cpp *.h)
set(TargetName "MMGEditor")
project(${TargetName})

# Create dynamic lib (.dll/.so)
# add_library(${TargetName} SHARED ${SOURCES})
add_executable(${TargetName} ${SOURCES})

# Add EasyMidiFileParserCpp
target_link_libraries(${TargetName} EasyMidiFileParserCpp)
target_link_libraries(${TargetName} PlayMIDI)

target_link_libraries(${TargetName} glad)
target_link_libraries(${TargetName} glfw)
target_link_libraries(${TargetName} imgui)

# Add fluidsynth
set(enable-pkgconfig false) 
set(BUILD_SHARED_LIBS ON)

target_link_libraries(${TargetName} "${ModulesDirectory}/fluidsynth/Downloads/lib/libfluidsynth.dll.a")
target_compile_definitions(${TargetName} PRIVATE MMGEditor_EXPORT)

target_include_directories(${TargetName} PUBLIC 
    "${PROJECT_SOURCE_DIR}/Public/" 
    "${ModulesDirectory}/fluidsynth/Downloads/include"
    "${ModulesDirectory}/EasyMidiFileParserCpp/Public"
    "${ModulesDirectory}/PlayMIDI"
    "${ModulesDirectory}/glfw/include"
    "${ModulesDirectory}/glfw/deps"
    "${ModulesDirectory}/imgui/imgui"
)

add_custom_command(TARGET ${TargetName}
  POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E copy_directory 
  "${ModulesDirectory}/fluidsynth/Downloads/bin/" 
  ${MMGEditor_BINARY_DIR}/$<CONFIG>/.
)

add_custom_command(TARGET ${TargetName}
  POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E copy_directory 
  ${EasyMidiFileParserCpp_BINARY_DIR}/$<CONFIG>/.
  ${MMGEditor_BINARY_DIR}/$<CONFIG>/.
)

add_custom_command(TARGET ${TargetName}
  POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E copy_directory 
  ${PlayMIDI_BINARY_DIR}/$<CONFIG>/.
  ${MMGEditor_BINARY_DIR}/$<CONFIG>/.
)

add_custom_command(TARGET ${TargetName}
  POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E copy_directory 
  ${glad_BINARY_DIR}/$<CONFIG>/.
  ${MMGEditor_BINARY_DIR}/$<CONFIG>/.
)

add_custom_command(TARGET ${TargetName}
  POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E copy_directory 
  ${GLFW_BINARY_DIR}/src/$<CONFIG>/.
  ${MMGEditor_BINARY_DIR}/$<CONFIG>/.
)

add_custom_command(TARGET ${TargetName}
  POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E copy_directory 
  ${imgui_BINARY_DIR}/$<CONFIG>/.
  ${MMGEditor_BINARY_DIR}/$<CONFIG>/.
)