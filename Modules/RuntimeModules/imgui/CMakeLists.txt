cmake_minimum_required(VERSION 3.10)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# add_compile_definitions(GLAD_GLAPI_EXPORT)
# add_compile_definitions(IMGUI_API=__declspec\(dllexport\))

# FILE(GLOB SOURCES Src/* Include/*)
FILE(GLOB SOURCES imgui/*.cpp imgui/*.h 
    imgui/backends/imgui_impl_glfw.cpp imgui/backends/imgui_impl_glfw.h 
    imgui/backends/imgui_impl_opengl3.cpp imgui/backends/imgui_impl_opengl3.h)
set(TargetName "imgui")
project(${TargetName})

# Create dynamic lib (.dll/.so)
add_library(${TargetName} STATIC ${SOURCES})

target_link_libraries(${TargetName} glfw)

add_custom_command(TARGET ${TargetName}
  POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E copy_directory 
  ${GLFW_BINARY_DIR}/src/$<CONFIG>/.
  ${imgui_BINARY_DIR}/$<CONFIG>/.
)

target_include_directories(${TargetName} PUBLIC 
    "${ModulesDirectory}/imgui/imgui"
    "${ModulesDirectory}/glfw/include"
)
