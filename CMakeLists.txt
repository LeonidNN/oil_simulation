cmake_minimum_required(VERSION 3.18)
project(reservoir_core LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(pybind11 CONFIG REQUIRED)

add_library(reservoir_core MODULE cpp/reservoir_core.cpp)
target_link_libraries(reservoir_core PRIVATE pybind11::module)
target_compile_definitions(reservoir_core PRIVATE PYBIND11_DETAILED_ERROR_MESSAGES=0)

find_package(OpenMP)
if(OpenMP_CXX_FOUND)
  target_link_libraries(reservoir_core PRIVATE OpenMP::OpenMP_CXX)
endif()

if (WIN32)
  set_target_properties(reservoir_core PROPERTIES SUFFIX ".pyd")
endif()
