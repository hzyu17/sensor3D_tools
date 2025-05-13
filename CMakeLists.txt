cmake_minimum_required(VERSION 3.5)
enable_testing()
project(SensorData3D CXX C)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -w -O3")


message("Finding Eigen3 Library")
find_package(Eigen3 3.4.0 REQUIRED)
include_directories(${EIGEN3_INCLUDE_DIRS})

message("Finding pybind11 Library")
find_package(pybind11 REQUIRED)

# Cereal dependency
include(FetchContent)
FetchContent_Declare(
  cereal
  GIT_REPOSITORY https://github.com/USCiLab/cereal.git
  GIT_TAG        v1.3.2            # or whatever version you need
)
FetchContent_MakeAvailable(cereal)


pybind11_add_module(bind_SDF
    src/bind_sdf.cpp)

target_link_libraries(bind_SDF PRIVATE ${EIGEN3_LIBRARIES} cereal)