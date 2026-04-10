cmake_minimum_required(VERSION 3.20)

project(nn_coursework LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

include(FetchContent)

set(EIGEN_BUILD_TESTING OFF CACHE BOOL "" FORCE)
set(EIGEN_BUILD_DOC OFF CACHE BOOL "" FORCE)

FetchContent_Declare(
  Eigen
  GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
  GIT_TAG 3.4.0
  GIT_SHALLOW TRUE
)
FetchContent_MakeAvailable(Eigen)

set(NN_SOURCES
    app/main.cpp
    core/random/random.cpp
    nn/layers/linear.cpp
    nn/layers/sigmoid.cpp
    nn/loss/mse.cpp
    nn/model/network.cpp
    nn/model/network_builder.cpp
    nn/optim/sgd.cpp
    train/train.cpp
)

add_executable(coursework ${NN_SOURCES})

target_include_directories(coursework PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

target_link_libraries(coursework PRIVATE Eigen3::Eigen)

if(MSVC)
  target_compile_options(coursework PRIVATE /W4 /permissive-)
else()
  target_compile_options(coursework PRIVATE -Wall -Wextra -Wpedantic)
endif()
