cmake_minimum_required(VERSION 3.20)

project(nn_coursework LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(NN_BUILD_EXAMPLE "Build MNIST training example" ON)
option(NN_BUILD_TESTING "Build unit tests" ON)

include(FetchContent)

set(EIGEN_BUILD_TESTING OFF CACHE BOOL "" FORCE)
set(EIGEN_BUILD_DOC OFF CACHE BOOL "" FORCE)
set(BUILD_TESTING OFF CACHE BOOL "" FORCE)

FetchContent_Declare(
  Eigen
  GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
  GIT_TAG 3.4.0
  GIT_SHALLOW TRUE
)
FetchContent_MakeAvailable(Eigen)

add_library(nn INTERFACE)
add_library(nn::nn ALIAS nn)

target_compile_features(nn INTERFACE cxx_std_20)
target_include_directories(nn INTERFACE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>)
target_link_libraries(nn INTERFACE Eigen3::Eigen)

if(MSVC)
  target_compile_options(nn INTERFACE /W4 /permissive-)
else()
  target_compile_options(nn INTERFACE -Wall -Wextra -Wpedantic)
endif()

if(NN_BUILD_EXAMPLE)
  add_executable(nn_example app/main.cpp)
  target_link_libraries(nn_example PRIVATE nn::nn)
  if(MSVC)
    target_compile_options(nn_example PRIVATE /W4 /permissive-)
  else()
    target_compile_options(nn_example PRIVATE -Wall -Wextra -Wpedantic)
  endif()
endif()

if(NN_BUILD_TESTING)
  include(CTest)
  enable_testing()
  add_subdirectory(tests)
endif()
