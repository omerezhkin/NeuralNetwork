include(FetchContent)

FetchContent_Declare(
  Catch2
  GIT_REPOSITORY https://github.com/catchorg/Catch2.git
  GIT_TAG v3.5.4
  GIT_SHALLOW TRUE
)
FetchContent_MakeAvailable(Catch2)

add_executable(nn_tests
    core_tests.cpp
    linear_tests.cpp
    activation_tests.cpp
    loss_tests.cpp
    optimizer_tests.cpp
    network_trainer_tests.cpp
)

target_link_libraries(nn_tests PRIVATE nn::nn Catch2::Catch2WithMain)

if(MSVC)
  target_compile_options(nn_tests PRIVATE /W4 /permissive-)
else()
  target_compile_options(nn_tests PRIVATE -Wall -Wextra -Wpedantic)
endif()

include(Catch)
catch_discover_tests(nn_tests)
