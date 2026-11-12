cmake_minimum_required(VERSION 3.24)
project(perception_perf_lab LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE RelWithDebInfo CACHE STRING "" FORCE)
endif()

set(PPL_SANITIZE "" CACHE STRING "Sanitizer to enable: address, thread, or empty")
if(PPL_SANITIZE)
  add_compile_options(-fsanitize=${PPL_SANITIZE})
  add_link_options(-fsanitize=${PPL_SANITIZE})
endif()

add_compile_options(-Wall -Wextra -Wpedantic -fno-omit-frame-pointer)

find_package(Threads REQUIRED)
find_package(GTest REQUIRED)
find_package(benchmark REQUIRED)

add_library(ppl src/stages.cpp src/latency_recorder.cpp)
target_include_directories(ppl PUBLIC include)
target_link_libraries(ppl PUBLIC Threads::Threads)

add_executable(ppl_run src/main.cpp)
target_link_libraries(ppl_run PRIVATE ppl)

enable_testing()
add_executable(ppl_tests tests/test_latency_recorder.cpp)
target_link_libraries(ppl_tests PRIVATE ppl GTest::gtest_main)
include(GoogleTest)
gtest_discover_tests(ppl_tests)

add_executable(ppl_bench bench/bench_stages.cpp)
target_link_libraries(ppl_bench PRIVATE ppl benchmark::benchmark_main)