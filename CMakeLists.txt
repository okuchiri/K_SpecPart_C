cmake_minimum_required(VERSION 3.16)

project(K_SpecPart_C VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(Eigen3 REQUIRED)

add_executable(K_SpecPart
    src/embedding.cpp
    src/golden_evaluator.cpp
    src/graphification.cpp
    src/hypergraph.cpp
    src/io.cpp
    src/isolate_islands.cpp
    src/main.cpp
    src/overlay.cpp
    src/specpart.cpp
)

target_include_directories(K_SpecPart
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_link_libraries(K_SpecPart
    PRIVATE
        Eigen3::Eigen
)

target_compile_options(K_SpecPart
    PRIVATE
        -Wall
        -Wextra
        -Wpedantic
)
