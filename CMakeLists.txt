cmake_minimum_required(VERSION 3.16)

project(K_SpecPart_C VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(Eigen3 REQUIRED)
find_package(Threads REQUIRED)

add_executable(K_SpecPart
    src/cut_distillation.cpp
    src/embedding.cpp
    src/external_tools.cpp
    src/golden_evaluator.cpp
    src/graphification.cpp
    src/hypergraph.cpp
    src/io.cpp
    src/isolate_islands.cpp
    src/main.cpp
    src/metis.cpp
    src/optimal_partitioner.cpp
    src/overlay.cpp
    src/projection.cpp
    src/specpart.cpp
    src/lsst.cpp
    src/triton_refiner.cpp
    src/tree_partition.cpp
)

target_include_directories(K_SpecPart
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_link_libraries(K_SpecPart
    PRIVATE
        Eigen3::Eigen
        Threads::Threads
)

target_compile_options(K_SpecPart
    PRIVATE
        -Wall
        -Wextra
        -Wpedantic
)
