cmake_minimum_required (VERSION 3.0)
project (Valuation CXX)

# Compiler settings
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14 -Wextra")

# Project settings
set (Valuation_VERSION_MAJOR 0)
set (Valuation_VERSION_MINOR 1)

#include_directories(${ROOT})
#include_directories("${ROOT}/includes")
include_directories("${PROJECT_SOURCE_DIR}/src")
include_directories("${PROJECT_BINARY_DIR}")

file(GLOB_RECURSE Valuation_SOURCES "src/*.c*")
file(GLOB_RECURSE Valuation_HEADERS "src/*.hpp")

set (Valuation_INCLUDE_DIRS "")
foreach (_headerFile ${Valuation_HEADERS})
    get_filename_component(_dir ${_headerFile} PATH)
    list (APPEND Valuation_INCLUDE_DIRS ${_dir})
endforeach()
list(REMOVE_DUPLICATES Valuation_INCLUDE_DIRS)

configure_file (
    "${PROJECT_SOURCE_DIR}/src/ValuationConfig.h.in"
    "${PROJECT_BINARY_DIR}/ValuationConfig.h"
    )

add_executable(Valuation ${Valuation_SOURCES})
target_include_directories(Valuation PRIVATE ${Valuation_INCLUDE_DIRS})

# Dependencies
find_package(MPI REQUIRED)
include_directories(${MPI_INCLUDE_PATH})
target_link_libraries(Valuation ${MPI_LIBRARIES})

find_package(Boost COMPONENTS mpi serialization REQUIRED)
if (Boost_FOUND)
    include_directories(${Boost_INCLUDE_DIR})
    target_link_libraries(Valuation ${Boost_LIBRARIES})
endif()

find_package (Eigen3 3.3 REQUIRED NO_MODULE)
target_link_libraries (Valuation Eigen3::Eigen)
