cmake_minimum_required (VERSION 3.5)
project (Valuation C CXX)

enable_testing()
option(USE_CLANG "build wqith LLVM/CLANG" OFF)
option(USE_MPI "build with MPI" ON)

# Compiler settings
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED on)
set(THREADS_PREFER_PTHREAD_FLAG ON)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17 -Wextra -O3 -flto -ftree-vectorize -march=native -mfpmath=sse -Wno-unused-parameter") # -g3 -fno-inline
set(PROJECT_EXTERNAL "${CMAKE_SOURCE_DIR}/external")
link_directories(${PROJECT_EXTERNAL}/lib)

#@TODO: load and install pybind11 and check for consistent version
# Project settings
set (Valuation_VERSION_MAJOR 0)
set (Valuation_VERSION_MINOR 2)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/Modules")
include_directories("${PROJECT_SOURCE_DIR}/src")
include_directories("${PROJECT_BINARY_DIR}")

file(GLOB_RECURSE Valuation_SOURCES "src/*.c*")
file(GLOB_RECURSE Valuation_HEADERS "src/*.hpp")
#@TODO: include these for test target
file(GLOB_RECURSE Valuation_test_SOURCES "test/*.c*")
file(GLOB_RECURSE Valuation_test_HEADERS "test/*.hpp")

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
find_package(Threads REQUIRED)

find_package(LAPACK)
if(LAPACK_FOUND)
    target_link_libraries(${PROJECT_NAME} ${LAPACK_LIBS})
    target_link_libraries(${PROJECT_NAME} ${LAPACK_LIBRARY})
else()
    message("LAPACK not found")
endif()

find_package(BLAS)
if(BLAS_FOUND)
    target_link_libraries(${PROJECT_NAME} ${BLAS_LIBS})
else()
    message("BLAS not found")
endif()



#find_package(GTest)
#find_package(GMock) {GMOCK_INCLUDE_DIRS}

if(USE_MPI)
    find_package(MPI REQUIRED)
    include_directories(${MPI_INCLUDE_PATH})
    target_link_libraries(Valuation ${MPI_LIBRARIES})
    add_definitions(-DUSE_MPI)
endif()

if(use_mpi)
    find_package(Boost COMPONENTS system filesystem mpi serialization REQUIRED)
else()
    find_package(Boost COMPONENTS system filesystem serialization REQUIRED)
endif()

if (Boost_FOUND)
    include_directories(${Boost_INCLUDE_DIRS})
    target_link_libraries(Valuation ${Boost_LIBRARIES})
else()
    message("boost not found!")
endif()
message(${Boost_INCLUDE_DIRS})
message(${Boost_LIBRARIES})


#add external
add_subdirectory(${PROJECT_EXTERNAL})
add_dependencies(${PROJECT_NAME} trng)
add_dependencies(${PROJECT_NAME} eigen3)
ExternalProject_Get_Property(gtest source_dir)
include_directories(${PROJECT_EXTERNAL}/include)
add_dependencies(${PROJECT_NAME} gtest)
ExternalProject_Get_Property(gtest binary_dir)
target_link_libraries(${PROJECT_NAME} libgtest trng4 Threads::Threads m ${EXTRA_LIBRARIES} ${LIBS})


# this is the old PRNG library, this could be retired in the future (use R and stan math)
#target_link_libraries(${PROJECT_NAME} ${GTEST_LIBRARIES})
#target_link_libraries(${PROJECT_NAME} ${SUNDIALS_LIBRARIES} ${RLDFLAGS_l} ${RINSIDELIBS_l} ${RCPPLIBS_l})




# ===== PyBind11 Library
# =
# =         This library can be used to export the behavior of Valuation
# =         as a Python module.
# =

find_package(pybind11 REQUIRED)
pybind11_add_module(PyVal MODULE py_interface/PythonInterface.cpp ${Valuation_SOURCES})
set_target_properties(PyVal PROPERTIES OUTPUT_NAME PyVal)
target_include_directories(PyVal PUBLIC ${Valuation_INCLUDE_DIRS} ${PROJECT_EXTERNAL}/include ${Boost_INCLUDE_DIRS})
target_link_libraries(PyVal PUBLIC libgtest trng4 Threads::Threads m ${EXTRA_LIBRARIES} ${Boost_LIBRARIES} ${LIBS}) #
