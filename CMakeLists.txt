cmake_minimum_required (VERSION 3.5)
project (Valuation CXX)

enable_testing()

# Compiler settings
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED on)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14 -Wextra")
set(PROJECT_EXTERNAL "${CMAKE_SOURCE_DIR}/external")
link_directories(${PROJECT_EXTERNAL}/lib)

# Project settings
set (Valuation_VERSION_MAJOR 0)
set (Valuation_VERSION_MINOR 2)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/Modules")
include_directories("${PROJECT_SOURCE_DIR}/src")
include_directories("${PROJECT_BINARY_DIR}")

#include(FindOpenMP)
#if(OPENMP_FOUND)
#	set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
#	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
#	set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
#else(OPENMP_FOUND)
#	message("ERROR: OpenMP could not be found.")
#endif(OPENMP_FOUND)

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

add_executable(Valuation ${Valuation_SOURCES} src/Sampler.hpp src/N2_network.hpp src/N2_network.cpp src/ER_Network.hpp src/ER_Network.cpp src/IOHelper.hpp test/TestStanMath.cpp src/EigenDenseBaseAddons.hpp)
target_include_directories(Valuation PRIVATE ${Valuation_INCLUDE_DIRS})


# Dependencies

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
#include_directories(${GTEST_INCLUDE_DIRS} $)

find_package(MPI REQUIRED)
include_directories(${MPI_INCLUDE_PATH})
target_link_libraries(Valuation ${MPI_LIBRARIES})

find_package(Boost COMPONENTS system filesystem mpi serialization REQUIRED)
if (Boost_FOUND)
    include_directories(${Boost_INCLUDE_DIR})
    target_link_libraries(Valuation ${Boost_LIBRARIES})
else()
    message("Boost not found!")
endif()

# Workaround for travis-ci since FindEigen3 is not prevent on ubuntu 14.04
#find_package(PkgConfig)
#pkg_search_module(Eigen3 REQUIRED eigen3)
#if(NOT EIGEN3_FOUND)
#    # Fallback to cmake_modules
#    message("pkg_search unsuccessful. falling back to find_package")
#    find_package(Eigen3 REQUIRED)
#    set(EIGEN3_INCLUDE_DIRS ${EIGEN_INCLUDE_DIRS})
#    set(EIGEN3_LIBRARIES ${EIGEN_LIBRARIES})
#    target_link_libraries (${PROJECT_NAME} Eigen3::Eigen)
#else()
#    include_directories(${EIGEN3_INCLUDE_DIR})
#    include_directories(${EIGEN3_INCLUDE_DIRS})
#    add_definitions(${EIGEN3_DEFINITIONS})
#    set(EIGEN3_INCLUDE_DIRS ${EIGEN3_INCLUDE_DIR})
#endif()


#add external
add_subdirectory(${PROJECT_EXTERNAL})
add_dependencies(${PROJECT_NAME} trng)
add_dependencies(${PROJECT_NAME} eigen3)
include_directories(${EIGEN_INC})
ExternalProject_Get_Property(gtest source_dir)
target_include_directories(${PROJECT_NAME} PRIVATE ${PROJECT_EXTERNAL}/include)
add_dependencies(${PROJECT_NAME} gtest)
ExternalProject_Get_Property(gtest binary_dir)
target_link_libraries(${PROJECT_NAME} libgtest trng4)


# this is the old PRNG library, this could be retired in the future (use R and stan math)
target_link_libraries(${PROJECT_NAME} pthread m)
#target_link_libraries(${PROJECT_NAME} ${GTEST_LIBRARIES})
target_link_libraries(${PROJECT_NAME} ${EXTRA_LIBRARIES})
target_link_libraries(${PROJECT_NAME} ${LIBS})
#target_link_libraries(${PROJECT_NAME} Eigen) #Eigen3::Eigen
#target_link_libraries(${PROJECT_NAME} ${SUNDIALS_LIBRARIES})
#target_link_libraries(${PROJECT_NAME} ${RLDFLAGS_l})
#target_link_libraries(${PROJECT_NAME} ${RINSIDELIBS_l})
#target_link_libraries(${PROJECT_NAME} ${RCPPLIBS_l})


#include(${CMAKE_CURRENT_LIST_DIR}/GTests.cmake)
