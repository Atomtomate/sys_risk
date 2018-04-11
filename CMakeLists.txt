cmake_minimum_required (VERSION 3.0)
project (Valuation CXX)

# Compiler settings
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14 -Wextra -Wno-ignored-attributes")

# Project settings
set (Valuation_VERSION_MAJOR 0)
set (Valuation_VERSION_MINOR 1)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/Modules")
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

# RInside stuff
execute_process(COMMAND R RHOME OUTPUT_VARIABLE R_HOME)
set(NUM_TRUNC_CHARS 2)
set(RPATH "R")
set(RSCRIPT_PATH "Rscript")
set(RCPPFLAGS_CMD " ${RPATH} " " CMD " " config " " --cppflags ")

execute_process(COMMAND ${RPATH} CMD config --cppflags OUTPUT_VARIABLE RCPPFLAGS)
string(SUBSTRING ${RCPPFLAGS} ${NUM_TRUNC_CHARS} -1 RCPPFLAGS)
include_directories(${RCPPFLAGS})

execute_process(COMMAND R CMD config --ldflags OUTPUT_VARIABLE RLDFLAGS)
string(LENGTH ${RLDFLAGS} RLDFLAGS_LEN)
if (${RLDFLAGS} MATCHES "[-][L]([^ ;])+")
    string(SUBSTRING ${CMAKE_MATCH_0} ${NUM_TRUNC_CHARS} -1 RLDFLAGS_L)
    string(STRIP ${RLDFLAGS_L} RLDFLAGS_L )
    link_directories(${RLDFLAGS_L} )
endif()
if (${RLDFLAGS} MATCHES "[-][l]([^;])+")
    string(SUBSTRING ${CMAKE_MATCH_0} ${NUM_TRUNC_CHARS} -1 RLDFLAGS_l)
    string(STRIP ${RLDFLAGS_l} RLDFLAGS_l )
endif()

execute_process(COMMAND Rscript -e "Rcpp:::CxxFlags()" OUTPUT_VARIABLE RCPPINCL)
string(SUBSTRING ${RCPPINCL} ${NUM_TRUNC_CHARS} -1 RCPPINCL)
include_directories(${RCPPINCL})

execute_process(COMMAND Rscript -e "Rcpp:::LdFlags()" OUTPUT_VARIABLE RCPPLIBS)

execute_process(COMMAND Rscript -e "RInside:::CxxFlags()" OUTPUT_VARIABLE RINSIDEINCL)
string(SUBSTRING ${RINSIDEINCL} ${NUM_TRUNC_CHARS} -1 RINSIDEINCL)
include_directories(${RINSIDEINCL})

execute_process(COMMAND Rscript -e "RInside:::LdFlags()" OUTPUT_VARIABLE RINSIDELIBS)

execute_process(COMMAND Rscript -e "cat(paste('-I', system.file('include', package = 'RcppEigen'), sep=''))" OUTPUT_VARIABLE RCPPEIGENINCL)

string(SUBSTRING ${RCPPEIGENINCL} ${NUM_TRUNC_CHARS} -1 RCPPEIGENINCL)
include_directories(${RCPPEIGENINCL})
if (${RCPPLIBS} MATCHES "[-][L]([^ ;])+")
    string(SUBSTRING ${CMAKE_MATCH_0} ${NUM_TRUNC_CHARS} -1 RCPPLIBS_L)
    link_directories(${RCPPLIBS_L} )
endif()

if (${RCPPLIBS} MATCHES "[-][l][R]([^;])+")
    string(SUBSTRING ${CMAKE_MATCH_0} ${NUM_TRUNC_CHARS} -1 RCPPLIBS_l)
endif()

if (${RINSIDELIBS} MATCHES "[-][L]([^ ;])+")
    string(SUBSTRING ${CMAKE_MATCH_0} ${NUM_TRUNC_CHARS} -1 RINSIDELIBS_L)
    link_directories(${RINSIDELIBS_L})
endif()

if (${RINSIDELIBS} MATCHES "[-][l][R]([^;])+")
    string(SUBSTRING ${CMAKE_MATCH_0} ${NUM_TRUNC_CHARS} -1 RINSIDELIBS_l)
endif()

# If the faster 'gold' linker is used, to avoid complaints about undefined symbol
SET(CMAKE_FIND_LIBRARY_SUFFIXES_SAVED ${CMAKE_FIND_LIBRARY_SUFFIXES}) # Backup
LIST(APPEND CMAKE_FIND_LIBRARY_SUFFIXES ".so.3")
FIND_LIBRARY(BLAS_LIBRARY blas)
SET(CMAKE_FIND_LIBRARY_SUFFIXES ${CMAKE_FIND_LIBRARY_SUFFIXES_SAVED}) # Restore
LIST(APPEND EXTRA_LIBRARIES ${BLAS_LIBRARY})

execute_process(COMMAND R CMD config CXXFLAGS OUTPUT_VARIABLE RCXXFLAGS)

execute_process(COMMAND R CMD config BLAS_LIBS OUTPUT_VARIABLE RBLAS)

execute_process(COMMAND R CMD config LAPACK_LIBS OUTPUT_VARIABLE RLAPACK)

FIND_LIBRARY(LAPACK_LIBRARY lapack)

# Dependencies
target_link_libraries(Valuation "trng4")

find_package(MPI REQUIRED)
include_directories(${MPI_INCLUDE_PATH})
target_link_libraries(Valuation ${MPI_LIBRARIES})

find_package(Boost COMPONENTS mpi serialization REQUIRED)
if (Boost_FOUND)
    include_directories(${Boost_INCLUDE_DIR})
    target_link_libraries(Valuation ${Boost_LIBRARIES})
endif()

find_package (Eigen3 3.3 REQUIRED NO_MODULE)
message("Found Eigen3")
target_link_libraries (Valuation Eigen3::Eigen)
target_link_libraries(Valuation ${RLDFLAGS_l})
target_link_libraries(Valuation ${BLAS_LIBS})
target_link_libraries(Valuation ${LAPACK_LIBS})
target_link_libraries(Valuation ${RINSIDELIBS_l})
target_link_libraries(Valuation ${RCPPLIBS_l})
target_link_libraries(Valuation ${EXTRA_LIBRARIES})
target_link_libraries(Valuation ${LAPACK_LIBRARY})
target_link_libraries(Valuation ${LIBS})
