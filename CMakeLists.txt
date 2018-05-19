cmake_minimum_required (VERSION 3.0)
project (Valuation CXX)


# Compiler settings
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14 -Wextra -Wno-ignored-attributes")

# Project settings
set (Valuation_VERSION_MAJOR 0)
set (Valuation_VERSION_MINOR 1)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/Modules")
#include_directories(${ROOT})
#include_directories("${ROOT}/includes")
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

add_executable(Valuation ${Valuation_SOURCES} src/Sampler.hpp src/Examples.hpp src/Examples.cpp)
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

find_package(GTest)
#find_package(GMock) {GMOCK_INCLUDE_DIRS}
include_directories(${GTEST_INCLUDE_DIRS} $)

# this is the old PRNG library, this could be retired in the future (use R and stan math)
target_link_libraries(Valuation trng4 pthread m)

find_package(MPI REQUIRED)
include_directories(${MPI_INCLUDE_PATH})
target_link_libraries(Valuation ${MPI_LIBRARIES})

find_package(Boost COMPONENTS mpi serialization REQUIRED)
if (Boost_FOUND)
    include_directories(${Boost_INCLUDE_DIR})
    target_link_libraries(Valuation ${Boost_LIBRARIES})
endif()

#SUNDIALS (reqired by stan math)
find_package(SUNDIALS REQUIRED COMPONENTS sundials_idas sundials_cvodes)
set(SUNDIALS_STATIC_LIBRARIES)
if (SUNDIALS_FOUND)
	if (NOT WIN32)
		# On Linux and Mac OS we can look for .a archive files
		foreach(_LIB IN LISTS SUNDIALS_LIBRARIES)
			get_filename_component(_LIBROOT ${_LIB} DIRECTORY)
			get_filename_component(_LIBNAME ${_LIB} NAME_WE)
			find_file(_LIBSTATIC "${_LIBNAME}.a" PATHS ${_LIBROOT} NO_DEFAULT_PATH)
			if (_LIBSTATIC)
				list(APPEND SUNDIALS_STATIC_LIBRARIES ${_LIBSTATIC})
			endif()
			unset(_LIBROOT CACHE)
			unset(_LIBNAME CACHE)
			unset(_LIBSTATIC CACHE)
			unset(_LIB CACHE)
			unset(_LIBROOT)
			unset(_LIBNAME)
			unset(_LIBSTATIC)
			unset(_LIB)
		endforeach()

		# Fall back to dynamic libs if we haven't found static libs
		if (NOT SUNDIALS_STATIC_LIBRARIES)
			set(SUNDIALS_STATIC_LIBRARIES ${SUNDIALS_LIBRARIES})  
		endif()
	else()
		# On Windows we have no way to distinguish .lib files that import dlls or represent static libs
		set(SUNDIALS_STATIC_LIBRARIES ${SUNDIALS_LIBRARIES})  
	endif()
endif()

# Determine SUNDIALS interface version
if (SUNDIALS_FOUND)
	if ("${SUNDIALS_VERSION}" VERSION_LESS "3.0.0")
		add_definitions(-DCADET_SUNDIALS_IFACE_2)
	elseif ("${SUNDIALS_VERSION}" VERSION_LESS "4.0.0")
		add_definitions(-DCADET_SUNDIALS_IFACE_3)
	endif()
endif()

message("Found SUNDIALS: ${SUNDIALS_FOUND}")
if (SUNDIALS_FOUND)
	message("  Version ${SUNDIALS_VERSION}")
	message("  Includes ${SUNDIALS_INCLUDE_DIRS}")
	message("  Libs ${SUNDIALS_LIBRARIES}")
	message("  Static libs ${SUNDIALS_STATIC_LIBRARIES}")
    include_directories(${SUNDIALS_INCLUDE_DIRS})
endif()


find_package (Eigen3 3.3 REQUIRED NO_MODULE)
message("Found Eigen3")

#add libs
include_directories(${CMAKE_SOURCE_DIR}/libs)

target_link_libraries(Valuation ${GTEST_BOTH_LIBRARIES})
target_link_libraries(Valuation ${SUNDIALS_LIBRARIES})
target_link_libraries(Valuation Eigen3::Eigen)
target_link_libraries(Valuation ${RLDFLAGS_l})
target_link_libraries(Valuation ${BLAS_LIBS})
target_link_libraries(Valuation ${LAPACK_LIBS})
target_link_libraries(Valuation ${RINSIDELIBS_l})
target_link_libraries(Valuation ${RCPPLIBS_l})
target_link_libraries(Valuation ${EXTRA_LIBRARIES})
target_link_libraries(Valuation ${LAPACK_LIBRARY})
target_link_libraries(Valuation ${LIBS})


include(${CMAKE_CURRENT_LIST_DIR}/GTests.cmake)
