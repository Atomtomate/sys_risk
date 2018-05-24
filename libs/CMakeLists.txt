# Obtaining and installing nescessary libraries
include(${CMAKE_ROOT}/Modules/ExternalProject.cmake)

# Prerequisists: check for git
find_package(Git REQUIRED)
if(NOT GIT_FOUND)
    message("git not found!")
endif()

# obtain and build TRNG
#TOD: find acutal .a file
ExternalProject_Add(trng
        PREFIX ${CMAKE_CURRENT_BINARY_DIR}/libs
        GIT_REPOSITORY https://github.com/rabauke/trng4
        GIT_TAG origin/master
        UPDATE_COMMAND ${GIT_EXECUTABLE} pull -s recursive -X theirs origin master
        PATCH_COMMAND ""
        BUILD_IN_SOURCE 1
        SOURCE_DIR ${CMAKE_SOURCE_DIR}/libs/trng_download
        INSTALL_DIR ${CMAKE_SOURCE_DIR}/libs/trng
        CONFIGURE_COMMAND autoreconf -i -f > /dev/null 2>&1 || autoreconf && ./configure CC=${CMAKE_C_COMPILER} CXX=${CMAKE_CXX_COMPILER} "CXXFLAGS=${EXTERNAL_CXX_FLAGS}" "LDFLAGS=${EXTERNAL_LD_FLAGS}" --prefix=${CMAKE_CURRENT_BINARY_DIR}
#        CONFIGURE_COMMAND ./configure --prefix=<INSTALL_DIR>
        BUILD_COMMAND make
        INSTALL_COMMAND make install
        GIT_PROGRESS 1
        LOG_CONFIGURE 0
        LOG_BUILD 1
        #CMAKE_ARGS -DCMAKE_INSTALL_PREFIX=${CMAKE_SOURCE_DIR}/libs/trng
        )
#find_path(TRNG_INCLUDE_DIR trng/trng.a)
#if((NOT TRNG_INCLUDE_DIR) OR (NOT EXISTS ${TRNG_INCLUDE_DIR}))
#    message("-- TRNG not found")
#    execute_process(COMMAND git submodule update --init -- trng WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})
#endif()

set(TRNG_LIB ${TRNG_LIB} PARENT_SCOPE)