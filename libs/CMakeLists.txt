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
        GIT_REPOSITORY "https://github.com/rabauke/trng4"
        GIT_TAG "master"
        UPDATE_COMMAND ""
        PATCH_COMMAND ""
        SOURCE_DIR "${CMAKE_SOURCE_DIR}/libs/trng_download"
        INSTALL_DIR "${CMAKE_SOURCE_DIR}/libs/trng"
        CONFIGURE_COMMAND autoreconf -f -i  > /dev/null 2>&1 && <SOURCE_DIR>/configure --prefix=<INSTALL_DIR>
        BUILD_COMMAND "make"
        INSTALL_COMMAND make install
        GIT_PROGRESS 1
        LOG_CONFIGURE 1
        LOG_BUILD 1
        #CMAKE_ARGS -DCMAKE_INSTALL_PREFIX=${CMAKE_SOURCE_DIR}/libs/trng
        )
#find_path(TRNG_INCLUDE_DIR trng/trng.a)
#if((NOT TRNG_INCLUDE_DIR) OR (NOT EXISTS ${TRNG_INCLUDE_DIR}))
#    message("-- TRNG not found")
#    execute_process(COMMAND git submodule update --init -- trng WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})
#endif()

set(TRNG_LIB ${TRNG_LIB} PARENT_SCOPE)