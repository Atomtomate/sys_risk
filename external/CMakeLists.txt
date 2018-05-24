# Obtaining and installing nescessary libraries
include(${CMAKE_ROOT}/Modules/ExternalProject.cmake)

# Prerequisists: check for git
find_package(Git REQUIRED)
if(NOT GIT_FOUND)
    message("git not found!")
endif()

# obtain and build TRNG
ExternalProject_Add(eigen3
        PREFIX ${CMAKE_CURRENT_BINARY_DIR}/external
        GIT_REPOSITORY https://github.com/eigenteam/eigen-git-mirror
        GIT_TAG origin/master
        UPDATE_COMMAND ${GIT_EXECUTABLE} pull -s recursive -X theirs origin master
        PATCH_COMMAND ""
        BUILD_IN_SOURCE 1
        SOURCE_DIR ${CMAKE_SOURCE_DIR}/external/eigen
        CONFIGURE_COMMAND ""
        BUILD_COMMAND ""
        INSTALL_COMMAND ""
        GIT_PROGRESS 1
        )
set(EIGEN_INC ${CMAKE_CURRENT_BINARY_DIR}/external/eigen/Eigen PARENT_SCOPE)

ExternalProject_Add(trng
        PREFIX ${CMAKE_CURRENT_BINARY_DIR}/external
        GIT_REPOSITORY https://github.com/rabauke/trng4
        GIT_TAG origin/master
        UPDATE_COMMAND ${GIT_EXECUTABLE} pull -s recursive -X theirs origin master
        PATCH_COMMAND ""
        BUILD_IN_SOURCE 1
        SOURCE_DIR ${CMAKE_SOURCE_DIR}/external/trng_download
        INSTALL_DIR ${CMAKE_SOURCE_DIR}/external
        CONFIGURE_COMMAND autoreconf -i -f > /dev/null 2>&1 || autoreconf && ./configure CC=${CMAKE_C_COMPILER} CXX=${CMAKE_CXX_COMPILER} "CXXFLAGS=${EXTERNAL_CXX_FLAGS}" "LDFLAGS=${EXTERNAL_LD_FLAGS}" --prefix=<INSTALL_DIR>
        BUILD_COMMAND make
        INSTALL_COMMAND make install && libtool --finish <INSTALL_DIR>/lib
        GIT_PROGRESS 1
        LOG_CONFIGURE 0
        LOG_BUILD 1
        )

set(TRNG_LIB <INSTALL_DIR> PARENT_SCOPE)