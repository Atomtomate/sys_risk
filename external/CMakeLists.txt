# adapted from https://crascit.com/2015/07/25/cmake-gtest/

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
set(EIGEN_INC ${CMAKE_SOURCE_DIR}/external/eigen PARENT_SCOPE)
include_directories(${CMAKE_SOURCE_DIR}/external/eigen)

ExternalProject_Add(trng
        PREFIX ${CMAKE_CURRENT_BINARY_DIR}/external
        GIT_REPOSITORY https://github.com/rabauke/trng4
        GIT_TAG origin/master
        GIT_PROGRESS 1
        LOG_CONFIGURE 0
        LOG_BUILD 1
        UPDATE_COMMAND ${GIT_EXECUTABLE} pull -s recursive -X theirs origin master
        PATCH_COMMAND ""
        BUILD_IN_SOURCE 1
        SOURCE_DIR ${CMAKE_SOURCE_DIR}/external/trng_download
        INSTALL_DIR ${CMAKE_SOURCE_DIR}/external
        CONFIGURE_COMMAND autoreconf -i -f > /dev/null 2>&1 || autoreconf && ./configure CC=${CMAKE_C_COMPILER} CXX=${CMAKE_CXX_COMPILER} "CXXFLAGS=${EXTERNAL_CXX_FLAGS}" "LDFLAGS=${EXTERNAL_LD_FLAGS}" --prefix=<INSTALL_DIR>
        BUILD_COMMAND make
        INSTALL_COMMAND make install && "libtool --finish ${CMAKE_SOURCE_DIR}/external/lib"
        )

set(TRNG_LIB INSTALL_DIR ${CMAKE_SOURCE_DIR}/external PARENT_SCOPE)

ExternalProject_Add(gtest
        PREFIX ${CMAKE_CURRENT_BINARY_DIR}
        GIT_REPOSITORY https://github.com/google/googletest.git
        GIT_TAG release-1.8.0
        CMAKE_ARGS -DCMAKE_INSTALL_PREFIX=${CMAKE_SOURCE_DIR}/external
                   -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}
                   -Dgtest_force_shared_crt=ON
        INSTALL_COMMAND ""
        LOG_DOWNLOAD ON
        LOG_CONFIGURE ON
        LOG_BUILD ON
        )

add_library(libgtest IMPORTED STATIC GLOBAL)
add_dependencies(libgtest gtest)
set_target_properties(libgtest PROPERTIES
        "IMPORTED_LOCATION" "${CMAKE_SOURCE_DIR}/external/lib/libgtest.a"
        "IMPORTED_LINK_INTERFACE_LIBRARIES" "${CMAKE_THREAD_LIBS_INIT}"
       )