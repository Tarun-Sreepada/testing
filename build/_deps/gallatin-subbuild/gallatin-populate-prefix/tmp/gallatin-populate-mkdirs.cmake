# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file Copyright.txt or https://cmake.org/licensing for details.

cmake_minimum_required(VERSION 3.5)

# If CMAKE_DISABLE_SOURCE_CHANGES is set to true and the source directory is an
# existing directory in our source tree, calling file(MAKE_DIRECTORY) on it
# would cause a fatal error, even though it would be a no-op.
if(NOT EXISTS "/export/home1/ltarun/testing/build/_deps/gallatin-src")
  file(MAKE_DIRECTORY "/export/home1/ltarun/testing/build/_deps/gallatin-src")
endif()
file(MAKE_DIRECTORY
  "/export/home1/ltarun/testing/build/_deps/gallatin-build"
  "/export/home1/ltarun/testing/build/_deps/gallatin-subbuild/gallatin-populate-prefix"
  "/export/home1/ltarun/testing/build/_deps/gallatin-subbuild/gallatin-populate-prefix/tmp"
  "/export/home1/ltarun/testing/build/_deps/gallatin-subbuild/gallatin-populate-prefix/src/gallatin-populate-stamp"
  "/export/home1/ltarun/testing/build/_deps/gallatin-subbuild/gallatin-populate-prefix/src"
  "/export/home1/ltarun/testing/build/_deps/gallatin-subbuild/gallatin-populate-prefix/src/gallatin-populate-stamp"
)

set(configSubDirs )
foreach(subDir IN LISTS configSubDirs)
    file(MAKE_DIRECTORY "/export/home1/ltarun/testing/build/_deps/gallatin-subbuild/gallatin-populate-prefix/src/gallatin-populate-stamp/${subDir}")
endforeach()
if(cfgdir)
  file(MAKE_DIRECTORY "/export/home1/ltarun/testing/build/_deps/gallatin-subbuild/gallatin-populate-prefix/src/gallatin-populate-stamp${cfgdir}") # cfgdir has leading slash
endif()
