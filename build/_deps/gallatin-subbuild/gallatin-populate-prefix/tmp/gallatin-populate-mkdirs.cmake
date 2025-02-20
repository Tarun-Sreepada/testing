# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file Copyright.txt or https://cmake.org/licensing for details.

cmake_minimum_required(VERSION 3.5)

file(MAKE_DIRECTORY
  "/home/tarun/testing/build/_deps/gallatin-src"
  "/home/tarun/testing/build/_deps/gallatin-build"
  "/home/tarun/testing/build/_deps/gallatin-subbuild/gallatin-populate-prefix"
  "/home/tarun/testing/build/_deps/gallatin-subbuild/gallatin-populate-prefix/tmp"
  "/home/tarun/testing/build/_deps/gallatin-subbuild/gallatin-populate-prefix/src/gallatin-populate-stamp"
  "/home/tarun/testing/build/_deps/gallatin-subbuild/gallatin-populate-prefix/src"
  "/home/tarun/testing/build/_deps/gallatin-subbuild/gallatin-populate-prefix/src/gallatin-populate-stamp"
)

set(configSubDirs )
foreach(subDir IN LISTS configSubDirs)
    file(MAKE_DIRECTORY "/home/tarun/testing/build/_deps/gallatin-subbuild/gallatin-populate-prefix/src/gallatin-populate-stamp/${subDir}")
endforeach()
if(cfgdir)
  file(MAKE_DIRECTORY "/home/tarun/testing/build/_deps/gallatin-subbuild/gallatin-populate-prefix/src/gallatin-populate-stamp${cfgdir}") # cfgdir has leading slash
endif()
