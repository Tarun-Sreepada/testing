cmake_minimum_required(VERSION 3.10)
project(cuEFIM LANGUAGES CXX CUDA)

# set(CMAKE_CXX_STANDARD 12)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
# compile with O3 optimization
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")

# Collect the source files.
set(SOURCES
    src/main.cu
    src/io.cpp
    src/mine.cu
)

# Create the executable target.
add_executable(cuEFIM ${SOURCES})

# Enable separable compilation for dynamic parallelism (if needed).
set_target_properties(cuEFIM PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
