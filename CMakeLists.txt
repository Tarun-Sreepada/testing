cmake_minimum_required(VERSION 3.18 FATAL_ERROR)

project(cuEFIM
    VERSION 1.0
    LANGUAGES CXX CUDA
)

# Set C++ and CUDA Standards
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Specify CUDA Architecture
set(CMAKE_CUDA_ARCHITECTURES 75)

# Automatically collect all .cpp and .cu files in src/
file(GLOB SRC_FILES "src/*.cpp" "src/*.cu")

# Create executable, including main.cu
add_executable(${PROJECT_NAME} main.cu ${SRC_FILES})

# Include directories for your project and ouroboros
target_include_directories(${PROJECT_NAME} PRIVATE
    ${PROJECT_SOURCE_DIR}/include
    # ${PROJECT_SOURCE_DIR}/external/Ouroboros/include
)

# Add debugging flags for compute-sanitizer
target_compile_options(${PROJECT_NAME} PRIVATE
    # Host compiler flags (C++):
    $<$<COMPILE_LANGUAGE:CXX>:-O3 -g>  # Disable optimizations and add debug symbols
    # Device compiler flags (CUDA):
    $<$<COMPILE_LANGUAGE:CUDA>:-O3 -G -g --expt-relaxed-constexpr -Xcompiler=-g -Xcompiler=-DVERBOSE>
)

# Set CUDA properties
set_target_properties(${PROJECT_NAME} PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

# Ensure debugging info is available
set_target_properties(${PROJECT_NAME} PROPERTIES
    CUDA_DEBUG_INFO ON
)

