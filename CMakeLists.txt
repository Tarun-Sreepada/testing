# CMakeLists.txt

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

# Add external/ouroboros subdirectory
add_subdirectory(external/Ouroboros)

# Create executable, including main.cu
add_executable(${PROJECT_NAME} main.cu ${SRC_FILES})

# Include directories for your project and ouroboros
target_include_directories(${PROJECT_NAME} PRIVATE
    ${PROJECT_SOURCE_DIR}/include
    ${PROJECT_SOURCE_DIR}/external/Ouroboros/include
)

# Add -O3 optimization to both host (CXX) and device (CUDA) code.
# Note we also keep your existing flags:
#   -Xcompiler=-DVERBOSE
#   --expt-relaxed-constexpr
# and simply insert -O3 for both host and device compilation.
target_compile_options(${PROJECT_NAME} PRIVATE
    # Host compiler flags (C++):
    $<$<COMPILE_LANGUAGE:CXX>:-O3>
    # Device compiler flags (CUDA):
    $<$<COMPILE_LANGUAGE:CUDA>:-O3 --expt-relaxed-constexpr -Xcompiler=-O3 -Xcompiler=-DVERBOSE>
)

# Set CUDA properties
set_target_properties(${PROJECT_NAME} PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)
