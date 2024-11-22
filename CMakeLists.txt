cmake_minimum_required(VERSION 3.10)
project(io_uring_sample)

# Set the C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Set debug and optimization flags globally
set(CMAKE_CXX_FLAGS "-g3 -O3")

# Add executables
add_executable(io_uring_enter main.cpp)
add_executable(benchmark benchmark.cpp)

# Find and link liburing
find_library(LIBURING uring REQUIRED)
if (LIBURING)
    target_link_libraries(io_uring_enter PRIVATE ${LIBURING})
    target_link_libraries(benchmark PRIVATE ${LIBURING})
else()
    message(FATAL_ERROR "liburing not found. Please ensure it is installed and accessible.")
endif()

# Add a custom command to print the compile message
add_custom_command(
    OUTPUT show_flags
    COMMAND ${CMAKE_COMMAND} -E echo "Compiling with -g -O3"
    VERBATIM
)

# Add a dependency to ensure the message appears before the builds
add_custom_target(show_flags_target DEPENDS show_flags)

add_dependencies(io_uring_enter show_flags_target)
add_dependencies(benchmark show_flags_target)
