# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.28

# compile CUDA with /usr/local/cuda-12.6/bin/nvcc
CUDA_DEFINES = -DTEST_CHUNKS

CUDA_INCLUDES = --options-file CMakeFiles/main_c.dir/includes_CUDA.rsp

CUDA_FLAGS =  -gencode=arch=compute_70,code=sm_70  -Xcompiler -D_FORCE_INLINES -DVERBOSE --expt-extended-lambda -use_fast_math --expt-relaxed-constexpr -keep --ptxas-options=-v -lineinfo -std=c++17 "--generate-code=arch=compute_86,code=[compute_86,sm_86]"

