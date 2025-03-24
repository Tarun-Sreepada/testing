#pragma once
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <iostream>

// #include <gallatin/allocators/global_allocator.cuh>

namespace gallatin
{
    namespace allocators
    {

        // Use __device__ __managed__ so the variables are accessible from both host and device.
        __device__ __managed__ char *allocator_base = nullptr;
        __device__ __managed__ unsigned long long allocator_offset = 0;
        __device__ __managed__ unsigned long long allocator_total = 0;

        // Initializes the global allocator with the requested total memory.
        // The second parameter is unused (dummy) but provided to match the given function signature.
        void init_global_allocator(size_t total_memory, int /*dummy*/)
        {
            allocator_total = total_memory;
            cudaError_t err = cudaMalloc(&allocator_base, total_memory);
            if (err != cudaSuccess)
            {
                std::fprintf(stderr, "cudaMalloc failed: %s\n", cudaGetErrorString(err));
                std::exit(EXIT_FAILURE);
            }
            allocator_offset = 0;
        }

        // Frees the entire allocated memory.
        void free_global_allocator()
        {
            if (allocator_base)
            {
                cudaFree(allocator_base);
                allocator_base = nullptr;
            }
            allocator_offset = 0;
            allocator_total = 0;
        }

        // Returns a pointer to a block of memory of the requested size from the global pool.
        // Alignment defaults to 8 bytes. On failure, returns nullptr.
        __device__ void *global_malloc(unsigned long long size)
        {

            // Round up to the next multiple of 8
            size = (size + 7) & ~7;

            unsigned long long ret = atomicAdd(&allocator_offset, size);
            return allocator_base + ret;
        }

        // Bump allocators typically do not support freeing individual allocations.
        // This function is provided to match the API but is a no-op.
        __device__ void global_free(void * /*ptr*/)
        {
            // No-op: individual allocations cannot be freed in a bump allocator.
        }

    } // namespace allocators
} // namespace gallatin

using namespace gallatin::allocators;
