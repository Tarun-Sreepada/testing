#ifndef ALLOCATOR_CUH
#define ALLOCATOR_CUH

#include <cuda_runtime.h>
#include <iostream>
#include <cstring> // for memcpy

//------------------------------------------------------------------------------
// 64-bit Bump Allocator structure using unsigned long long for poolSize.
//------------------------------------------------------------------------------
struct BumpAllocator {
    char                  *pool;      // Pointer to memory pool (allocated in unified memory)
    unsigned long long     poolSize;  // Total size of the pool in bytes (64-bit unsigned)
    unsigned long long     offset;    // Current bump pointer offset (64-bit)
};

uint64_t next_pow2(uint64_t x) {
	return x == 1 ? 1 : 1<<(64-__builtin_clzl(x-1));
}

__device__ uint64_t g_next_pow2(uint64_t x) {
    return x == 1 ? 1 : 1<<(64-__clzll(x-1));
}

//------------------------------------------------------------------------------
// Device function: bump allocator allocation (inline definition in header).
// Marking it as static inline ensures that each translation unit gets its own copy.
//------------------------------------------------------------------------------
__device__ static inline char *bump_alloc(BumpAllocator *alloc, unsigned long long size) {
    unsigned long long oldVal, newVal;
    // round up size to nearest power of 2
    size = (size + 7) & ~7;

    while (true) {
        // Read the current allocation offset.
        oldVal = alloc->offset;
        if (oldVal + size > alloc->poolSize) {
            // Not enough room at the end; wrap around.
            // We prepare to allocate from offset 0.
            newVal = size;
            // Attempt to atomically change the offset from oldVal to newVal.
            if (atomicCAS(&(alloc->offset), oldVal, newVal) == oldVal) {
                // Allocation successful: return pointer from the beginning of the pool.
                return (alloc->pool);
            }
        } else {
            // There is enough room at the end of the buffer.
            newVal = oldVal + size;
            if (atomicCAS(&(alloc->offset), oldVal, newVal) == oldVal) {
                return (alloc->pool + oldVal);
            }
        }
        // If the CAS failed, some other thread updated the offset.
        // Try again.
    }
}

//------------------------------------------------------------------------------
// Host function: bump allocator allocation (for use on host).
//------------------------------------------------------------------------------
#ifdef __CUDACC__
__host__ inline void *bump_alloc_host(BumpAllocator *alloc, ssize_t size) {
    // Check if there is enough room in the pool
    if (alloc->offset + size > alloc->poolSize){
        std::cout << "Bump allocator out of memory || size: " << size 
                  << " offset: " << alloc->offset 
                  << " poolSize: " << alloc->poolSize << std::endl;
        return nullptr;
    }
    void *ptr = alloc->pool + alloc->offset;
    alloc->offset += size;
    return ptr;
}
#endif

//------------------------------------------------------------------------------
// Host helper functions to create and free a unified bump allocator.
//------------------------------------------------------------------------------
#ifdef __CUDACC__
__host__ inline BumpAllocator *createUnifiedBumpAllocator(unsigned long long poolSize) {
    BumpAllocator *alloc = nullptr;
    // Allocate the allocator structure in unified memory.
    cudaMallocManaged(&alloc, sizeof(BumpAllocator));
    // Allocate the pool in unified memory so both host and device can access it.
    cudaMallocManaged(&(alloc->pool), poolSize);
    alloc->poolSize = poolSize;
    alloc->offset   = 0;
    return alloc;
}

__host__ inline void freeUnifiedBumpAllocator(BumpAllocator *alloc) {
    if (!alloc) return;
    cudaFree(alloc->pool);
    cudaFree(alloc);
}
#endif

//------------------------------------------------------------------------------
// Template helper: Allocate memory from the bump allocator's pool and copy host data into it.
//------------------------------------------------------------------------------
template <typename T>
T *bump_allocate_and_copy(BumpAllocator *alloc, const T *host_data, size_t count) {
    size_t bytes = count * sizeof(T);
    T *dest = reinterpret_cast<T*>(bump_alloc_host(alloc, bytes));
    if (dest) {
        // Since the memory is allocated in unified memory, we can safely use memcpy.
        memcpy(dest, host_data, bytes);
    }
    return dest;
}

#endif  // ALLOCATOR_CUH
