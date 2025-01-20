#ifndef ALLOCATOR_CUH
#define ALLOCATOR_CUH

#include <cuda_runtime.h>

//------------------------------------------------------------------------------
// 64-bit Bump Allocator structure
//------------------------------------------------------------------------------
struct BumpAllocator {
    char          *pool;      // Pointer to memory pool (allocated in unified memory)
    size_t         poolSize;  // Total size of the pool in bytes.
    unsigned long long offset;  // Current bump pointer offset (64-bit)
};

//------------------------------------------------------------------------------
// Device function: bump allocator allocation (inline definition in header).
// Marking it as static inline ensures that each translation unit gets its own copy.
//------------------------------------------------------------------------------
__device__ static inline void *bump_alloc(BumpAllocator *alloc, unsigned int size) {
    unsigned long long oldVal, newVal;
    while (true) {
        // Read the current allocation offset.
        oldVal = alloc->offset;
        if (oldVal + size > alloc->poolSize) {
            // Not enough room at the end; wrap around.
            // We prepare to allocate from offset 0.
            newVal = size;
            // Attempt to atomically change the offset from oldVal to newVal.
            // Only one thread will succeed in “resetting” the offset.
            if (atomicCAS(&(alloc->offset), oldVal, newVal) == oldVal) {
                // Allocation successful: return pointer from the beginning of the pool.
                return reinterpret_cast<void*>(alloc->pool);
            }
        } else {
            // There is enough room at the end of the buffer.
            newVal = oldVal + size;
            if (atomicCAS(&(alloc->offset), oldVal, newVal) == oldVal) {
                return reinterpret_cast<void*>(alloc->pool + oldVal);
            }
        }
        // If the CAS failed, then some other thread updated the offset.
        // Try again.
    }
}


//------------------------------------------------------------------------------
// Host function: bump allocator allocation (for use on host).
// This version is defined in a source file (or you can also inline it if desired).
//------------------------------------------------------------------------------
#ifdef __CUDACC__
__host__ inline void *bump_alloc_host(BumpAllocator *alloc, size_t size) {
    if (alloc->offset + size > alloc->poolSize)
        return nullptr;
    void *ptr = alloc->pool + alloc->offset;
    alloc->offset += size;
    return ptr;
}
#endif

//------------------------------------------------------------------------------
// Other helper function declarations that can be defined either inline or in a .cu file.
// For example:
#ifdef __CUDACC__
__host__ inline BumpAllocator *createUnifiedBumpAllocator(size_t poolSize) {
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
// This version uses the host-side bump allocation routine.
//------------------------------------------------------------------------------
template <typename T>
T *bump_allocate_and_copy(BumpAllocator *alloc, const T *host_data, size_t count) {
    size_t bytes = count * sizeof(T);
    T *dest = reinterpret_cast<T*>(bump_alloc_host(alloc, bytes));
    if (dest) {
        // Since the memory is allocated in unified memory, we can use memcpy.
        memcpy(dest, host_data, bytes);
    }
    return dest;
}

#endif  // ALLOCATOR_CUH
