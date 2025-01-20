// unified_bump_allocator.cu

#include <cstdio>
#include <cstring>    // for memcpy
#include <cuda_runtime.h>

//------------------------------------------------------------------------------
// 64-bit Bump Allocator structure
//------------------------------------------------------------------------------
// We use cudaMallocManaged so that the host can read/update the offset and pool.
struct BumpAllocator {
    char          *pool;      // Pointer to memory pool (allocated in unified memory)
    size_t         poolSize;  // Total size of the pool in bytes.
    unsigned long long offset;  // Current bump pointer offset (64-bit)
};

//------------------------------------------------------------------------------
// Device function: bump allocator allocation (for use on GPU).
// Uses atomicAdd so that multiple threads can allocate concurrently.
// Returns a pointer to a memory chunk of 'size' bytes from the pool, or nullptr
// if there is not enough room.
//------------------------------------------------------------------------------
__device__ inline void *bump_alloc(BumpAllocator *alloc, unsigned int size) {
    unsigned long long old = atomicAdd(&(alloc->offset), static_cast<unsigned long long>(size));
    if (old + size > alloc->poolSize) {
        // Not enough room.
        return nullptr;
    }
    return reinterpret_cast<void*>(alloc->pool + old);
}

//------------------------------------------------------------------------------
// Host function: bump allocator allocation (for use on host).
// This simple version assumes that no concurrent host or device allocations
// are occurring at the same time (i.e. serial access).
//------------------------------------------------------------------------------
void *bump_alloc_host(BumpAllocator *alloc, size_t size) {
    if (alloc->offset + size > alloc->poolSize)
        return nullptr;
    void *ptr = alloc->pool + alloc->offset;
    alloc->offset += size;
    return ptr;
}

//------------------------------------------------------------------------------
// Kernel that uses the bump allocator (device-side allocation).
// Each thread allocates space for one integer from the pool, writes its index into it,
// and then copies that value to an output array. If allocation fails, -1 is stored.
//------------------------------------------------------------------------------
__global__ void kernel_using_allocator(BumpAllocator *alloc, int *out, int N) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < N) {
        // Allocate space for one int.
        int *local_ptr = reinterpret_cast<int*>(bump_alloc(alloc, sizeof(int)));
        if (local_ptr != nullptr) {
            *local_ptr = idx;
            out[idx] = *local_ptr;
        } else {
            out[idx] = -1;
        }
    }
}

//------------------------------------------------------------------------------
// Host helper: Create a unified bump allocator.
// Allocates the BumpAllocator structure and its pool in unified (managed) memory.
//------------------------------------------------------------------------------
BumpAllocator *createUnifiedBumpAllocator(size_t poolSize) {
    BumpAllocator *alloc = nullptr;
    // Allocate the allocator structure in managed memory.
    cudaMallocManaged(&alloc, sizeof(BumpAllocator));
    // Allocate the pool in managed memory so both host and device can see it.
    cudaMallocManaged(&(alloc->pool), poolSize);
    alloc->poolSize = poolSize;
    alloc->offset   = 0;
    return alloc;
}

//------------------------------------------------------------------------------
// Host helper: Free the bump allocator and its associated pool.
void freeUnifiedBumpAllocator(BumpAllocator *alloc) {
    if (!alloc) return;
    cudaFree(alloc->pool);
    cudaFree(alloc);
}

//------------------------------------------------------------------------------
// Template helper: Allocate memory from the bump allocator's pool and copy host data into it.
// This version uses the host-side bump allocation routine (i.e. serial allocation).
template <typename T>
T *bump_allocate_and_copy(BumpAllocator *alloc, const T *host_data, size_t count) {
    size_t bytes = count * sizeof(T);
    T *dest = reinterpret_cast<T*>(bump_alloc_host(alloc, bytes));
    if (dest) {
        // Copy from host buffer to the allocated region.
        // (If your pool were device-only memory you would need to use cudaMemcpy,
        //  but because we allocated the pool in managed memory we can use memcpy directly.)
        memcpy(dest, host_data, bytes);
    }
    return dest;
}

//------------------------------------------------------------------------------
// Main: Demonstrates both device-side bump allocation and host-side bump allocation+copy.
int main() {
    //-------------------------------------------------------------------------
    // Example 1: Use bump allocator from device code.
    //-------------------------------------------------------------------------
    const int N = 256;
    int *device_out = nullptr;
    // Allocate output array in unified memory.
    cudaMallocManaged(&device_out, N * sizeof(int));

    // Create a unified bump allocator with a 1 MB pool.
    size_t poolSize = 1 << 20;  // 1 MB
    BumpAllocator *alloc = createUnifiedBumpAllocator(poolSize);

    // Launch a kernel that uses the bump allocator.
    int threads = 64;
    int blocks  = (N + threads - 1) / threads;
    kernel_using_allocator<<<blocks, threads>>>(alloc, device_out, N);
    cudaDeviceSynchronize();

    // Print results from device-side allocation.
    printf("Kernel results (using bump_alloc on device):\n");
    for (int i = 0; i < N; i++) {
        if (device_out[i] == -1)
            printf("Allocation failed at index %d\n", i);
        else
            printf("device_out[%d] = %d\n", i, device_out[i]);
    }

    //-------------------------------------------------------------------------
    // Example 2: Use bump allocator to copy host data into the preallocated pool.
    //-------------------------------------------------------------------------
    int hostArray[5] = {10, 20, 30, 40, 50};
    // Allocate and copy data into the bump allocator's pool.
    int *bumpCopiedArray = bump_allocate_and_copy(alloc, hostArray, 5);
    if (bumpCopiedArray == nullptr) {
        printf("Failed to allocate and copy host data into bump pool.\n");
    } else {
        printf("\nData copied into bump allocator pool:\n");
        for (int i = 0; i < 5; i++) {
            printf("%d ", bumpCopiedArray[i]);
        }
        printf("\n");
    }

    // Clean up.
    cudaFree(device_out);
    freeUnifiedBumpAllocator(alloc);
    cudaDeviceReset();
    return 0;
}
