#ifndef ALLOCATOR_CUH
#define ALLOCATOR_CUH

#include <cuda_runtime.h>
#include <iostream>
#include <cstring> // for memcpy
#include <iostream>
#include <string>
#include <vector>
#include <cstring>
#include <stdint.h>
#include <cuda_runtime.h>
#include <chrono>
#include <unordered_map>
#include <map>
#include <fstream>
#include <algorithm>

//------------------------------------------------------------------------------
// 64-bit Bump Allocator structure using unsigned long long for poolSize.
//------------------------------------------------------------------------------
struct BumpAllocator {
    void                  *pool;      // Pointer to memory pool (allocated in unified memory)
    unsigned long long     poolSize;  // Total size of the pool in bytes (64-bit unsigned)
    unsigned long long     offset;    // Current bump pointer offset (64-bit)
};

//------------------------------------------------------------------------------
// Device function: bump allocator allocation (inline definition in header).
// Marking it as static inline ensures that each translation unit gets its own copy.
//------------------------------------------------------------------------------
__device__ static inline void *bump_alloc(BumpAllocator *alloc, unsigned long long size) {
    unsigned long long oldVal, newVal;

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


struct Item
{
    uint32_t id;
    uint32_t utility;
};

// __device__ void mine_kernel_d(BumpAllocator *alloc, uint32_t *pattern, Item *items, uint32_t nItems,
//                               uint32_t *start, uint32_t *end, uint32_t *utility, uint32_t nTransactions,
//                               uint32_t *primary, uint32_t numPrimary, uint32_t maxItem,
//                               uint32_t minUtil, uint32_t *high_utility_patterns)


struct WorkItem {
    uint32_t *pattern;
    Item *items;
    uint32_t num_items;
    uint32_t *start;
    uint32_t *end;
    uint32_t *utility;
    uint32_t num_transactions;
    uint32_t *primary;
    uint32_t num_primary;
    uint32_t max_item;
};



template<typename T, int CAPACITY>
struct AtomicWorkQueue {
    T items[CAPACITY];
    unsigned int head;
    unsigned int tail;
    uint32_t work_count;

    // Initialize the queue
    __host__ __device__ void init() {
        head = 0;
        tail = 0;
        work_count = 0;
    }

    // Atomic read of work count
    __device__ uint32_t get_work_count() {
        return atomicAdd(&work_count, 0);  // Atomic read
    }

    __device__ bool enqueue(const T &item) {
        unsigned int curTail, curHead, nextTail;
        bool success = false;
        
        do {
            curTail = atomicAdd(&tail, 0);
            curHead = atomicAdd(&head, 0);
            nextTail = (curTail + 1) % CAPACITY;

            if (nextTail == curHead) return false;  // Full
            
            // Attempt to claim slot
            if (atomicCAS(&tail, curTail, nextTail) == curTail) {
                items[curTail] = item;
                atomicAdd(&work_count, 1);  // Increment ONLY after successful insert
                __threadfence();  // Ensure memory consistency
                success = true;
                break;
            }
        } while (true);

        return success;
    }

    __device__ bool dequeue(T &out) {
        unsigned int curHead, curTail, nextHead;
        bool success = false;

        do {
            curHead = atomicAdd(&head, 0);
            curTail = atomicAdd(&tail, 0);

            if (curHead == curTail) return false;  // Empty
            
            nextHead = (curHead + 1) % CAPACITY;
            
            // Attempt to claim item
            if (atomicCAS(&head, curHead, nextHead) == curHead) {
                out = items[curHead];
                __threadfence();  // Ensure memory consistency
                success = true;
                break;
            }
        } while (true);

        return success;
    }

    // Host-side enqueue (keep original implementation)
    __host__ bool host_enqueue(const T &item) {
        unsigned int curTail = tail;
        unsigned int nextTail = (curTail + 1) % CAPACITY;
        
        if (nextTail == head) return false;
        
        items[curTail] = item;
        tail = nextTail;
        work_count++;  // No atomics needed for host-side
        return true;
    }
};




#endif  // ALLOCATOR_CUH
