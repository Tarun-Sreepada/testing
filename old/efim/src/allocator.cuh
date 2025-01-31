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


#pragma once

#include <cuda_runtime.h>
#include <iostream>
#include <cstring>  // for memcpy

//------------------------------------------------------------------------------
// Page-based Allocator structure
//------------------------------------------------------------------------------
struct PageAllocator {
    void   *pool;       // Pointer to the unified memory pool
    size_t  poolSize;   // Total size of the pool in bytes
    size_t  pageSize;   // Size of each page in bytes (e.g., 64 KB)
    size_t  pageCount;  // Number of pages in the pool

    bool   *pageUsed;   // Boolean array [pageCount], true = page is used, false = page is free

    int    *lock;       // Lock for synchronization (device-side only)
};

//------------------------------------------------------------------------------
// createUnifiedPageAllocator:
//   - Allocates a large memory pool and the allocator metadata in unified memory.
//   - Initializes all pages to "free".
//------------------------------------------------------------------------------
inline PageAllocator *createUnifiedPageAllocator(size_t poolSize, size_t pageSize = 64 * 1024 /*64KB*/)
{
    // Create the allocator struct in unified memory
    PageAllocator *alloc = nullptr;
    cudaMallocManaged(&alloc, sizeof(PageAllocator));

    // Allocate the pool in unified memory
    cudaMallocManaged(&(alloc->pool), poolSize);

    alloc->poolSize  = poolSize;
    alloc->pageSize  = pageSize;
    alloc->pageCount = poolSize / pageSize; // Assume perfect division for simplicity

    // Allocate the page usage array (pageCount) in unified memory
    cudaMallocManaged(&(alloc->pageUsed), alloc->pageCount * sizeof(bool));

    // Initialize all pages to free (false)
    for (size_t i = 0; i < alloc->pageCount; i++) {
        alloc->pageUsed[i] = false;
    }

    // Allocate and initialize the device-side lock
    cudaMallocManaged(&(alloc->lock), sizeof(int));
    *(alloc->lock) = 0; // Initialize to unlocked state

    return alloc;
}

//------------------------------------------------------------------------------
// freeUnifiedPageAllocator:
//   - Frees the pool and the allocator structure itself.
//------------------------------------------------------------------------------
inline void freeUnifiedPageAllocator(PageAllocator *alloc)
{
    if (!alloc) return;
    cudaFree(alloc->pageUsed);
    cudaFree(alloc->pool);
    cudaFree(alloc);
}

//------------------------------------------------------------------------------
// page_alloc_host:
//   - Allocates `size` bytes from the allocator by searching for a range of
//     consecutive free pages large enough to hold `size`.
//   - Returns a pointer if success; returns nullptr if not enough contiguous space.
//------------------------------------------------------------------------------
inline void *page_alloc_host(PageAllocator *alloc, size_t size)
{
    if (!alloc || size == 0) return nullptr;

    // Number of pages required for this allocation
    const size_t pagesNeeded = (size + alloc->pageSize - 1) / alloc->pageSize;

    // Simple linear search for `pagesNeeded` consecutive free pages
    size_t consecutiveCount = 0;
    size_t startIndex       = 0;

    for (size_t i = 0; i < alloc->pageCount; i++) {
        if (!alloc->pageUsed[i]) {
            // This page is free
            if (consecutiveCount == 0) {
                // Potential start of a new free region
                startIndex = i;
            }
            consecutiveCount++;
        } else {
            // Page is used; reset the counter
            consecutiveCount = 0;
        }

        // Check if we found enough pages
        if (consecutiveCount == pagesNeeded) {
            // Mark these pages as used
            for (size_t j = 0; j < pagesNeeded; j++) {
                alloc->pageUsed[startIndex + j] = true;
            }
            // Return the pointer
            char *basePtr = static_cast<char*>(alloc->pool);
            return basePtr + (startIndex * alloc->pageSize);
        }
    }

    // If we get here, not enough contiguous space was found
    std::cerr << "PageAllocator out of memory or no contiguous block found. "
              << "Requested: " << size << " bytes\n";
    return nullptr;
}

//------------------------------------------------------------------------------
// page_free_host:
//   - Frees a previously allocated block of size `size` starting at `ptr`.
//   - Calculates which pages correspond to this range and marks them as free.
//------------------------------------------------------------------------------
inline void page_free_host(PageAllocator *alloc, void *ptr, size_t size)
{
    if (!alloc || !ptr || size == 0) return;

    // Figure out which page index 'ptr' corresponds to
    uintptr_t base   = reinterpret_cast<uintptr_t>(alloc->pool);
    uintptr_t target = reinterpret_cast<uintptr_t>(ptr);

    if (target < base || (target - base) >= alloc->poolSize) {
        // Pointer is out of range for this allocator
        std::cerr << "Attempt to free memory outside this allocator's pool.\n";
        return;
    }

    // Page index where ptr resides
    size_t pageIndex = (target - base) / alloc->pageSize;

    // How many pages does `size` occupy?
    size_t pagesNeeded = (size + alloc->pageSize - 1) / alloc->pageSize;

    // Mark pages as free
    for (size_t i = 0; i < pagesNeeded; i++) {
        if (pageIndex + i < alloc->pageCount) {
            alloc->pageUsed[pageIndex + i] = false;
        }
    }
}

//------------------------------------------------------------------------------
// page_allocate_and_copy:
//   - Allocates enough pages to store `count * sizeof(T)` bytes.
//   - Copies data from host_data to the newly allocated region (memcpy).
//------------------------------------------------------------------------------
template <typename T>
T *page_allocate_and_copy(PageAllocator *alloc, const T *host_data, size_t count)
{
    if (!alloc || !host_data || count == 0) return nullptr;

    size_t bytes = count * sizeof(T);
    T *dest = reinterpret_cast<T*>(page_alloc_host(alloc, bytes));
    if (dest) {
        // Since the memory is in unified memory, we can directly use memcpy.
        memcpy(dest, host_data, bytes);
    }
    return dest;
}



//------------------------------------------------------------------------------
// Device-side lock/unlock
//------------------------------------------------------------------------------
__device__ inline void page_lock_device(PageAllocator *alloc)
{
    // Spin until lock == 0, then set lock = 1
    while (atomicCAS(alloc->lock, 0, 1) != 0) {
        // busy-wait
    }
}

__device__ inline void page_unlock_device(PageAllocator *alloc)
{
    // Release the lock
    atomicExch(alloc->lock, 0);
}

//------------------------------------------------------------------------------
// page_alloc_device:
//   - Device-side version of page_alloc_host.
//   - Uses a naive spin-lock around a linear search.
//------------------------------------------------------------------------------
__device__ inline void *page_alloc_device(PageAllocator *alloc, size_t size)
{
    if (!alloc || size == 0) return nullptr;

    // Number of pages required
    size_t pagesNeeded = (size + alloc->pageSize - 1) / alloc->pageSize;

    // Acquire lock
    page_lock_device(alloc);

    size_t consecutiveCount = 0;
    size_t startIndex       = 0;
    bool   success          = false;

    // Linear search for free pages
    for (size_t i = 0; i < alloc->pageCount; i++) {
        if (!alloc->pageUsed[i]) {
            // This page is free
            if (consecutiveCount == 0) {
                // potential start
                startIndex = i;
            }
            consecutiveCount++;
        } else {
            consecutiveCount = 0;
        }

        if (consecutiveCount == pagesNeeded) {
            // Mark used
            for (size_t j = 0; j < pagesNeeded; j++) {
                alloc->pageUsed[startIndex + j] = true;
            }
            success = true;
            break;
        }
    }

    // If success, compute pointer
    void *result = nullptr;
    if (success) {
        char *basePtr = static_cast<char*>(alloc->pool);
        result = basePtr + (startIndex * alloc->pageSize);
    }

    page_unlock_device(alloc);

    // If we get here and success == false => out of memory
    return result;
}

//------------------------------------------------------------------------------
// page_free_device:
//   - Device-side version of page_free_host.
//   - Locks, calculates page index, marks them free.
//------------------------------------------------------------------------------
__device__ inline void page_free_device(PageAllocator *alloc, void *ptr, size_t size)
{
    if (!alloc || !ptr || size == 0) return;

    uintptr_t base   = reinterpret_cast<uintptr_t>(alloc->pool);
    uintptr_t target = reinterpret_cast<uintptr_t>(ptr);

    // Check pointer is within the pool
    if (target < base || (target - base) >= alloc->poolSize) {
        // We can't do printf in device code without special setup, but let's just bail.
        return;
    }

    size_t pageIndex    = (target - base) / alloc->pageSize;
    size_t pagesToFree  = (size + alloc->pageSize - 1) / alloc->pageSize;

    page_lock_device(alloc);

    for (size_t i = 0; i < pagesToFree; i++) {
        if ((pageIndex + i) < alloc->pageCount) {
            alloc->pageUsed[pageIndex + i] = false;
        }
    }

    page_unlock_device(alloc);
}




struct Item
{
    uint32_t id;
    uint32_t utility;
};


struct WorkItem {
    uint32_t *pattern;
    Item *items;
    uint32_t num_items;
    uint32_t *start;
    uint32_t *end;
    uint32_t *utility;
    uint32_t num_transactions;
    uint32_t primary_item;
    uint32_t max_item;

    void *base_ptr;
    uint64_t bytes = 0;
    uint32_t *work_done;
    uint32_t work_count;
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
