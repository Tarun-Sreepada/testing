#pragma once
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdint.h>
#include <string.h>

//------------------------------------------------------------------------------
// A header stored at the beginning of every allocated block.
//------------------------------------------------------------------------------
struct AllocationHeader {
    size_t pagesUsed; // Number of pages allocated for this block.
    size_t startPage; // Starting page index in the pool.
};

struct CudaMemoryManager {
    void* pool;         // Base pointer to the memory pool.
    size_t numPages;    // Total number of pages in the pool.
    size_t pageSize;    // Size (in bytes) of each page.
    int lock;           // (Optional) Spinlock (not used in this simple version).
    int pagesAllocated; // Bump pointer: number of pages already allocated.

     __device__ void* malloc(size_t bytes) {
        // Total bytes = user bytes + header.
        size_t totalBytes = bytes + sizeof(AllocationHeader);
        // Compute the number of pages needed (round up).
        size_t pagesNeeded = (totalBytes + pageSize - 1) / pageSize;

        int oldPages, newPages;
        // Atomically reserve pages using a CAS loop.
        do {
            oldPages = pagesAllocated;
            // Check for available space.
            if (oldPages + pagesNeeded > numPages)
                return nullptr;
            newPages = oldPages + pagesNeeded;
            // Attempt to update pagesAllocated from oldPages to newPages.
        } while (atomicCAS(&pagesAllocated, oldPages, newPages) != oldPages);

        // Compute the starting address of the allocation.
        size_t allocStart = oldPages;
        char* blockPtr = (char*)pool + allocStart * pageSize;

        // Write the allocation header.
        AllocationHeader* header = (AllocationHeader*)blockPtr;
        header->pagesUsed = pagesNeeded;
        header->startPage = allocStart;

        // Return pointer to user memory (after the header).
        return blockPtr + sizeof(AllocationHeader);
    }

    //--------------------------------------------------------------------------
    // Device–side free function.
    //
    // For this bump allocator, free does nothing.
    //--------------------------------------------------------------------------
    __device__ void free(void* /*ptr*/) {
        // No-op: individual frees are not supported in a bump allocator.
    }
};

//------------------------------------------------------------------------------
// Host–side helper: createCudaMemoryManager
//
// Allocates and initializes a CudaMemoryManager (and its pool) in managed memory.
//------------------------------------------------------------------------------
__host__ CudaMemoryManager* createCudaMemoryManager(size_t numPages, size_t pageSize) {
    cudaError_t err;
    CudaMemoryManager* mm = nullptr;
    err = cudaMallocManaged(&mm, sizeof(CudaMemoryManager));
    if (err != cudaSuccess || mm == nullptr) {
        fprintf(stderr, "Failed to allocate CudaMemoryManager structure.\n");
        return nullptr;
    }
    mm->numPages = numPages;
    mm->pageSize = pageSize;
    mm->lock = 0;
    mm->pagesAllocated = 0;

    size_t totalBytes = numPages * pageSize;
    err = cudaMallocManaged(&mm->pool, totalBytes);
    if (err != cudaSuccess || mm->pool == nullptr) {
        fprintf(stderr, "Failed to allocate memory pool (%zu bytes).\n", totalBytes);
        cudaFree(mm);
        return nullptr;
    }
    cudaMemset(mm->pool, 0, totalBytes);
    return mm;
}
