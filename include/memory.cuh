#pragma once
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdint.h>

#define BITS_PER_WORD 32


struct AllocationHeader
{
    size_t pagesUsed;
    size_t startPage;
};


struct CudaMemoryManager
{
    void *pool;           // Base pointer to the GPU memory pool
    size_t numPages;      // Total number of pages
    size_t pageSize;      // Size of each page in bytes
    unsigned int *bitset; // Bitset array: 1 => page allocated, 0 => free
    size_t bitsetLength;  // Number of 32-bit words in 'bitset'
    size_t nextFreePage;  // Next page index to start scanning from
    int lock;             // Simple device-side spinlock (0=unlocked, 1=locked)

    __device__ void *malloc(size_t bytes)
    {
        // 1) Calculate how many pages we need
        size_t pagesNeeded = (bytes + pageSize + sizeof(AllocationHeader)) / pageSize;
        if (pagesNeeded == 0 || pagesNeeded > numPages)
        {
            return nullptr;
        }

        // 2) Naive scan for contiguous free pages starting at nextFreePage
        size_t candidatePage = nextFreePage;
        bool found = false;
        size_t foundStart = 0;
        size_t freeCount = 0;

        size_t scanned = 0;
        while (scanned < numPages)
        {
            size_t wordIdx = candidatePage / BITS_PER_WORD;
            size_t bitPos = candidatePage % BITS_PER_WORD;
            unsigned int word = bitset[wordIdx];
            unsigned int mask = (1U << bitPos);

            bool allocated = (word & mask) != 0;
            if (!allocated)
            {
                // page is free
                if (freeCount == 0)
                {
                    foundStart = candidatePage;
                }
                freeCount++;
                if (freeCount >= pagesNeeded)
                {
                    found = true;
                    break;
                }
            }
            else
            {
                // reset free-count
                freeCount = 0;
            }

            candidatePage = (candidatePage + 1) % numPages;
            scanned++;
        }

        if (!found)
        {
            // No contiguous run found
            return nullptr;
        }

        // We found [foundStart .. foundStart+pagesNeeded-1].
        // 3) Acquire lock and confirm it is still free; then mark allocated.
        acquireLock(&lock);

        bool stillFree = true;
        for (size_t i = 0; i < pagesNeeded; i++)
        {
            size_t checkPage = (foundStart + i) % numPages;
            size_t wIdx = checkPage / BITS_PER_WORD;
            size_t bPos = checkPage % BITS_PER_WORD;
            unsigned int w = bitset[wIdx];
            if (w & (1U << bPos))
            {
                // It's allocated after all
                stillFree = false;
                break;
            }
        }

        if (!stillFree)
        {
            // Some other thread allocated these pages in between.
            // Advance nextFreePage and fail (or retry).
            nextFreePage = (foundStart + 1) % numPages;
            releaseLock(&lock);
            return nullptr;
        }

        // Mark all these pages as allocated
        for (size_t i = 0; i < pagesNeeded; i++)
        {
            markPageAllocated((foundStart + i) % numPages);
        }

        // Write an AllocationHeader in the *first* page
        char *blockPtr = static_cast<char *>(pool) + foundStart * pageSize;
        AllocationHeader *header = reinterpret_cast<AllocationHeader *>(blockPtr);
        header->pagesUsed = pagesNeeded;
        header->startPage = foundStart;

        // The user pointer is after the header
        void *userPtr = blockPtr + sizeof(AllocationHeader);

        // Update nextFreePage
        nextFreePage = (foundStart + pagesNeeded) % numPages;

        releaseLock(&lock);

        return userPtr;
    }

    __device__ void free(void *ptr)
    {
        if (!ptr)
            return;

        // The header is immediately before the user pointer
        char *headerPtr = reinterpret_cast<char *>(ptr) - sizeof(AllocationHeader);
        AllocationHeader *header = reinterpret_cast<AllocationHeader *>(headerPtr);

        size_t startPage = header->startPage;
        size_t pagesToFree = header->pagesUsed;
        if (pagesToFree == 0)
        {
            return; // invalid
        }

        acquireLock(&lock);
        for (size_t i = 0; i < pagesToFree; i++)
        {
            markPageFree((startPage + i) % numPages);
        }
        releaseLock(&lock);
    }

private:

    __device__ void acquireLock(int *lck)
    {
        // spin until we set from 0->1
        while (atomicCAS(lck, 0, 1) != 0)
        {
            // spin
        }
    }

    __device__ void releaseLock(int *lck)
    {
        atomicExch(lck, 0);
    }

    __device__ void markPageAllocated(size_t pageIndex)
    {
        size_t wordIndex = pageIndex / BITS_PER_WORD;
        unsigned int bitMask = 1U << (pageIndex % BITS_PER_WORD);
        atomicOr(&bitset[wordIndex], bitMask);
    }

    __device__ void markPageFree(size_t pageIndex)
    {
        size_t wordIndex = pageIndex / BITS_PER_WORD;
        unsigned int bitMask = ~(1U << (pageIndex % BITS_PER_WORD));
        atomicAnd(&bitset[wordIndex], bitMask);
    }
};

__host__
CudaMemoryManager* createCudaMemoryManager(size_t numPages, size_t pageSize)
{
    // 1) Allocate the manager object in unified memory (host/device visible).
    CudaMemoryManager* mm = nullptr;
    cudaError_t err = cudaMallocManaged(&mm, sizeof(CudaMemoryManager));
    if (err != cudaSuccess || !mm) {
        fprintf(stderr, "Failed to allocate the CudaMemoryManager struct.\n");
        return nullptr;
    }

    // 2) Zero-initialize it (safe for all fields).
    memset(mm, 0, sizeof(CudaMemoryManager));

    // 3) Fill in the basic fields.
    mm->numPages  = numPages;
    mm->pageSize  = pageSize;
    mm->nextFreePage = 0;
    mm->lock      = 0; // unlocked

    // 4) Compute the pool size in bytes
    size_t totalBytes = numPages * pageSize;

    // 5) Allocate the pool on the device
    err = cudaMalloc(&(mm->pool), totalBytes);
    if (err != cudaSuccess || !mm->pool) {
        fprintf(stderr, "Failed to allocate the device pool of size %zu.\n", totalBytes);
        cudaFree(mm);
        return nullptr;
    }

    // 6) Allocate the bitset (1 bit per page => 1/32 words per page).
    mm->bitsetLength = (numPages + (BITS_PER_WORD - 1)) / BITS_PER_WORD; 
    err = cudaMalloc(&(mm->bitset), mm->bitsetLength * sizeof(unsigned int));
    if (err != cudaSuccess || !mm->bitset) {
        fprintf(stderr, "Failed to allocate bitset of length %zu.\n", mm->bitsetLength);
        cudaFree(mm->pool);
        cudaFree(mm);
        return nullptr;
    }

    // 7) Clear the bitset to 0 => all pages free
    //    We can do this on the host using cudaMemset:
    err = cudaMemset(mm->bitset, 0, mm->bitsetLength * sizeof(unsigned int));
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to memset bitset to zero.\n");
        // Cleanup everything
        cudaFree(mm->bitset);
        cudaFree(mm->pool);
        cudaFree(mm);
        return nullptr;
    }

    // 8) Done! Return the pointer to the manager.
    return mm;
}