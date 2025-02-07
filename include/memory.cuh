#pragma once
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdint.h>

#define BITS_PER_WORD 32


// struct AllocationHeader
// {
//     size_t pagesUsed;
//     size_t startPage;
// };


// struct CudaMemoryManager
// {
//     void *pool;           // Base pointer to the GPU memory pool
//     size_t numPages;      // Total number of pages
//     size_t pageSize;      // Size of each page in bytes
//     unsigned int *bitset; // Bitset array: 1 => page allocated, 0 => free
//     size_t bitsetLength;  // Number of 32-bit words in 'bitset'
//     size_t nextFreePage;  // Next page index to start scanning from
//     int lock;             // Simple device-side spinlock (0=unlocked, 1=locked)

//     __device__ void *malloc(size_t bytes)
//     {
//         // 1) Calculate how many pages we need
//         size_t pagesNeeded = (bytes + pageSize + sizeof(AllocationHeader)) / pageSize;
//         if (pagesNeeded == 0 || pagesNeeded > numPages)
//         {
//             return nullptr;
//         }

//         // 2) Naive scan for contiguous free pages starting at nextFreePage
//         size_t candidatePage = nextFreePage;
//         bool found = false;
//         size_t foundStart = 0;
//         size_t freeCount = 0;

//         size_t scanned = 0;
//         while (scanned < numPages)
//         {
//             size_t wordIdx = candidatePage / BITS_PER_WORD;
//             size_t bitPos = candidatePage % BITS_PER_WORD;
//             unsigned int word = bitset[wordIdx];
//             unsigned int mask = (1U << bitPos);

//             bool allocated = (word & mask) != 0;
//             if (!allocated)
//             {
//                 // page is free
//                 if (freeCount == 0)
//                 {
//                     foundStart = candidatePage;
//                 }
//                 freeCount++;
//                 if (freeCount >= pagesNeeded)
//                 {
//                     found = true;
//                     break;
//                 }
//             }
//             else
//             {
//                 // reset free-count
//                 freeCount = 0;
//             }

//             candidatePage = (candidatePage + 1) % numPages;
//             scanned++;
//         }

//         if (!found)
//         {
//             // No contiguous run found
//             return nullptr;
//         }

//         // We found [foundStart .. foundStart+pagesNeeded-1].
//         // 3) Acquire lock and confirm it is still free; then mark allocated.
//         acquireLock(&lock);

//         bool stillFree = true;
//         for (size_t i = 0; i < pagesNeeded; i++)
//         {
//             size_t checkPage = (foundStart + i) % numPages;
//             size_t wIdx = checkPage / BITS_PER_WORD;
//             size_t bPos = checkPage % BITS_PER_WORD;
//             unsigned int w = bitset[wIdx];
//             if (w & (1U << bPos))
//             {
//                 // It's allocated after all
//                 stillFree = false;
//                 break;
//             }
//         }

//         if (!stillFree)
//         {
//             // Some other thread allocated these pages in between.
//             // Advance nextFreePage and fail (or retry).
//             nextFreePage = (foundStart + 1) % numPages;
//             releaseLock(&lock);
//             return nullptr;
//         }

//         // Mark all these pages as allocated
//         for (size_t i = 0; i < pagesNeeded; i++)
//         {
//             markPageAllocated((foundStart + i) % numPages);
//         }

//         // Write an AllocationHeader in the *first* page
//         char *blockPtr = static_cast<char *>(pool) + foundStart * pageSize;
//         AllocationHeader *header = reinterpret_cast<AllocationHeader *>(blockPtr);
//         header->pagesUsed = pagesNeeded;
//         header->startPage = foundStart;

//         // The user pointer is after the header
//         void *userPtr = blockPtr + sizeof(AllocationHeader);

//         // Update nextFreePage
//         nextFreePage = (foundStart + pagesNeeded) % numPages;

//         releaseLock(&lock);

//         // memset to 0
//         memset(userPtr, 0, bytes);

//         return userPtr;
//     }

//     __device__ void free(void *ptr)
//     {
//         if (!ptr)
//             return;

//         // The header is immediately before the user pointer
//         char *headerPtr = reinterpret_cast<char *>(ptr) - sizeof(AllocationHeader);
//         AllocationHeader *header = reinterpret_cast<AllocationHeader *>(headerPtr);

//         size_t startPage = header->startPage;
//         size_t pagesToFree = header->pagesUsed;
//         if (pagesToFree == 0)
//         {
//             return; // invalid
//         }

//         acquireLock(&lock);
//         for (size_t i = 0; i < pagesToFree; i++)
//         {
//             markPageFree((startPage + i) % numPages);
//         }
//         releaseLock(&lock);
//     }

// private:

//     __device__ void acquireLock(int *lck)
//     {
//         // spin until we set from 0->1
//         while (atomicCAS(lck, 0, 1) != 0)
//         {
//             // spin
//         }
//     }

//     __device__ void releaseLock(int *lck)
//     {
//         atomicExch(lck, 0);
//     }

//     __device__ void markPageAllocated(size_t pageIndex)
//     {
//         size_t wordIndex = pageIndex / BITS_PER_WORD;
//         unsigned int bitMask = 1U << (pageIndex % BITS_PER_WORD);
//         atomicOr(&bitset[wordIndex], bitMask);
//     }

//     __device__ void markPageFree(size_t pageIndex)
//     {
//         size_t wordIndex = pageIndex / BITS_PER_WORD;
//         unsigned int bitMask = ~(1U << (pageIndex % BITS_PER_WORD));
//         atomicAnd(&bitset[wordIndex], bitMask);
//     }
// };

// __host__
// CudaMemoryManager* createCudaMemoryManager(size_t numPages, size_t pageSize)
// {
//     // 1) Allocate the manager object in unified memory (host/device visible).
//     CudaMemoryManager* mm = nullptr;
//     cudaError_t err = cudaMallocManaged(&mm, sizeof(CudaMemoryManager));
//     if (err != cudaSuccess || !mm) {
//         fprintf(stderr, "Failed to allocate the CudaMemoryManager struct.\n");
//         return nullptr;
//     }

//     // 2) Zero-initialize it (safe for all fields).
//     memset(mm, 0, sizeof(CudaMemoryManager));

//     // 3) Fill in the basic fields.
//     mm->numPages  = numPages;
//     mm->pageSize  = pageSize;
//     mm->nextFreePage = 0;
//     mm->lock      = 0; // unlocked

//     // 4) Compute the pool size in bytes
//     size_t totalBytes = numPages * pageSize;

//     // 5) Allocate the pool on the device
//     // err = cudaMalloc(&(mm->pool), totalBytes);

//     err = cudaMallocManaged(&(mm->pool), totalBytes);
//     if (err != cudaSuccess || !mm->pool) {
//         fprintf(stderr, "Failed to allocate the device pool of size %zu.\n", totalBytes);
//         cudaFree(mm);
//         return nullptr;
//     }

//     // 6) Allocate the bitset (1 bit per page => 1/32 words per page).
//     mm->bitsetLength = (numPages + (BITS_PER_WORD - 1)) / BITS_PER_WORD; 
//     err = cudaMalloc(&(mm->bitset), mm->bitsetLength * sizeof(unsigned int));
//     if (err != cudaSuccess || !mm->bitset) {
//         fprintf(stderr, "Failed to allocate bitset of length %zu.\n", mm->bitsetLength);
//         cudaFree(mm->pool);
//         cudaFree(mm);
//         return nullptr;
//     }

//     // 7) Clear the bitset to 0 => all pages free
//     //    We can do this on the host using cudaMemset:
//     err = cudaMemset(mm->bitset, 0, mm->bitsetLength * sizeof(unsigned int));
//     if (err != cudaSuccess) {
//         fprintf(stderr, "Failed to memset bitset to zero.\n");
//         // Cleanup everything
//         cudaFree(mm->bitset);
//         cudaFree(mm->pool);
//         cudaFree(mm);
//         return nullptr;
//     }

//     // 8) Done! Return the pointer to the manager.
//     return mm;
// }


//-------------------------------------------------------------------------------
// Structures used by the allocator
//-------------------------------------------------------------------------------

// A header stored at the beginning of every allocated block.
struct AllocationHeader {
    size_t pagesUsed;   // Number of pages allocated for this block.
    size_t startPage;   // Starting page index in the pool.
};

// A node in the free‐list that describes a contiguous segment of free pages.
struct FreeSegment {
    size_t startPage;    // Starting page index of this free segment.
    size_t length;       // Number of contiguous free pages in the segment.
    FreeSegment* next;   // Next segment in the free list.
};

struct CudaMemoryManager {
    // Memory pool fields.
    void*  pool;        // Base pointer to the memory pool.
    size_t numPages;    // Total number of pages in the pool.
    size_t pageSize;    // Size (in bytes) of each page.

    // A simple spinlock to protect free‐list operations.
    int lock;

    // Free list for tracking contiguous free segments (in units of pages).
    FreeSegment* freeListHead;

    FreeSegment* freeSegments; // Array of free-list nodes.
    size_t       maxSegments;  // Total number of nodes allocated (set to numPages).
    int pagesAllocated;

    // A stack (array) of indices into freeSegments[] that are not in use.
    int* freeSegmentStack;
    int  freeSegmentStackTop;  // Points to the next free slot in freeSegmentStack.

    //----------------------------------------------------------------------------
    // Device-side spinlock routines.
    //----------------------------------------------------------------------------
    __device__ void acquireLock(int *lck) {
        while (atomicCAS(lck, 0, 1) != 0) {
            // spin
        }
    }
    __device__ void releaseLock(int *lck) {
        atomicExch(lck, 0);
    }

    __device__ void* malloc(size_t bytes) {
        // Compute total bytes needed (including header) and round up to pages.
        size_t totalBytes = bytes + sizeof(AllocationHeader);
        size_t pagesNeeded = (totalBytes + pageSize - 1) / pageSize;
        if (pagesNeeded == 0 || pagesNeeded > numPages)
            return nullptr;

        acquireLock(&lock);

        // Search free list for a segment with enough pages.
        FreeSegment* prev = nullptr;
        FreeSegment* curr = freeListHead;
        while (curr) {
            if (curr->length >= pagesNeeded)
                break;
            prev = curr;
            curr = curr->next;
        }
        if (!curr) {
            // No free segment can satisfy the allocation.
            releaseLock(&lock);
            return nullptr;
        }
        size_t allocStart = curr->startPage;

        // Set up the allocation header in the pool.
        char* blockPtr = (char*)pool + allocStart * pageSize;
        AllocationHeader* header = (AllocationHeader*)blockPtr;
        header->pagesUsed = pagesNeeded;
        header->startPage = allocStart;

        // Update the free segment.
        if (curr->length == pagesNeeded) {
            // Exact fit: remove this node from the free list.
            if (prev)
                prev->next = curr->next;
            else
                freeListHead = curr->next;
            // Return the node to the freeSegmentStack.
            int idx = curr - freeSegments;  // pointer arithmetic yields index.
            freeSegmentStack[freeSegmentStackTop++] = idx;
        } else {
            // Otherwise, shrink the segment.
            curr->startPage += pagesNeeded;
            curr->length    -= pagesNeeded;
        }
        releaseLock(&lock);

        // Return pointer to user memory (after the header).
        void* userPtr = blockPtr + sizeof(AllocationHeader);
        // (Optionally, you could clear the user memory here.)
        memset(userPtr, 0, bytes);

        atomicAdd(&pagesAllocated, pagesNeeded);
        return userPtr;
    }

    __device__ void free(void* ptr) {
        if (!ptr)
            return;
        char* headerPtr = (char*)ptr - sizeof(AllocationHeader);
        AllocationHeader* header = (AllocationHeader*)headerPtr;
        size_t startPage   = header->startPage;
        size_t pagesToFree = header->pagesUsed;
        if (pagesToFree == 0)
            return;

        acquireLock(&lock);

        // Get a free node from our freeSegmentStack.
        if (freeSegmentStackTop <= 0) {
            // This should not happen if maxSegments was chosen large enough.
            releaseLock(&lock);
            return;
        }
        int nodeIdx = freeSegmentStack[--freeSegmentStackTop];
        FreeSegment* newSegment = &freeSegments[nodeIdx];
        newSegment->startPage = startPage;
        newSegment->length    = pagesToFree;
        newSegment->next      = nullptr;

        // Insert the new segment into the free list (keeping it sorted by startPage).
        if (!freeListHead || startPage < freeListHead->startPage) {
            newSegment->next = freeListHead;
            freeListHead = newSegment;
        } else {
            FreeSegment* curr = freeListHead;
            while (curr->next && curr->next->startPage < startPage)
                curr = curr->next;
            newSegment->next = curr->next;
            curr->next = newSegment;
        }

        // Merge adjacent free segments.
        FreeSegment* curr = freeListHead;
        while (curr && curr->next) {
            if (curr->startPage + curr->length == curr->next->startPage) {
                // Merge curr and curr->next.
                curr->length += curr->next->length;
                FreeSegment* toFree = curr->next;
                curr->next = toFree->next;
                int idx = toFree - freeSegments;
                freeSegmentStack[freeSegmentStackTop++] = idx;
            } else {
                curr = curr->next;
            }
        }

        releaseLock(&lock);
        atomicSub(&pagesAllocated, pagesToFree);
    }


};

//-------------------------------------------------------------------------------
// Host-side creation function.
// This routine allocates and initializes the CudaMemoryManager in unified memory.
// It creates a pool (of numPages * pageSize bytes) and sets up the free-list to
// indicate that the entire pool is initially free.
//-------------------------------------------------------------------------------
// __host__
// CudaMemoryManager* createCudaMemoryManager(size_t numPages, size_t pageSize) {
//     CudaMemoryManager* mm = nullptr;
//     cudaError_t err = cudaMallocManaged(&mm, sizeof(CudaMemoryManager));
//     if (err != cudaSuccess || !mm) {
//         fprintf(stderr, "Failed to allocate CudaMemoryManager structure.\n");
//         return nullptr;
//     }
//     memset(mm, 0, sizeof(CudaMemoryManager));
//     mm->numPages = numPages;
//     mm->pageSize = pageSize;
//     mm->lock = 0; // initially unlocked

//     // Allocate the memory pool.
//     size_t totalBytes = numPages * pageSize;
//     err = cudaMalloc(&mm->pool, totalBytes);
//     if (err != cudaSuccess || !mm->pool) {
//         fprintf(stderr, "Failed to allocate memory pool (%zu bytes).\n", totalBytes);
//         cudaFree(mm);
//         return nullptr;
//     }

//     // Allocate the freeSegments array.
//     mm->maxSegments = numPages; // worst-case: one node per page.
//     err = cudaMallocManaged(&mm->freeSegments, mm->maxSegments * sizeof(FreeSegment));
//     if (err != cudaSuccess || !mm->freeSegments) {
//         fprintf(stderr, "Failed to allocate freeSegments array.\n");
//         cudaFree(mm->pool);
//         cudaFree(mm);
//         return nullptr;
//     }

//     // Allocate the freeSegmentStack.
//     err = cudaMallocManaged(&mm->freeSegmentStack, mm->maxSegments * sizeof(int));
//     if (err != cudaSuccess || !mm->freeSegmentStack) {
//         fprintf(stderr, "Failed to allocate freeSegmentStack array.\n");
//         cudaFree(mm->freeSegments);
//         cudaFree(mm->pool);
//         cudaFree(mm);
//         return nullptr;
//     }

//     // Initialize the freeSegmentStack with indices [0, maxSegments-1].
//     mm->freeSegmentStackTop = mm->maxSegments;
//     for (int i = 0; i < mm->maxSegments; i++) {
//         mm->freeSegmentStack[i] = i;
//     }

//     // Create the initial free segment covering the entire pool.
//     int initIdx = mm->freeSegmentStack[--mm->freeSegmentStackTop];
//     mm->freeSegments[initIdx].startPage = 0;
//     mm->freeSegments[initIdx].length    = numPages;
//     mm->freeSegments[initIdx].next      = nullptr;
//     mm->freeListHead = &mm->freeSegments[initIdx];
//     mm->pagesAllocated = 0;

//     return mm;
// }

__host__
CudaMemoryManager* createCudaMemoryManager(size_t numPages, size_t pageSize) {
    cudaError_t err;

    // 1. Allocate the CudaMemoryManager structure on the device.
    CudaMemoryManager* d_mm = nullptr;
    err = cudaMalloc(&d_mm, sizeof(CudaMemoryManager));
    if (err != cudaSuccess || !d_mm) {
        fprintf(stderr, "Failed to allocate CudaMemoryManager structure.\n");
        return nullptr;
    }

    // 2. Set up a host temporary structure for initialization.
    CudaMemoryManager host_mm;
    memset(&host_mm, 0, sizeof(CudaMemoryManager));
    host_mm.numPages = numPages;
    host_mm.pageSize = pageSize;
    host_mm.lock = 0;  // initially unlocked

    // 3. Allocate the memory pool on the device.
    size_t totalBytes = numPages * pageSize;
    err = cudaMalloc(&host_mm.pool, totalBytes);
    if (err != cudaSuccess || !host_mm.pool) {
        fprintf(stderr, "Failed to allocate memory pool (%zu bytes).\n", totalBytes);
        cudaFree(d_mm);
        return nullptr;
    }

    // 4. Allocate the freeSegments array on the device.
    host_mm.maxSegments = numPages; // Worst-case: one node per page.
    err = cudaMalloc(&host_mm.freeSegments, host_mm.maxSegments * sizeof(FreeSegment));
    if (err != cudaSuccess || !host_mm.freeSegments) {
        fprintf(stderr, "Failed to allocate freeSegments array.\n");
        cudaFree(host_mm.pool);
        cudaFree(d_mm);
        return nullptr;
    }

    // 5. Allocate the freeSegmentStack array on the device.
    err = cudaMalloc(&host_mm.freeSegmentStack, host_mm.maxSegments * sizeof(int));
    if (err != cudaSuccess || !host_mm.freeSegmentStack) {
        fprintf(stderr, "Failed to allocate freeSegmentStack array.\n");
        cudaFree(host_mm.freeSegments);
        cudaFree(host_mm.pool);
        cudaFree(d_mm);
        return nullptr;
    }

    // 6. Initialize the freeSegmentStack.
    //    Because freeSegmentStack is allocated in device memory, prepare a temporary host array and copy it.
    int* h_freeSegmentStack = new int[host_mm.maxSegments];
    for (int i = 0; i < host_mm.maxSegments; i++) {
        h_freeSegmentStack[i] = i;
    }
    err = cudaMemcpy(host_mm.freeSegmentStack, h_freeSegmentStack,
                     host_mm.maxSegments * sizeof(int), cudaMemcpyHostToDevice);
    delete[] h_freeSegmentStack;
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to copy freeSegmentStack to device.\n");
        cudaFree(host_mm.freeSegmentStack);
        cudaFree(host_mm.freeSegments);
        cudaFree(host_mm.pool);
        cudaFree(d_mm);
        return nullptr;
    }
    host_mm.freeSegmentStackTop = host_mm.maxSegments;

    // 7. Create the initial free segment covering the entire pool.
    //    Pop one index from the freeSegmentStack.
    int initIdx = host_mm.freeSegmentStackTop - 1;
    host_mm.freeSegmentStackTop--; // Remove the last element.
    
    // Prepare a FreeSegment representing the entire pool.
    FreeSegment initSegment;
    initSegment.startPage = 0;
    initSegment.length = numPages;
    initSegment.next = nullptr;
    
    // Copy this FreeSegment into the freeSegments array at index initIdx.
    err = cudaMemcpy((char*)host_mm.freeSegments + initIdx * sizeof(FreeSegment),
                     &initSegment, sizeof(FreeSegment), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to initialize the free segment.\n");
        cudaFree(host_mm.freeSegmentStack);
        cudaFree(host_mm.freeSegments);
        cudaFree(host_mm.pool);
        cudaFree(d_mm);
        return nullptr;
    }
    
    // Set freeListHead to point to the initial segment.
    host_mm.freeListHead = (FreeSegment*)((char*)host_mm.freeSegments + initIdx * sizeof(FreeSegment));
    
    // 8. Set pagesAllocated to 0.
    host_mm.pagesAllocated = 0;

    // 9. Copy the host temporary structure into the device-allocated CudaMemoryManager.
    err = cudaMemcpy(d_mm, &host_mm, sizeof(CudaMemoryManager), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to copy CudaMemoryManager structure to device.\n");
        cudaFree(host_mm.freeSegmentStack);
        cudaFree(host_mm.freeSegments);
        cudaFree(host_mm.pool);
        cudaFree(d_mm);
        return nullptr;
    }

    return d_mm;
}