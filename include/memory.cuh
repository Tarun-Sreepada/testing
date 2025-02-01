#include <stdio.h>
#include <cuda_runtime.h>
#include <sys/types.h>  // For ssize_t (if not already defined)

// Increase MAX_FREE_BLOCKS if heavy fragmentation is expected.
#define MAX_FREE_BLOCKS 16384

// // A free list node representing a contiguous free block of pages.
// // Using ssize_t for large page indices and counts.
// struct FreeBlock {
//     ssize_t startPage;  // Starting page index in the pool
//     ssize_t numPages;   // Number of contiguous free pages in this block
// };

// // The CUDA memory manager structure using a void pointer for the pool.
// struct CudaMemoryManager {
//     void* pool;            // Pointer to the memory pool (unified memory)
//     ssize_t numPages;      // Total number of pages in the pool
//     unsigned int pageSize; // Page size in bytes (e.g., 2048)
    
//     // Free list: an array of free blocks, maintained in sorted order by startPage.
//     FreeBlock freeList[MAX_FREE_BLOCKS];
//     int freeBlockCount;    // Number of valid free blocks in the free list.
    
//     // A simple spin-lock (0: unlocked, 1: locked) to protect free list updates.
//     int lock;
// };

// __device__ void acquireLock(int* lock) {
//     // Spin until we can atomically set lock from 0 to 1.
//     while (atomicCAS(lock, 0, 1) != 0) {
//         // Optionally, insert a short delay or __nanosleep().
//         __threadfence_system();
//         __nanosleep(1000);
//     }
// }

// __device__ void releaseLock(int* lock) {
//     atomicExch(lock, 0);
// }

// /*
//  * createMemoryManager:
//  *   totalBytes: total size (in bytes) of the memory pool you wish to manage.
//  *   pageSize:   size (in bytes) of each page (for example, 2048).
//  *
//  * Returns a pointer to a unified-memory–allocated CudaMemoryManager,
//  * or nullptr if an allocation error occurs.
//  */
// CudaMemoryManager* createMemoryManager(unsigned int numPages, unsigned int pageSize) {
//     cudaError_t err;
    
//     // Allocate unified memory for the manager structure.
//     CudaMemoryManager* mgr = nullptr;
//     err = cudaMallocManaged(&mgr, sizeof(CudaMemoryManager));
//     if (err != cudaSuccess) {
//         printf("Error allocating unified memory for CudaMemoryManager: %s\n", cudaGetErrorString(err));
//         return nullptr;
//     }
    
//     // Calculate the total number of pages (rounding up).
//     // Use 1ULL to ensure 64-bit arithmetic.
//     mgr->numPages = numPages;
//     mgr->pageSize = pageSize;
//     ssize_t totalBytes = 1ULL * numPages * pageSize;
    
//     // Allocate unified memory for the memory pool.
//     err = cudaMalloc(&mgr->pool, totalBytes);
//     if (err != cudaSuccess) {
//         printf("Error allocating unified memory for pool: %s\n", cudaGetErrorString(err));
//         cudaFree(mgr);
//         return nullptr;
//     }
    
//     // Initialize the free list: the entire pool is free.
//     mgr->freeBlockCount = 1;
//     mgr->freeList[0].startPage = 0;
//     mgr->freeList[0].numPages = numPages;
    
//     // Initialize the spin-lock to unlocked (0).
//     mgr->lock = 0;
    
//     // Ensure that the device sees the initialized memory.
//     cudaDeviceSynchronize();
    
//     return mgr;
// }



// __device__ void* deviceMemMalloc(CudaMemoryManager* mgr, size_t bytes) {
//     // Round up 'bytes' to a multiple of pageSize.
//     ssize_t pagesNeeded = (ssize_t)((bytes + mgr->pageSize - 1u) / mgr->pageSize);

//     // Acquire the lock protecting the free list.
//     acquireLock(&mgr->lock);

//     int foundIndex = -1;
//     // Search for a free block with enough pages.
//     for (int i = 0; i < mgr->freeBlockCount; i++) {
//         if (mgr->freeList[i].numPages >= pagesNeeded) {
//             foundIndex = i;
//             break;
//         }
//     }

//     if (foundIndex < 0) {
//         // No free block is large enough.
//         releaseLock(&mgr->lock);
//         return nullptr;
//     }

//     // Get the starting page of the found free block.
//     ssize_t startPage = mgr->freeList[foundIndex].startPage;
//     // Update the free list entry.
//     if (mgr->freeList[foundIndex].numPages == pagesNeeded) {
//         // Exact match—remove this free block.
//         for (int j = foundIndex; j < mgr->freeBlockCount - 1; j++) {
//             mgr->freeList[j] = mgr->freeList[j + 1];
//         }
//         mgr->freeBlockCount--;
//     } else {
//         // Block is larger than needed—update it.
//         mgr->freeList[foundIndex].startPage += pagesNeeded;
//         mgr->freeList[foundIndex].numPages -= pagesNeeded;
//     }

//     // Release the lock.
//     releaseLock(&mgr->lock);

//     // Return the pointer to the allocated block.
//     // (Cast the pool to a char pointer for pointer arithmetic.)
//     return (void*)((char*)(mgr->pool) + startPage * mgr->pageSize);
// }

// __device__ void deviceMemFree(CudaMemoryManager* mgr, void* ptr, size_t bytes) {
//     // Compute the offset from the beginning of the pool.
//     char* cptr = (char*)ptr;
//     size_t offset = cptr - (char*)(mgr->pool);
//     ssize_t startPage = offset / mgr->pageSize;
//     ssize_t pagesToFree = (ssize_t)((bytes + mgr->pageSize - 1u) / mgr->pageSize);

//     // Acquire the lock to update the free list.
//     acquireLock(&mgr->lock);

//     // Determine the insertion index in the sorted free list.
//     int i = 0;
//     while (i < mgr->freeBlockCount && mgr->freeList[i].startPage < startPage)
//         i++;

//     // Insert the new free block at index 'i' (if there is room in the free list).
//     if (mgr->freeBlockCount < MAX_FREE_BLOCKS) {
//         // Shift subsequent free blocks to make room.
//         for (int j = mgr->freeBlockCount; j > i; j--) {
//             mgr->freeList[j] = mgr->freeList[j - 1];
//         }
//         mgr->freeList[i].startPage = startPage;
//         mgr->freeList[i].numPages = pagesToFree;
//         mgr->freeBlockCount++;
//     } else {
//         // Free list overflow. In production, signal an error.
//         releaseLock(&mgr->lock);
//         return;
//     }

//     // Attempt to merge with the previous free block.
//     if (i > 0) {
//         if (mgr->freeList[i - 1].startPage + mgr->freeList[i - 1].numPages == mgr->freeList[i].startPage) {
//             // Merge current block into the previous block.
//             mgr->freeList[i - 1].numPages += mgr->freeList[i].numPages;
//             // Remove the current block from the list.
//             for (int j = i; j < mgr->freeBlockCount - 1; j++) {
//                 mgr->freeList[j] = mgr->freeList[j + 1];
//             }
//             mgr->freeBlockCount--;
//             i--;  // Adjust index to point to the merged block.
//         }
//     }

//     // Attempt to merge with the next free block.
//     if (i < mgr->freeBlockCount - 1) {
//         if (mgr->freeList[i].startPage + mgr->freeList[i].numPages == mgr->freeList[i + 1].startPage) {
//             // Merge the next block into the current block.
//             mgr->freeList[i].numPages += mgr->freeList[i + 1].numPages;
//             // Remove the next block.
//             for (int j = i + 1; j < mgr->freeBlockCount - 1; j++) {
//                 mgr->freeList[j] = mgr->freeList[j + 1];
//             }
//             mgr->freeBlockCount--;
//         }
//     }

//     // Release the lock.
//     releaseLock(&mgr->lock);
// }


#include <cuda_runtime.h>
#include <stdio.h>
#include <sys/types.h>

#define BITS_PER_WORD 32

//----------------------------------------------------------------------
// CudaMemoryManager using a bitset for allocation tracking
//----------------------------------------------------------------------

struct CudaMemoryManager {
    void* pool;            // Pointer to the memory pool.
    size_t numPages;       // Total number of pages in the pool.
    unsigned int pageSize; // Page size in bytes (e.g., 2048).

    // Bitset: one bit per page; stored as an array of 32-bit words.
    unsigned int* bitset;  // 0 = free, 1 = allocated.
    size_t bitsetLength;   // Number of 32-bit words in the bitset.

    // A simple spin-lock to protect updates to the bitset.
    int lock;
};

//----------------------------------------------------------------------
// Spin-lock functions (device-side)
//----------------------------------------------------------------------

__device__ void acquireLock(int* lock) {
    // Try to set the lock from 0 to 1. If not successful, delay a little.
    while (atomicCAS(lock, 0, 1) != 0) {
        // A short busy-wait loop.
        // for (volatile int i = 0; i < 100; i++) { }
        // __nanosleep(10);
        // __threadfence_system();
    }
}

__device__ void releaseLock(int* lock) {
    atomicExch(lock, 0);
}

//----------------------------------------------------------------------
// Bitset helper functions (device-side)
//----------------------------------------------------------------------


// Mark a page as allocated.
__device__ void markPageAllocated(CudaMemoryManager* mgr, size_t pageIndex) {
    size_t wordIndex = pageIndex / BITS_PER_WORD;
    unsigned int bitMask = 1U << (pageIndex % BITS_PER_WORD);
    atomicOr(&mgr->bitset[wordIndex], bitMask);
    // mgr->bitset[wordIndex] |= bitMask;
}

// Mark a page as free.
__device__ void markPageFree(CudaMemoryManager* mgr, size_t pageIndex) {
    size_t wordIndex = pageIndex / BITS_PER_WORD;
    unsigned int bitMask = ~(1U << (pageIndex % BITS_PER_WORD));
    atomicAnd(&mgr->bitset[wordIndex], bitMask);
    // mgr->bitset[wordIndex] &= bitMask;
}

//----------------------------------------------------------------------
// Device-side allocation functions (deviceMemMalloc and deviceMemFree)
//----------------------------------------------------------------------

/*
 * deviceMemMalloc:
 *   Allocates at least 'bytes' bytes from the pool.
 *   Rounds up the requested size to a multiple of the page size,
 *   then scans the bitset for a contiguous block of free pages.
 *
 *   Returns a pointer into the pool if successful, or nullptr if not.
 */
__device__ void* deviceMemMalloc(CudaMemoryManager* mgr, size_t bytes) {
    size_t pagesNeeded = (bytes + mgr->pageSize - 1) / mgr->pageSize;
    if (pagesNeeded == 0 || pagesNeeded > mgr->numPages) return nullptr;

    acquireLock(&mgr->lock);

    size_t startPage = 0;
    size_t currentRun = 0;

    for (size_t wordIdx = 0; wordIdx < mgr->bitsetLength; ++wordIdx) {
        unsigned int word = mgr->bitset[wordIdx];
        unsigned int freeWord = ~word;

        if (freeWord == 0) {
            currentRun = 0;
            continue;
        }

        // Process each free run in the current word
        while (freeWord != 0) {
            // Find the first free page in the current segment
            int pos = __ffs(freeWord) - 1;
            size_t firstPage = wordIdx * BITS_PER_WORD + pos;

            // Calculate maximum contiguous free pages from this position
            unsigned int mask = freeWord >> pos;
            int runLength = __ffs(~mask) - 1; // Find first 0 after pos
            if (runLength <= 0) runLength = BITS_PER_WORD - pos;

            // Check continuation from previous run
            if (currentRun == 0) {
                startPage = firstPage;
                currentRun = runLength;
            } else if (startPage + currentRun == firstPage) {
                currentRun += runLength;
            } else {
                // Check if previous run was sufficient
                if (currentRun >= pagesNeeded) break;
                startPage = firstPage;
                currentRun = runLength;
            }

            // Check if we've found enough pages
            if (currentRun >= pagesNeeded) break;

            // Move to next potential free area in the word
            freeWord &= ~((1U << (pos + runLength)) - 1);
        }

        if (currentRun >= pagesNeeded) break;
    }

    if (currentRun >= pagesNeeded) {
        // Mark pages as allocated
        for (size_t i = startPage; i < startPage + pagesNeeded; ++i) {
            markPageAllocated(mgr, i);
        }
        releaseLock(&mgr->lock);
        return static_cast<char*>(mgr->pool) + startPage * mgr->pageSize;
    }

    releaseLock(&mgr->lock);
    return nullptr;
}
/*
 * deviceMemFree:
 *   Frees a block previously allocated by deviceMemMalloc.
 *
 *   Parameters:
 *     - mgr: pointer to the memory manager.
 *     - ptr: pointer previously returned by deviceMemMalloc.
 *     - bytes: the original allocation size in bytes.
 */
__device__ void deviceMemFree(CudaMemoryManager* mgr, void* ptr, size_t bytes) {
    size_t offset = (char*)ptr - (char*)mgr->pool;
    size_t startPage = offset / mgr->pageSize;
    size_t pagesToFree = (bytes + mgr->pageSize - 1) / mgr->pageSize;

    acquireLock(&mgr->lock);
    for (size_t i = startPage; i < startPage + pagesToFree; i++) {
        markPageFree(mgr, i);
    }
    releaseLock(&mgr->lock);
}

//----------------------------------------------------------------------
// Host-side initialization for CudaMemoryManager using a bitset
//----------------------------------------------------------------------

CudaMemoryManager* createMemoryManager(unsigned long long total_bytes, unsigned int pageSize) {
    cudaError_t err;

    // Allocate unified memory for the manager structure.
    CudaMemoryManager* mgr = nullptr;
    err = cudaMallocManaged(&mgr, sizeof(CudaMemoryManager));
    if (err != cudaSuccess) {
        printf("Error allocating CudaMemoryManager: %s\n", cudaGetErrorString(err));
        return nullptr;
    }

    // Calculate the total number of pages (rounding up).
    size_t numPages = (total_bytes + pageSize - 1) / pageSize;

    mgr->pageSize = pageSize;
    mgr->numPages = numPages;

    // Allocate the memory pool.
    err = cudaMallocManaged(&mgr->pool, mgr->numPages * pageSize * 1ULL);

    if (err != cudaSuccess) {
        printf("Error allocating pool: %s\n", cudaGetErrorString(err));
        cudaFree(mgr);
        return nullptr;
    }

    std::cout << "Allocated: " << mgr->numPages * pageSize * 1ULL  / (1024 * 1024) << " MB" << std::endl;
    std::cout << "Num pages: " << mgr->numPages << std::endl;
    std::cout << "Page size: " << mgr->pageSize << std::endl;

    // Allocate the bitset.
    mgr->bitsetLength = (mgr->numPages + BITS_PER_WORD - 1) / BITS_PER_WORD;
    std::cout << "Bitset length: " << mgr->bitsetLength << std::endl;
    err = cudaMalloc(&mgr->bitset, mgr->bitsetLength * sizeof(unsigned int));
    if (err != cudaSuccess) {
        printf("Error allocating bitset: %s\n", cudaGetErrorString(err));
        cudaFree(mgr->pool);
        cudaFree(mgr);
        return nullptr;
    }

    // Initialize the bitset (all pages free).
    // for (size_t i = 0; i < mgr->bitsetLength; i++) {
    //     mgr->bitset[i] = 0;
    // }
    cudaMemset(mgr->bitset, 0, mgr->bitsetLength * sizeof(unsigned int));

    // Initialize the lock.
    mgr->lock = 0;

    // Synchronize to ensure the manager is visible to the device.
    cudaDeviceSynchronize();

    return mgr;
}
