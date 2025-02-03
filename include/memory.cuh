#include <cuda_runtime.h>
#include <stdio.h>
#include <sys/types.h>

#define BITS_PER_WORD 32

struct CudaMemoryManager {
    void* pool;             // Pointer to the memory pool.
    size_t numPages;        // Total number of pages in the pool.
    unsigned int pageSize;  // Page size in bytes (e.g., 2048).

    // Bitset: one bit per page; stored as an array of 32-bit words.
    unsigned int* bitset;   // 0 = free, 1 = allocated.
    size_t bitsetLength;    // Number of 32-bit words in the bitset.

    // A simple spin-lock to protect updates to the bitset.
    int lock;

    // Candidate starting page index to resume scanning.
    size_t nextFreePage;
};

__device__ void acquireLock(int* lock) {
    // Try to set the lock from 0 to 1. If not successful, delay a little.
    while (atomicCAS(lock, 0, 1) != 0) {
        // __nanosleep(10);
        // __threadfence_system();
    }
}

__device__ void releaseLock(int* lock) {
    atomicExch(lock, 0);
}

// Mark a page as allocated.
__device__ void markPageAllocated(CudaMemoryManager* mgr, size_t pageIndex) {
    size_t wordIndex = pageIndex / BITS_PER_WORD;
    unsigned int bitMask = 1U << (pageIndex % BITS_PER_WORD);
    // atomicOr(&mgr->bitset[wordIndex], bitMask);
    mgr->bitset[wordIndex] |= bitMask;
}

// Mark a page as free.
__device__ void markPageFree(CudaMemoryManager* mgr, size_t pageIndex) {
    size_t wordIndex = pageIndex / BITS_PER_WORD;
    unsigned int bitMask = ~(1U << (pageIndex % BITS_PER_WORD));
    // atomicAnd(&mgr->bitset[wordIndex], bitMask);
    mgr->bitset[wordIndex] &= bitMask;
}

__device__ void* deviceMemMalloc(CudaMemoryManager* mgr, size_t bytes) {
    size_t pagesNeeded = (bytes + mgr->pageSize - 1) / mgr->pageSize;
    if (pagesNeeded == 0 || pagesNeeded > mgr->numPages) return nullptr;

    size_t candidatePage = mgr->nextFreePage;
    // printf("Candidate page: %lu\n", candidatePage);
    bool found = false;
    size_t foundStart = 0;
    size_t freeCount = 0;

    // Start scanning from candidatePage.
    size_t startWord = candidatePage / 32;
    size_t startBit = candidatePage % 32;
    size_t totalWords = mgr->bitsetLength;
    size_t scannedPages = 0;

    // We'll scan at most numPages entries.
    while (scannedPages < mgr->numPages) {
        size_t wordIdx = (startWord) % totalWords;
        unsigned int word = mgr->bitset[wordIdx];

        // For the first word, mask out bits before startBit.
        if (scannedPages == 0 && startBit != 0) {
            word |= ~0U << startBit;  // Mark earlier bits as allocated.
        }

        // Invert the word to have free bits = 1.
        unsigned int freeWord = ~word;
        if (freeWord != 0) {
            // __ffs returns 1-indexed position of least-significant 1.
            int bitPos = __ffs(freeWord) - 1;
            // The absolute page index of the free bit.
            size_t absPage = wordIdx * 32 + bitPos;
            // Reset our running counter if not contiguous.
            if (freeCount == 0) {
                // We want our candidate to be no less than candidatePage.
                if (absPage < candidatePage && scannedPages == 0) {
                    // We're still in the masked-out region; move to next free candidate.
                    scannedPages += (32 - startBit);
                    startWord++; 
                    continue;
                }
                foundStart = absPage;
                freeCount = 0;
            }

            // Now count contiguous free bits in the current word starting at bitPos.
            // Use __ffs on the shifted word to determine how many free bits are in a row.
            unsigned int shifted = freeWord >> bitPos;
            int contiguous = __ffs(~shifted);  // __ffs returns first zero (allocated bit) position.
            if (contiguous == 0) {
                // If __ffs returns 0, then shifted is all ones.
                contiguous = 32 - bitPos;
            } else {
                contiguous = contiguous - 1;
            }

            freeCount += contiguous;
            scannedPages += contiguous;

            if (freeCount >= pagesNeeded) {
                found = true;
                break;
            }
            // Move to the next word.
            startWord++;
        } else {
            // No free bits in this word.
            scannedPages += 32;
            freeCount = 0;  // Reset the contiguous free count.
            startWord++;
        }
    }

    if (!found) return nullptr;

    // --- Now, acquire the lock and re-validate the found block ---
    acquireLock(&mgr->lock);

    // Re-check that the candidate block is free.
    bool candidateStillFree = true;
    for (size_t i = 0; i < pagesNeeded; i++) {
        size_t checkPage = (foundStart + i) % mgr->numPages;
        size_t wordIdx = checkPage / 32;
        unsigned int mask = 1U << (checkPage % 32);
        if (mgr->bitset[wordIdx] & mask) {
            candidateStillFree = false;
            break;
        }
    }

    if (!candidateStillFree) {
        // Candidate was taken; update nextFreePage and try again.
        mgr->nextFreePage = (foundStart + 1) % mgr->numPages;
        releaseLock(&mgr->lock);
        return deviceMemMalloc(mgr, bytes);
    }

    // Mark the pages as allocated.
    for (size_t i = 0; i < pagesNeeded; i++) {
        size_t pageIdx = (foundStart + i) % mgr->numPages;
        markPageAllocated(mgr, pageIdx);
    }

    // Update nextFreePage pointer.
    mgr->nextFreePage = (foundStart + pagesNeeded) % mgr->numPages;
    releaseLock(&mgr->lock);

    return static_cast<char*>(mgr->pool) + foundStart * mgr->pageSize;
}


__device__ void deviceMemFree(CudaMemoryManager* mgr, void* ptr, size_t bytes) {
    size_t offset = static_cast<char*>(ptr) - static_cast<char*>(mgr->pool);
    size_t startPage = offset / mgr->pageSize;
    size_t pagesToFree = (bytes + mgr->pageSize - 1) / mgr->pageSize;

    acquireLock(&mgr->lock);
    for (size_t i = 0; i < pagesToFree; i++) {
        markPageFree(mgr, (startPage + i) % mgr->numPages);
    }
    releaseLock(&mgr->lock);
}


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
