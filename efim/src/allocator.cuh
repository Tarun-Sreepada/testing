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
    // The storage for our work items.
    T items[CAPACITY];

    // 'head' points to the next element to dequeue.
    // 'tail' points to the next free slot for enqueuing.
    unsigned int head;
    unsigned int tail;
    uint32_t work_count;

    // Initialize the queue.
    __host__ __device__ void init() {
        head = 0;
        tail = 0;
        work_count = 0;
    }

    // Returns true if the queue is empty.
    __device__ bool isEmpty() {
        // Atomic read is not strictly necessary when just reading,
        // but use atomicAdd(&var, 0) to be safe.
        unsigned int curHead = atomicAdd(&head, 0);
        unsigned int curTail = atomicAdd(&tail, 0);
        return (curHead == curTail);
    }

    // Returns true if the queue is full.
    // One slot is left unused to differentiate between full and empty.
    __device__ bool isFull() {
        unsigned int curTail = atomicAdd(&tail, 0);
        unsigned int curHead = atomicAdd(&head, 0);
        return (((curTail + 1) % CAPACITY) == curHead);
    }

    // Atomically enqueue a new work item.
    // Returns false if the queue is full.
    __device__ bool enqueue(const T &item) {
        unsigned int curTail, curHead, nextTail;
        do {
            // Atomically read tail and head.
            curTail = atomicAdd(&tail, 0);
            curHead = atomicAdd(&head, 0);
            atomicAdd(&work_count, 1);

            nextTail = (curTail + 1) % CAPACITY;
            // If the nextTail equals head, the queue is full.
            if (nextTail == curHead) {
                return false;
            }
            // Try to reserve the slot by atomically updating the tail.
            // atomicCAS returns the old value at &tail.
        } while (atomicCAS(&tail, curTail, nextTail) != curTail);

        // Successfully reserved slot at position curTail.
        items[curTail] = item;
        return true;
    }

    __host__ bool host_enqueue(const T &item) {
        unsigned int curTail, curHead, nextTail;
        curTail = tail;
        curHead = head;
        nextTail = (curTail + 1) % CAPACITY;
        work_count += 1;
        if (nextTail == curHead) {
            return false;
        }
        tail = nextTail;
        items[curTail] = item;
        return true;

    }

    __device__ bool dequeue(T &out) {
        unsigned int curHead, curTail, nextHead;

        // Read the current indices.
        curHead = atomicAdd(&head, 0);
        curTail = atomicAdd(&tail, 0);

        // Check if the queue is empty.
        if (curHead == curTail) {
            return false;
        }

        nextHead = (curHead + 1) % CAPACITY;
        // Try to claim the work item by atomically updating head.
        if (atomicCAS(&head, curHead, nextHead) == curHead) {
            // Successfully claimed the item, copy it out.
            out = items[curHead];
            // Optionally, update the work_count.
            return true;
        }

        // Failed to claim the item.
        return false;
    }

};





#endif  // ALLOCATOR_CUH
