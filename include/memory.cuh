#pragma once
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <iostream>

#include <gallatin/allocators/global_allocator.cuh>

using namespace gallatin::allocators;

// // Utility functions.

// // Returns the next power of two greater than or equal to x.
// __host__ __device__ uint64_t next_power_of_two(uint64_t x)
// {
//     uint64_t power = 1;
//     while (power < x)
//         power *= 2;
//     return power;
// }

// // Returns the integer base‑2 logarithm of x.
// __host__ __device__ int log2_int(uint64_t x)
// {
//     int r = 0;
//     while (x > 1)
//     {
//         x /= 2;
//         r++;
//     }
//     return r;
// }

// // Define a three‑state enum for block status.
// enum BlockState { 
//     FREE = 0,    // Block is available.
//     SPLIT = 1,   // Block has been split into children.
//     ALLOCATED = 2  // Block is allocated.
// };

// // CudaMemoryManager manages the memory pool via the buddy system.
// struct CudaMemoryManager
// {
//     uint64_t totalSize;      // Total size (in bytes) of the managed memory.
//     void *memory;            // Pointer to the unified memory pool.
//     uint64_t minBlockSize;   // Minimum block size.
//     int maxOrder;            // Maximum order such that minBlockSize * 2^(maxOrder) == totalSize.
//     volatile int lock;       // Spinlock for thread safety.
//     int treeSize;            // Size of the binary tree array (using 1-based indexing).
//     BlockState *tree;        // Array representing the buddy tree.
//                              // Each node is FREE, SPLIT, or ALLOCATED.
//     uint64_t blockSizeTable[64]; // Precomputed block sizes for levels 0..maxOrder (assumes maxOrder < 64).

//     // Returns the block size for a given level.
//     // Level 0 corresponds to the entire pool; level maxOrder is the smallest block.
//     __host__ __device__ uint64_t block_size(int level) const
//     {
//         // Instead of recomputing, use the lookup table.
//         return blockSizeTable[level];
//     }

//     // Computes the pointer to the beginning of the block at a given tree node.
//     // 'index' is the node index (1-based) and 'level' is its depth.
//     __host__ __device__ void *block_ptr(int index, int level) const
//     {
//         // At level L, the leftmost node has index = 1 << L.
//         int leftmost = 1 << level;
//         uint64_t offset = (uint64_t)(index - leftmost) * block_size(level);
//         return static_cast<char *>(memory) + offset;
//     }

//     // ---- Thread-Safety: Locking Functions ----

//     // Device-side spin lock.
//     __device__ void acquire_lock() {
//         while (atomicCAS((int*)&lock, 0, 1) != 0) { }
//     }
//     __device__ void release_lock() {
//         atomicExch((int*)&lock, 0);
//     }

//     // Host-side spin lock (using GCC builtins).
//     __host__ void acquire_lock_host() {
//         while (__sync_val_compare_and_swap(&lock, 0, 1) != 0) { }
//     }
//     __host__ void release_lock_host() {
//         __sync_lock_release(&lock);
//     }

//     // ---- Allocation with Lock and Lookup Table ----
//     // Given a request of size 'bytes', we compute the desired level as:
//     //   desiredLevel = maxOrder - (log2(needed) - log2(minBlockSize))
//     // where needed = next_power_of_two(bytes) rounded up to at least minBlockSize.
//     // Then we perform a DFS search restricted to nodes at that level.
//     __device__ void *malloc(uint64_t bytes)
//     {
//         if (bytes == 0)
//             return nullptr;

//         uint64_t needed = next_power_of_two(bytes);
//         if (needed < minBlockSize)
//             needed = minBlockSize;
//         if (needed > totalSize)
//             return nullptr;

//         // Compute desired level.
//         int desiredLevel = maxOrder - (log2_int(needed) - log2_int(minBlockSize));

//         // Acquire lock to ensure thread safety.
//         acquire_lock();

//         // Use a DFS search to find a free block at the desired level.
//         // We define a simple DFS stack frame.
//         struct Frame {
//             int index;  // Node index (1-based)
//             int level;  // Current tree level (0 = root)
//             int state;  // DFS state: 0 = not processed, 1 = left processed, 2 = both processed.
//         };

//         Frame stack[32];
//         int depth = 0;
//         stack[depth] = {1, 0, 0};  // Start at the root.
//         void *result = nullptr;

//         while (depth >= 0) {
//             Frame &top = stack[depth];

//             if (top.index >= treeSize) {
//                 depth--;
//                 continue;
//             }

//             uint64_t bs = block_size(top.level);
//             if (bs < needed) {
//                 depth--;
//                 continue;
//             }

//             if (top.level == desiredLevel) {
//                 // At the desired level, if the block is FREE, allocate it.
//                 if (tree[top.index] == FREE) {
//                     tree[top.index] = ALLOCATED;
//                     result = block_ptr(top.index, top.level);
//                     break;
//                 }
//             }
//             else {  // If current level is above desiredLevel (i.e. block is larger than needed)
//                 if (top.state == 0) {
//                     // Mark this node as SPLIT if it is free.
//                     if (tree[top.index] == FREE)
//                         tree[top.index] = SPLIT;
//                     top.state = 1;
//                     int child_index = 2 * top.index;
//                     if (child_index < treeSize) {
//                         depth++;
//                         stack[depth] = {child_index, top.level + 1, 0};
//                     } else {
//                         top.state = 2;
//                     }
//                     continue;
//                 }
//                 else if (top.state == 1) {
//                     top.state = 2;
//                     int child_index = 2 * top.index + 1;
//                     if (child_index < treeSize) {
//                         depth++;
//                         stack[depth] = {child_index, top.level + 1, 0};
//                     }
//                     continue;
//                 }
//                 else {
//                     // Both children have been processed; try to coalesce.
//                     int left = 2 * top.index;
//                     int right = left + 1;
//                     if (left < treeSize && right < treeSize &&
//                         tree[left] == FREE && tree[right] == FREE)
//                     {
//                         tree[top.index] = FREE;
//                     }
//                     else {
//                         tree[top.index] = SPLIT;
//                     }
//                     depth--;
//                     continue;
//                 }
//             }
//             depth--;
//         }
//         release_lock();
//         return result;
//     }

//     __host__ void *host_malloc(uint64_t bytes)
//     {
//         if (bytes == 0)
//             return nullptr;

//         uint64_t needed = next_power_of_two(bytes);
//         if (needed < minBlockSize)
//             needed = minBlockSize;
//         if (needed > totalSize)
//             return nullptr;

//         int desiredLevel = maxOrder - (log2_int(needed) - log2_int(minBlockSize));

//         acquire_lock_host();

//         struct Frame {
//             int index;
//             int level;
//             int state;
//         };

//         Frame stack[32];
//         int depth = 0;
//         stack[depth] = {1, 0, 0};
//         void *result = nullptr;

//         while (depth >= 0) {
//             Frame &top = stack[depth];
//             if (top.index >= treeSize) {
//                 depth--;
//                 continue;
//             }
//             uint64_t bs = block_size(top.level);
//             if (bs < needed) {
//                 depth--;
//                 continue;
//             }
//             if (top.level == desiredLevel) {
//                 if (tree[top.index] == FREE) {
//                     tree[top.index] = ALLOCATED;
//                     result = block_ptr(top.index, top.level);
//                     break;
//                 }
//             }
//             else {
//                 if (top.state == 0) {
//                     if (tree[top.index] == FREE)
//                         tree[top.index] = SPLIT;
//                     top.state = 1;
//                     int child_index = 2 * top.index;
//                     if (child_index < treeSize) {
//                         depth++;
//                         stack[depth] = {child_index, top.level + 1, 0};
//                     } else {
//                         top.state = 2;
//                     }
//                     continue;
//                 }
//                 else if (top.state == 1) {
//                     top.state = 2;
//                     int child_index = 2 * top.index + 1;
//                     if (child_index < treeSize) {
//                         depth++;
//                         stack[depth] = {child_index, top.level + 1, 0};
//                     }
//                     continue;
//                 }
//                 else {
//                     int left = 2 * top.index;
//                     int right = left + 1;
//                     if (left < treeSize && right < treeSize &&
//                         tree[left] == FREE && tree[right] == FREE)
//                     {
//                         tree[top.index] = FREE;
//                     }
//                     else {
//                         tree[top.index] = SPLIT;
//                     }
//                     depth--;
//                     continue;
//                 }
//             }
//             depth--;
//         }
//         release_lock_host();
//         return result;
//     }

//     // Free functions wrapped with lock.
//     __device__ void free(void *ptr)
//     {
//         if (ptr == nullptr)
//             return;
//         acquire_lock();
//         int totalNodes = 1 << (maxOrder + 1);
//         for (int i = 1; i < totalNodes; i++) {
//             int level = log2_int(i);
//             if (tree[i] == ALLOCATED) {
//                 void *start = block_ptr(i, level);
//                 if (start == ptr) {
//                     tree[i] = FREE;
//                     // Coalesce upward.
//                     int current = i;
//                     while (current > 1) {
//                         int parent = current / 2;
//                         int left = parent * 2;
//                         int right = left + 1;
//                         if (left < treeSize && right < treeSize &&
//                             tree[left] == FREE && tree[right] == FREE)
//                         {
//                             tree[parent] = FREE;
//                             current = parent;
//                         }
//                         else {
//                             break;
//                         }
//                     }
//                     break;
//                 }
//             }
//         }
//         release_lock();
//     }

//     __host__ void free_with_lock_host(void *ptr)
//     {
//         if (ptr == nullptr)
//             return;
//         acquire_lock_host();
//         int totalNodes = 1 << (maxOrder + 1);
//         for (int i = 1; i < totalNodes; i++) {
//             int level = log2_int(i);
//             if (tree[i] == ALLOCATED) {
//                 void *start = block_ptr(i, level);
//                 if (start == ptr) {
//                     tree[i] = FREE;
//                     int current = i;
//                     while (current > 1) {
//                         int parent = current / 2;
//                         int left = parent * 2;
//                         int right = left + 1;
//                         if (left < treeSize && right < treeSize &&
//                             tree[left] == FREE && tree[right] == FREE)
//                         {
//                             tree[parent] = FREE;
//                             current = parent;
//                         }
//                         else {
//                             break;
//                         }
//                     }
//                     break;
//                 }
//             }
//         }
//         release_lock_host();
//     }
// };

// // Creates and initializes a CudaMemoryManager.
// __host__ CudaMemoryManager *createCudaMemoryManager(uint64_t totalSize, uint64_t minBlockSize)
// {
//     CudaMemoryManager *allocator = nullptr;
//     cudaError_t err = cudaMallocManaged(&allocator, sizeof(CudaMemoryManager));
//     if (err != cudaSuccess)
//     {
//         printf("Error allocating CudaMemoryManager: %s\n", cudaGetErrorString(err));
//         return nullptr;
//     }

//     int maxOrder = log2_int(totalSize / minBlockSize);
//     std::cout << "Max Order: " << maxOrder << std::endl;

//     void *memory = nullptr;
//     err = cudaMallocManaged(&memory, totalSize);
//     if (err != cudaSuccess)
//     {
//         printf("Error allocating memory pool: %s\n", cudaGetErrorString(err));
//         cudaFree(allocator);
//         return nullptr;
//     }

//     allocator->totalSize = totalSize;
//     allocator->memory = memory;
//     allocator->minBlockSize = minBlockSize;
//     allocator->maxOrder = maxOrder;
//     allocator->lock = 0;

//     // For 1-based indexing.
//     allocator->treeSize = 1 << (maxOrder + 1);
//     std::cout << "Tree Size: " << allocator->treeSize << std::endl;
//     err = cudaMallocManaged(&allocator->tree, allocator->treeSize * sizeof(BlockState));
//     if (err != cudaSuccess)
//     {
//         printf("Error allocating binary tree: %s\n", cudaGetErrorString(err));
//         cudaFree(allocator->memory);
//         cudaFree(allocator);
//         return nullptr;
//     }

//     // Initialize all nodes to FREE.
//     memset(allocator->tree, 0, allocator->treeSize * sizeof(BlockState));

//     // Precompute block size lookup table for levels 0..maxOrder.
//     for (int level = 0; level <= maxOrder; level++) {
//         // At level L: block size = minBlockSize << (maxOrder - L)
//         allocator->blockSizeTable[level] = ((uint64_t)minBlockSize << (maxOrder - level));
//     }

//     return allocator;
// }
