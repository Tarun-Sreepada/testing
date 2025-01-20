#ifndef CUDA_MINING_H
#define CUDA_MINING_H

#include <stdint.h>
#include "io.h"
#include "allocator.cuh"

// Kernel that starts the mining process. It is expected to be launched from host code.
__global__ void mine(BumpAllocator *alloc, uint32_t *base_pattern,
                     Item *items, uint32_t nItems,
                     uint32_t *start, uint32_t *end, uint32_t *utility, uint32_t nTransactions,
                     uint32_t *primary, uint32_t maxItem,
                     uint32_t minUtil, uint32_t *pattern_counter);

// A simple helper kernel to print a pattern stored in device memory.
__global__ void print_pattern(uint32_t *pattern);

#endif // CUDA_MINING_H
