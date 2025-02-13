#pragma once
#include <cstdint>
#include "database.cuh"

#define CAPACITY 4 * 1024


struct WorkItem {

    int *pattern;
    int pattern_length;

    int utility;

    Database *db;

    int max_item;

    int primary;

    int *work_done;
    int work_count;

};


struct AtomicWorkStack {
    WorkItem items[CAPACITY];

    // 'top' is the index of the next free slot (i.e., valid items are in indices 0..top-1).
    unsigned int top;

    
    // Initialize the stack on the device.
    __device__ __host__ void init() {
        top = 0;
    }
    
    // Device push using an atomicCAS loop.
    __device__ bool push(WorkItem item) {

        int ret = atomicAdd(&top, 1);
        if (ret >= CAPACITY) {
            return false;
        }
        items[ret] = item;

        return true;
    }
    
    // Initialize the stack on the host.
    __host__ void init_host() {
        top = 0;
    }
    
};
